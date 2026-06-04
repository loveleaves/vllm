# vLLM Reasoning Outputs 深度设计文档

> **目标读者**：不熟悉 vLLM serving 层的开发者。读完本文后，你应能：
> 1. 理解 reasoning output 的设计动机与完整实现原理
> 2. 在旧版 vLLM 上从零实现同等功能
> 3. 掌握 vLLM OpenAI 服务层的核心数据流与扩展点
> 4. 具备独立为其他推理模型（如 QwQ、o1-style 模型）添加 parser 的能力

---

## 目录

1. [为什么需要 Reasoning Output](#1-为什么需要-reasoning-output)
2. [`<think>` 标签的技术基础](#2-think-标签的技术基础)
3. [vLLM OpenAI 服务端数据流总览](#3-vllm-openai-服务端数据流总览)
4. [Reasoning Output 功能设计](#4-reasoning-output-功能设计)
5. [逐层源码精读](#5-逐层源码精读)
6. [演进历史与关键设计决策](#6-演进历史与关键设计决策)

---

## 1. 为什么需要 Reasoning Output

### 1.1 推理模型的"思维链"输出

2024-2025 年涌现了一类新范式的大模型——**推理模型**（Reasoning Model）。与传统模型直接生成答案不同，这类模型在给出最终答案前会先输出一段完整的思考过程：

```
用户: 9.11 和 9.9 哪个大？

模型原始输出:
<think>
我需要比较 9.11 和 9.9。
9.11 = 9 + 0.11
9.9  = 9 + 0.90
0.11 < 0.90，所以 9.11 < 9.9。
</think>
9.9 更大。
```

代表模型：DeepSeek-R1 系列、Qwen3 系列（`enable_thinking=True` 模式）、QwQ。

思考内容被包裹在 `<think>…</think>` 标签内，最终答案跟在标签之后。

### 1.2 原始 vLLM serving 层的问题

vLLM 的 OpenAI 兼容服务层（基线 `fbb5bd4ce`，v0.6.6）将模型所有输出不加区分地放入 `message.content`：

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "<think>\n我需要比较 9.11 和 9.9...\n</think>\n9.9 更大。"
    }
  }]
}
```

**三个核心痛点**：

| 痛点 | 描述 |
|------|------|
| **UI 污染** | 客户端如果直接渲染 `content`，思考标签会原样展示给用户 |
| **二次解析负担** | 每个客户端都要自己写正则提取 `<think>` 块，重复劳动且容易出错 |
| **API 语义不清** | OpenAI API 约定 `content` 是最终答案，混入思考过程破坏了语义契约 |

### 1.3 目标 API 形态

本功能实现后，客户端接收到的响应如下：

**非流式（`stream=false`）**：

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "reasoning_content": "我需要比较 9.11 和 9.9...\n0.11 < 0.90，所以 9.11 < 9.9。",
      "content": "9.9 更大。"
    }
  }]
}
```

**流式（`stream=true`）**：

```
data: {"choices":[{"delta":{"reasoning_content":"我需要比较"},"finish_reason":null}]}
data: {"choices":[{"delta":{"reasoning_content":" 9.11 和 9.9"},"finish_reason":null}]}
...（若干 reasoning delta）
data: {"choices":[{"delta":{"content":"9.9 更大。"},"finish_reason":null}]}
data: {"choices":[{"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

规则：
- `reasoning_content` 非空 ↔ 模型输出中存在 `<think>…</think>` 块
- `content` 不含任何 `<think>`/`</think>` 标签
- 流式时，所有 `reasoning_content` delta 先于所有 `content` delta（与模型生成顺序一致）
- 若模型未开启 thinking 模式（如 Qwen3 的 `enable_thinking=false`），`reasoning_content` 为 `null`

---

## 2. `<think>` 标签的技术基础

### 2.1 特殊 token 的本质

理解 streaming 解析的关键在于：`<think>` 和 `</think>` 在 Qwen3/DeepSeek-R1 的词表中是**单个 token**，而非多字符序列。

以 Qwen3 tokenizer 为例：

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

tok.encode("<think>", add_special_tokens=False)  # → [151667]
tok.encode("</think>", add_special_tokens=False) # → [151668]

# 对比普通文本的分词
tok.encode("hello", add_special_tokens=False)   # → [14990]（单 token）
tok.encode("<think", add_special_tokens=False)  # → [151667] 还是多token？
# 这取决于是否被注册为 special token
```

通过 `tokenizer.get_vocab()` 可以查到：

```python
vocab = tok.get_vocab()
vocab["<think>"]   # 151667
vocab["</think>"]  # 151668
```

**为什么这很重要**：streaming 时模型逐 token 输出，每个 delta 只包含少量 token。如果 `<think>` 是单一 token，我们可以通过检查 `delta_token_ids` 来精确识别"进入思考模式"和"退出思考模式"的边界，而无需对文本做字符串搜索。

### 2.2 文本方法 vs token-ID 方法

两种实现路线的对比：

| 维度 | 文本方法 | Token-ID 方法 |
|------|---------|--------------|
| **实现** | 在 `delta_text` 中搜索 `"<think>"` 子串 | 在 `delta_token_ids` 中检查特定 ID |
| **可靠性** | 若分词器将 `<` 和 `t` 分开 token 化，跨 delta 时标签会被拆散，可能漏检 | 标签是单一 token，永远是原子操作，不会跨 delta 拆散 |
| **适用范围** | 任何 tokenizer（即使 `<think>` 不是 special token） | 要求词表中存在单 token 形式的 `<think>`/`</think>` |
| **实现复杂度** | 需处理跨 delta 的部分匹配（状态机复杂） | 无跨 delta 问题，逻辑简单 |

**本实现选择 token-ID 方法**，并在构造时检查词表，若不满足条件立即报错（fail-fast）：

```python
start_id = vocab.get("<think>")
end_id = vocab.get("</think>")
if start_id is None or end_id is None:
    raise RuntimeError("Tokenizer does not have '<think>' or '</think>' as single tokens.")
```

### 2.3 Qwen3 的 thinking 开关

Qwen3 模型通过聊天模板控制是否输出 `<think>` 块，而非通过推理时参数：

```python
# 开启 thinking（发给 apply_chat_template 的 kwargs）
{"enable_thinking": True}   # 模型会输出 <think>…</think>

# 关闭 thinking
{"enable_thinking": False}  # 模型直接输出答案，无 <think> 块
```

当 `enable_thinking=False` 时，模型不产生任何 `<think>` token，parser 检测不到 `think_start_token_id`，走"无思考块"分支，`reasoning_content` 为 `null`。**Reasoning parser 对此天然兼容**，无需特殊处理。

### 2.4 vLLM streaming 的 delta 结构

vLLM engine 每次向 serving 层吐出一个 `RequestOutput`，其中 `output.text` 是**增量文本**，`output.token_ids` 是**增量 token ID 列表**（通常是 1-4 个 token）。

```
时间轴:
  t=0: text="<think>", token_ids=[151667]          ← 思考开始
  t=1: text="\n分析",   token_ids=[198, 10947]      ← 思考内容
  t=2: text="...</think>", token_ids=[..., 151668]  ← 思考结束（可能含尾部内容）
  t=3: text="答案是",   token_ids=[...]             ← 正文开始
```

Parser 在每个 delta 到来时被调用一次，接收的参数包含：
- `previous_text`：截至上一 delta 的累积文本
- `current_text`：截至本 delta 的累积文本（= previous + delta）
- `delta_text`：本 delta 新增的文本
- `previous_token_ids`：截至上一 delta 的累积 token ID 列表
- `current_token_ids`：截至本 delta 的累积 token ID 列表
- `delta_token_ids`：本 delta 新增的 token ID 列表

通过检查 `previous_token_ids` 和 `delta_token_ids` 中是否包含 `think_start_token_id`/`think_end_token_id`，可以确定当前所处阶段。

---

## 3. vLLM OpenAI 服务端数据流总览

### 3.1 整体架构

vLLM 的 OpenAI 服务端由以下层级构成（reasoning output 修改主要集中在 **serving 层**）：

```
┌─────────────────────────────────────────────────────────────────────┐
│  HTTP 层（FastAPI）                                                  │
│  vllm/entrypoints/openai/api_server.py                              │
│  职责：路由注册、中间件、生命周期管理；透传 reasoning_parser 参数    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│  Serving 层 ★（本功能主战场）                                        │
│  vllm/entrypoints/openai/serving_chat.py  ← OpenAIServingChat       │
│  职责：请求预处理、engine 调用、响应构建；                           │
│         新增：通过 ReasoningParser 拆分 reasoning/content           │
├─────────────────────────────────────────────────────────────────────┤
│  协议层 ★（本功能修改）                                              │
│  vllm/entrypoints/openai/protocol.py                                │
│  职责：Pydantic 模型定义（请求/响应 schema）；                       │
│         新增：ChatMessage.reasoning_content, DeltaMessage.reasoning_content │
├─────────────────────────────────────────────────────────────────────┤
│  解析器插件层 ★（本功能新建）                                         │
│  vllm/entrypoints/openai/reasoning_parsers/                         │
│    abs_reasoning_parsers.py   ← ReasoningParser 基类 + Manager 注册表 │
│    deepseek_r1_reasoning_parser.py  ← DeepSeekR1/Qwen3 实现         │
│  职责：从模型文本输出中提取 reasoning/content；可插拔                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│  CLI 配置层 ★（本功能修改）                                           │
│  vllm/entrypoints/openai/cli_args.py                                │
│  职责：命令行参数解析与校验；                                        │
│         新增：--enable-reasoning, --reasoning-parser                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│  Engine 层（不修改）                                                  │
│  vllm/engine/async_llm_engine.py                                    │
│  职责：调度、KV cache 管理、模型执行；对 serving 层透明输出 tokens   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│  模型层（本功能新增 Qwen3）                                           │
│  vllm/model_executor/models/qwen3.py                                │
│  职责：Qwen3 模型推理（QK-norm + no bias）；vLLM v0.6.6 缺少此文件  │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 请求生命周期

以一个带 reasoning 的非流式请求为例，完整数据流如下：

```
客户端 POST /v1/chat/completions
    │
    ▼
api_server.py: create_chat_completion(request)
    │  将 request 转交给 OpenAIServingChat
    ▼
serving_chat.py: create_chat_completion()
    │  1. 预处理：消息 → prompt token ids（chat template）
    │  2. 构造 SamplingParams
    │  3. 调用 engine_client.generate()
    ▼
AsyncLLMEngine（不变）
    │  4. 调度、KV cache 分配、模型前向
    │  5. 逐 token 采样，累积 RequestOutput
    ▼
serving_chat.py: chat_completion_full_generator()
    │  6. 等待所有 token 生成完毕（non-streaming）
    │  7. ★ 检查 self.enable_reasoning
    │     → 是：调用 rp.extract_reasoning_content(output.text)
    │            得到 (reasoning_content, content)
    │            构造 ChatMessage(reasoning_content=..., content=...)
    │     → 否：构造 ChatMessage(content=output.text)
    ▼
api_server.py: 序列化 ChatCompletionResponse → JSON
    │  8. Pydantic model_dump_json(exclude_unset=True)
    │     若 reasoning_content=None，该字段被排除（exclude_unset）
    ▼
客户端接收响应
```

流式请求在步骤 6-7 处有所不同：每个 delta 到来时立即调用 `extract_reasoning_content_streaming()`，根据返回的 `DeltaMessage` 决定发 `reasoning_content` chunk 还是 `content` chunk，或直接跳过（控制 token）。

### 3.3 关键设计约束

**`exclude_unset=True` 的语义**：Pydantic 的 `model_dump_json(exclude_unset=True)` 只序列化显式赋值过的字段。这意味着当 `reasoning_content=None`（Python 的 `None`）时，如果该字段从未被赋值，则不会出现在 JSON 输出中；若赋值为 `None`，则会输出 `"reasoning_content": null`。

实现中始终显式传递 `reasoning_content` 参数：

```python
# reasoning_content=None 会出现在 JSON 中（客户端知道模型未思考）
ChatMessage(role=role, content=content, reasoning_content=reasoning_content)

# 未传 reasoning_content 时，该字段完全不存在于 JSON（旧行为兼容）
ChatMessage(role=role, content=output.text)
```

---

## 4. Reasoning Output 功能设计

### 4.1 三个路径分支

| 路径 | 触发条件 | 处理方式 |
|------|---------|---------|
| **非流式 + reasoning** | `stream=false` + `enable_reasoning=true` | 等待全量输出，一次调用 `extract_reasoning_content()`，返回 `ChatMessage` |
| **流式 + reasoning** | `stream=true` + `enable_reasoning=true` | 每个 delta 调用 `extract_reasoning_content_streaming()`，返回 `DeltaMessage` 或 `None` |
| **无 reasoning（默认）** | `enable_reasoning=false` | 走原有逻辑，`content=output.text`，无 `reasoning_content` 字段 |

### 4.2 流式解析状态机

流式解析器**不维护显式状态变量**，而是每次通过扫描累积的 `previous_token_ids` 推导当前所处阶段。完整决策树：

```
每个 delta 到来时执行以下判断：

delta_has_start = (think_start_id ∈ delta_token_ids)
delta_has_end   = (think_end_id   ∈ delta_token_ids)
prev_has_start  = (think_start_id ∈ previous_token_ids)
prev_has_end    = (think_end_id   ∈ previous_token_ids)

① delta_has_start AND NOT delta_has_end
      → 这是 <think> token，抑制（return None）

② delta_has_end AND NOT prev_has_end
      → 这是 </think> token（首次出现）
      → 切割 delta_text：
          trailing = delta_text.split("</think>", 1)[1]
          if trailing:  return DeltaMessage(content=trailing)   # 尾部正文
          else:         return None                             # 纯控制 token

③ prev_has_start AND NOT prev_has_end
      → 在思考块内部
      → return DeltaMessage(reasoning_content=delta_text)

④ prev_has_end
      → 思考块已结束
      → return DeltaMessage(content=delta_text)

⑤ 其他（没有 <think> 也没有 </think>）
      → 普通内容（thinking 模式未开启）
      → return DeltaMessage(content=delta_text)
```

**`None` 的语义**：returning `None` 告知 `serving_chat.py` 跳过本 delta，不向客户端发送任何 chunk。这是与 `ToolParser` 一致的约定，用于抑制不应暴露给用户的控制 token。

### 4.3 非流式解析逻辑

```
model_output = "<think>reasoning...</think>final answer"

start = model_output.find("<think>")    # 找到 → 7
end   = model_output.find("</think>")  # 找到 → ...

case 1: start == -1
    → 无思考块：return (None, model_output)

case 2: start != -1, end == -1
    → 思考块未闭合（生成被截断）：
    → reasoning = model_output[start+7:]
    → return (reasoning or None, None)

case 3: start != -1, end != -1
    → 完整思考块：
    → reasoning = model_output[start+7 : end]
    → content   = model_output[end+8 :]
    → return (reasoning or None, content or None)
```

空字符串被转换为 `None`，避免 `"reasoning_content": ""` 这样语义模糊的输出。

### 4.4 CLI 参数设计

新增两个启动参数，模仿 `--enable-auto-tool-choice` / `--tool-call-parser` 的对称设计：

```
--enable-reasoning          (store_true)
    启用 reasoning content 解析。需配合 --reasoning-parser 使用。

--reasoning-parser NAME     (str)
    指定使用哪个 parser。当前注册值：deepseek_r1, qwen3
    （两者指向同一实现，仅注册名称不同）
```

**校验规则**（`validate_parsed_serve_args`）：

```
--enable-reasoning 且无 --reasoning-parser  → TypeError
--enable-reasoning 且 --enable-auto-tool-choice → TypeError（互斥）
```

互斥的原因：两者都需要维护 `previous_texts` / `all_previous_token_ids` 状态，且两者对 delta 的路由逻辑互斥（tool_choice_auto 与 reasoning 各占一个 `elif` 分支）。合并支持在架构上可行但当前版本未实现。

### 4.5 协议扩展设计

`protocol.py` 中只需修改两个类，各加一个字段：

```python
class ChatMessage(OpenAIBaseModel):     # 非流式响应消息
    role: str
    content: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    reasoning_content: Optional[str] = None   # ← 新增，行 1207

class DeltaMessage(OpenAIBaseModel):    # 流式增量消息
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: List[DeltaToolCall] = Field(default_factory=list)
    reasoning_content: Optional[str] = None   # ← 新增，行 1248
```

`OpenAIBaseModel` 的 `model_config = ConfigDict(extra="allow")` 意味着即使客户端传入了未知字段也不会报错，新增字段对旧客户端完全透明。

---

## 5. 逐层源码精读

### 5.1 Layer 1：协议定义（`protocol.py`）

文件路径：`vllm/entrypoints/openai/protocol.py`

这是变更最小但影响面最大的一层。整个 reasoning output 功能的 API 合约由此处两个字段定义。

**`ChatMessage`（第 1203 行）**：

```python
class ChatMessage(OpenAIBaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    reasoning_content: Optional[str] = None  # ← 新增
```

`ChatMessage` 用于非流式响应的 `choices[i].message`。字段声明为 `Optional[str] = None` 而非 `str`，原因：
- 绝大多数请求不使用 reasoning 功能，`reasoning_content` 应缺席而非为空字符串
- `model_dump_json(exclude_unset=True)` 确保未赋值字段不出现在 JSON 中

**`DeltaMessage`（第 1244 行）**：

```python
class DeltaMessage(OpenAIBaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: List[DeltaToolCall] = Field(default_factory=list)
    reasoning_content: Optional[str] = None  # ← 新增
```

`DeltaMessage` 用于流式响应的每个 SSE chunk 的 `choices[i].delta`。每个 delta 只会填充 `reasoning_content` 和 `content` 中的一个（或两者都不填，只有 `role`），实现阶段隔离。

**Pydantic `exclude_unset` 的工作机制**：

```python
# 赋值了 reasoning_content 的 DeltaMessage
msg = DeltaMessage(reasoning_content="在思考")
msg.model_dump_json(exclude_unset=True)
# → '{"reasoning_content":"在思考"}'  ← content 字段缺席

# 未赋值 reasoning_content 的 DeltaMessage
msg = DeltaMessage(content="答案")
msg.model_dump_json(exclude_unset=True)
# → '{"content":"答案"}'  ← reasoning_content 字段缺席
```

这是保持流式输出简洁的关键：每个 chunk 只携带必要字段。

### 5.2 Layer 2a：抽象基类（`abs_reasoning_parsers.py`）

文件路径：`vllm/entrypoints/openai/reasoning_parsers/abs_reasoning_parsers.py`

这一层定义了**可插拔接口**和**注册表**，是整个 parser 插件系统的骨架。

**`ReasoningParser` 基类**：

```python
class ReasoningParser:
    def __init__(self, tokenizer: AnyTokenizer) -> None:
        self.model_tokenizer = tokenizer

    @cached_property
    def vocab(self) -> Dict[str, int]:
        return self.model_tokenizer.get_vocab()
        # cached_property：词表只查询一次，后续调用直接返回缓存
        # 对于大词表（Qwen3 约 152k token）有明显性能收益
```

`@cached_property` 的选择：`get_vocab()` 在 HuggingFace tokenizer 中会构建一个完整的 `{str: int}` 字典（对 Qwen3 词表 ~152K 条目约需数毫秒）。在流式场景下 parser 实例在请求生命周期内复用，`cached_property` 确保只执行一次。

**两个抽象方法的接口设计**：

```python
def extract_reasoning_content(
    self,
    model_output: str,           # 全量输出文本
    request: ChatCompletionRequest,  # 原始请求（供子类访问 model_params 等）
) -> Tuple[Optional[str], Optional[str]]:
    # 返回 (reasoning_content, content)
    raise NotImplementedError

def extract_reasoning_content_streaming(
    self,
    previous_text: str,
    current_text: str,
    delta_text: str,
    previous_token_ids: Sequence[int],
    current_token_ids: Sequence[int],
    delta_token_ids: Sequence[int],
) -> Optional[DeltaMessage]:
    # 返回 DeltaMessage(reasoning_content=...) 或 DeltaMessage(content=...) 或 None
    raise NotImplementedError
```

接口设计与 `ToolParser.extract_tool_calls_streaming` 高度对称，降低学习成本。`request` 参数留给子类使用（例如未来可以读取 `request.chat_template_kwargs` 中的 `enable_thinking` 开关）。

**`ReasoningParserManager` 注册表**：

```python
class ReasoningParserManager:
    reasoning_parsers: Dict[str, Type[ReasoningParser]] = {}
    # 类变量（非实例变量），全局单例字典，进程内共享

    @classmethod
    def get_reasoning_parser(cls, name: str) -> Type[ReasoningParser]:
        parser = cls.reasoning_parsers.get(name)
        if parser is None:
            raise KeyError(
                f"Reasoning parser '{name}' is not registered. "
                f"Available: {list(cls.reasoning_parsers.keys())}")
        return parser
        # 注意：返回的是 class 本身，不是实例
        # 调用方负责在每个请求时实例化：self.reasoning_parser(tokenizer)

    @classmethod
    def register_module(
        cls,
        name=None,          # str | List[str] | None
        force=True,         # 允许覆盖已有注册
        module=None,        # 直接传入 class（不作装饰器用）
    ) -> Union[Type, Callable]:
        def _register(parser_cls):
            names = [name] if isinstance(name, str) else (name or [])
            if not names:
                names = [parser_cls.__name__]  # 默认用类名
            for n in names:
                if n in cls.reasoning_parsers and not force:
                    raise KeyError(f"Reasoning parser '{n}' already registered.")
                cls.reasoning_parsers[n] = parser_cls
            return parser_cls   # 装饰器必须返回原类

        if module is not None:
            return _register(module)   # 直接调用模式
        return _register               # 装饰器模式
```

**注册时机**：`reasoning_parsers/__init__.py` 在模块导入时执行：

```python
from .abs_reasoning_parsers import ReasoningParser, ReasoningParserManager
from .deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
# 导入上面这行时，模块级的 @register_module 装饰器立即执行
# reasoning_parsers["deepseek_r1"] 和 reasoning_parsers["qwen3"] 此时注册完毕
```

`cli_args.py` 在模块顶部 `from vllm.entrypoints.openai.reasoning_parsers import ReasoningParserManager`，因此在解析命令行参数时 `valid_reasoning_parsers` 已有值，可正确填充 `--reasoning-parser` 的 metavar。

### 5.3 Layer 2b：具体实现（`deepseek_r1_reasoning_parser.py`）

文件路径：`vllm/entrypoints/openai/reasoning_parsers/deepseek_r1_reasoning_parser.py`

**注册**：

```python
@ReasoningParserManager.register_module(["deepseek_r1", "qwen3"])
class DeepSeekR1ReasoningParser(ReasoningParser):
```

同一实现注册了两个名称，原因：DeepSeek-R1 和 Qwen3 使用完全相同的 `<think>`/`</think>` 标签约定，代码无需区分。未来若出现差异（如不同 token ID 或不同标签字符串），可以分别注册不同子类。

**初始化**：

```python
def __init__(self, tokenizer: AnyTokenizer) -> None:
    super().__init__(tokenizer)
    vocab = self.vocab  # 触发 cached_property，查询词表
    start_id = vocab.get(self.THINK_START)   # "<think>"
    end_id   = vocab.get(self.THINK_END)     # "</think>"
    if start_id is None or end_id is None:
        raise RuntimeError(
            f"Tokenizer does not have '{self.THINK_START}' or "
            f"'{self.THINK_END}' as single tokens. "
            "Cannot use DeepSeekR1ReasoningParser with this tokenizer.")
    self.think_start_token_id: int = start_id   # Qwen3: 151667
    self.think_end_token_id: int = end_id        # Qwen3: 151668
```

`RuntimeError` 在**实例化**时抛出（而非 `get_reasoning_parser()` 时），因为 tokenizer 在服务启动时才可用，注册时无法检查。`serving_chat.py` 的调用方捕获此异常并转换为 HTTP 500 响应。

**非流式实现细节**：

```python
def extract_reasoning_content(self, model_output, request):
    start = model_output.find(self.THINK_START)   # "<think>"
    end   = model_output.find(self.THINK_END)     # "</think>"

    if start == -1:
        return None, model_output          # case 1: 无思考块

    if end == -1:
        # case 2: 思考块未闭合（max_tokens 截断）
        reasoning = model_output[start + len(self.THINK_START):]
        return reasoning or None, None
        # "reasoning or None"：若 <think> 后紧跟截断，reasoning="" 转为 None

    # case 3: 完整思考块
    reasoning = model_output[start + len(self.THINK_START) : end]
    content   = model_output[end + len(self.THINK_END):]
    return reasoning or None, content or None
```

注意 `find()` 找到的是**第一个**出现位置，多个 `<think>` 块时只处理第一个（已知局限，见设计决策节）。

**流式实现细节**（完整注解）：

```python
def extract_reasoning_content_streaming(
    self, previous_text, current_text, delta_text,
    previous_token_ids, current_token_ids, delta_token_ids,
):
    start_id = self.think_start_token_id
    end_id   = self.think_end_token_id

    # 四个布尔值覆盖所有状态组合
    delta_has_start = start_id in delta_token_ids
    delta_has_end   = end_id   in delta_token_ids
    prev_has_start  = start_id in previous_token_ids
    prev_has_end    = end_id   in previous_token_ids

    # ─── 分支①：delta 是 <think> token ───────────────────────────────
    if delta_has_start and not delta_has_end:
        return None          # 抑制开始控制 token

    # ─── 分支②：delta 包含 </think> token（首次）─────────────────────
    if delta_has_end and not prev_has_end:
        # 切割：delta_text = "...推理内容</think>正文开头"
        # split 最多切一刀，取右半部分
        after = delta_text.split(self.THINK_END, 1)
        trailing = after[1] if len(after) > 1 else ""
        if trailing:
            return DeltaMessage(content=trailing)  # 同 delta 里有正文
        return None  # 纯 </think> token，抑制

    # ─── 分支③：在思考块内部 ─────────────────────────────────────────
    if prev_has_start and not prev_has_end:
        return DeltaMessage(reasoning_content=delta_text)

    # ─── 分支④：思考块已结束 ─────────────────────────────────────────
    if prev_has_end:
        return DeltaMessage(content=delta_text)

    # ─── 分支⑤：无思考块（thinking 未开启）──────────────────────────
    return DeltaMessage(content=delta_text)
```

**分支②的"同 delta 切割"**：当模型在单个推理步骤中同时生成 `</think>` token 和少量正文 token 时（罕见但合法），`delta_text` 形如 `"</think>答案"`. `split("</think>", 1)` 返回 `["", "答案"]`，`trailing="答案"`，以 `DeltaMessage(content="答案")` 发出。注意此时**思考内容已在之前的 delta 中全部发出**，不会丢失。

### 5.4 Layer 3：Serving 层（`serving_chat.py`）

文件路径：`vllm/entrypoints/openai/serving_chat.py`

这是改动最多的文件，需要理解其在初始化、流式路径、非流式路径三处的变更。

**`__init__` 初始化（第 94-105 行）**：

```python
# set up reasoning parser
self.enable_reasoning: bool = enable_reasoning
self.reasoning_parser: Optional[Callable[[AnyTokenizer], ReasoningParser]] = None
if self.enable_reasoning:
    try:
        self.reasoning_parser = ReasoningParserManager \
            .get_reasoning_parser(reasoning_parser)
        # 存储的是 class（可调用），不是实例
        # 延迟到每个请求时再实例化（因为需要 tokenizer）
    except Exception as e:
        raise TypeError(
            f"Error: --enable-reasoning requires "
            f"--reasoning-parser '{reasoning_parser}' which has not been registered"
        ) from e
```

`self.reasoning_parser` 存储的是**类本身**（如 `DeepSeekR1ReasoningParser`），而非实例。原因：tokenizer 在请求时才通过 `await engine_client.get_tokenizer()` 获取，初始化时不可用。每次请求按需实例化：

```python
rp = self.reasoning_parser(tokenizer)   # 等价于 DeepSeekR1ReasoningParser(tokenizer)
```

**流式路径的状态初始化（第 307-316 行）**：

```python
should_stream_with_reasoning = (
    self._should_stream_with_reasoning_parsing(request))

# previous_texts / all_previous_token_ids：
# - tool_choice_auto 需要（用于 ToolParser）
# - reasoning 也需要（用于 ReasoningParser）
# 条件 OR，避免非必要情况分配这两个列表
if tool_choice_auto or should_stream_with_reasoning:
    previous_texts = [""] * num_choices
    all_previous_token_ids = [[]] * num_choices
else:
    previous_texts, all_previous_token_ids = None, None
```

**流式路径的 delta 路由（第 462-525 行）**，四个互斥分支：

```
if tool_choice_function_name:          # 指定工具名（named tool choice）
    ...
elif tool_choice_auto:                 # 自动工具选择
    delta_message = tool_parser.extract_tool_calls_streaming(...)
elif should_stream_with_reasoning:     # ← 新增分支
    delta_message = reasoning_parser_i.extract_reasoning_content_streaming(
        previous_text=previous_text,
        current_text=current_text,
        delta_text=delta_text,
        previous_token_ids=previous_token_ids,
        current_token_ids=current_token_ids,
        delta_token_ids=output.token_ids,
    )
    previous_texts[i] = current_text          # 更新累积文本
    all_previous_token_ids[i] = current_token_ids  # 更新累积 IDs
else:                                  # 普通文本 delta
    delta_message = DeltaMessage(content=delta_text)
```

当 `delta_message is None` 时，`serving_chat.py` 直接 `continue`，跳过本次 chunk 发送：

```python
if delta_message is None:
    continue   # 控制 token，不向客户端发送任何内容
```

**非流式路径（第 713-725 行）**：

```python
# if reasoning parsing is enabled, extract think block first
if self.enable_reasoning and self.reasoning_parser:
    try:
        rp = self.reasoning_parser(tokenizer)
    except RuntimeError as e:
        return self.create_error_response(str(e))
    reasoning_content, content = rp.extract_reasoning_content(
        output.text, request=request)
    message = ChatMessage(
        role=role,
        content=content,
        reasoning_content=reasoning_content,
    )
# elif ... （原有 tool_choice、普通消息等分支）
```

非流式路径将 reasoning 分支放在**最高优先级**（所有 elif 之前），这是因为 reasoning 和 tool choice 在本实现中是互斥的（CLI 层已校验），不存在同时为真的情况，但语义上 reasoning 的提取逻辑比 tool_choice 更基础（先提取思考内容，再判断是否有工具调用——未来扩展时的预留位置）。

**辅助方法**：

```python
def _should_stream_with_reasoning_parsing(
    self, request: ChatCompletionRequest
) -> bool:
    return self.enable_reasoning and self.reasoning_parser is not None
```

抽出为方法的原因：与 `_should_stream_with_auto_tool_parsing` 保持对称，且对称设计使得 `tool_choice_auto or should_stream_with_reasoning` 的布尔组合易于阅读。

### 5.5 Layer 4：CLI 与参数传递（`cli_args.py` + `api_server.py`）

**`cli_args.py`** 的两处新增：

```python
# ① 参数声明（紧跟 --tool-parser-plugin 之后）
parser.add_argument(
    "--enable-reasoning",
    action="store_true",
    default=False,
    help="Enable reasoning content parsing for models that produce "
         "<think>…</think> blocks (e.g. DeepSeek-R1, Qwen3). "
         "Use ``--reasoning-parser`` to specify the parser.")

valid_reasoning_parsers = ReasoningParserManager.reasoning_parsers.keys()
parser.add_argument(
    "--reasoning-parser",
    type=str,
    metavar="{" + ",".join(valid_reasoning_parsers) + "}",
    default=None,
    help="Select the reasoning parser for models that emit "
         "<think>…</think> blocks. Required when ``--enable-reasoning`` is set.")

# ② 校验逻辑（validate_parsed_serve_args 函数内）
if args.enable_reasoning and not args.reasoning_parser:
    raise TypeError("Error: --enable-reasoning requires --reasoning-parser")
if args.enable_reasoning and args.enable_auto_tool_choice:
    raise TypeError(
        "Error: --enable-reasoning and --enable-auto-tool-choice are "
        "mutually exclusive")
```

**`api_server.py`** 的参数透传：

```python
state.openai_serving_chat = OpenAIServingChat(
    engine_client,
    model_config,
    state.openai_serving_models,
    args.response_role,
    request_logger=request_logger,
    chat_template=args.chat_template,
    ...
    enable_auto_tools=args.enable_auto_tool_choice,
    tool_parser=args.tool_call_parser,
    enable_reasoning=args.enable_reasoning,     # ← 新增
    reasoning_parser=args.reasoning_parser,     # ← 新增
)
```

### 5.6 Layer 5：Qwen3 模型实现（`qwen3.py`）

文件路径：`vllm/model_executor/models/qwen3.py`

vLLM v0.6.6 缺少 Qwen3 支持（Qwen3 于 2025 年 4 月发布），需手动添加。这部分与 reasoning parsing 逻辑无关，但是端到端测试的前提。

**Qwen3 与 Qwen2 的架构差异**：

| 组件 | Qwen2 | Qwen3 |
|------|-------|-------|
| QKV projection bias | `bias=True` | `bias=False` |
| Q/K 归一化 | 无 | 每个 attention head 独立 RMSNorm（QK-norm） |
| 其他（MLP、FFN、词表嵌入等） | 相同 | 相同 |

**QK-norm 实现的关键挑战**：

vLLM 的 `RMSNorm` CUDA kernel（`ops.rms_norm`）期望输入为 2D 张量 `(num_tokens, hidden_size)`。而 Q/K 的形状是 `(num_tokens, num_heads * head_dim)`，对每个 head 独立归一化意味着归一化维度是 `head_dim`，而非 `num_heads * head_dim`。

错误做法（直接传入 3D 或错误 2D）：

```python
# 错误：kernel 对 num_tokens × (num_heads * head_dim) 整行做 norm
self.q_norm(q)  # q.shape = (T, num_heads * head_dim) — 错误！
```

正确做法（先 reshape 再 norm 再还原）：

```python
t = q.shape[0]  # num_tokens

# reshape: (T, num_heads * head_dim) → (T * num_heads, head_dim)
# 现在每行是一个 head 的向量，RMSNorm 对每行独立归一化
q = self.q_norm(
        q.reshape(t * self.num_heads, self.head_dim)
    ).reshape(t, self.q_size)
# 注意用 reshape 而非 view，因为 norm 输出可能非连续
```

**`@support_torch_compile` 与 `nn.Module.__init__` 的博弈**：

`Qwen3Model` 继承自 `Qwen2Model`，但如果直接调用 `super().__init__()`，`Qwen2Model.__init__` 会构建一批 `Qwen2DecoderLayer`，随后 `Qwen3Model.__init__` 再构建 `Qwen3DecoderLayer`，导致 KV cache 的 `Attention` 层被双重注册，名称冲突报错：

```
RuntimeError: Duplicate layer name: model.layers.0.self_attn.attn
```

解决方案：跳过 `Qwen2Model.__init__`，直接调用 `nn.Module.__init__(self)`，然后手动重建所有必要属性：

```python
class Qwen3Model(Qwen2Model):
    def __init__(self, *, vllm_config, prefix=""):
        nn.Module.__init__(self)   # 跳过 Qwen2Model.__init__
        # 手动构建：embed_tokens, layers(Qwen3DecoderLayer), norm
        ...
```

`@support_torch_compile` 装饰器直接作用于 `Qwen3Model`（而非 `Qwen2Model`），原因：装饰器在 `__init__` 中设置 `self.do_not_compile` 和 `self.vllm_config` 属性，绕过父类 `__init__` 后必须保证这些属性仍被正确设置。

---

## 6. 演进历史与关键设计决策

### 6.1 决策①：模仿 ToolParser 而非重新发明

**备选方案**：
- A）在 `serving_chat.py` 内直接内联字符串解析，无插件系统
- B）**仿照 `ToolParser` 建立独立插件层**（选择）
- C）复用 `ToolParser` 接口，将 reasoning 作为一种特殊 "tool"

选择 B 的原因：
- 不同推理模型使用不同的标签（DeepSeek-R1 和 Qwen3 用 `<think>`，未来可能有 `<reasoning>` 或其他变体），插件系统使得添加新 parser 只需新建一个文件
- `ToolParser` 已验证了这种设计在 vLLM 生产中可行
- C 的问题：`ToolParser` 接口返回 `DeltaToolCall`，与 `reasoning_content` 的 `str` 类型不匹配

**代价**：增加了约 100 行抽象代码（`abs_reasoning_parsers.py`）。对一个文件的小改动而言略显重，但为长期扩展性买单。

### 6.2 决策②：token-ID 方法 vs 文本方法（流式解析）

**备选方案**：
- A）**token-ID 状态检测**（选择）：通过 `think_start_token_id ∈ delta_token_ids` 推导阶段
- B）文本滑动窗口：在 `current_text` 中搜索 `"<think>"` 子串

选择 A 的原因：
- `<think>` 是 Qwen3/DeepSeek-R1 词表中的 special token，保证是单一 token，不会跨 delta 拆散
- 方案 B 需要处理部分匹配（如一个 delta 是 `"<thi"` 下一个是 `"nk>"`），需要更复杂的状态机
- 方案 A 的逻辑更紧凑，调试更容易

**代价**：需要在构造时验证词表（`vocab.get("<think>")`），若 tokenizer 不符合条件需 fail-fast。对不支持的 tokenizer（如 OPT 的原始词表）需先 `add_tokens`。

### 6.3 决策③：无显式状态变量的流式状态机

实现中**没有** `self.in_reasoning: bool = False` 之类的实例变量来跟踪状态，而是**每次从 `previous_token_ids` 重新推导**。

**原因**：
- `serving_chat.py` 每个请求创建一个新的 parser 实例，但 parser 实例在请求内跨 delta 调用。如果使用显式状态变量，需要确保实例生命周期与请求严格对应，否则状态污染。
- 无状态推导方式在调试时更直观（每次调用完全由参数决定，无隐藏状态）

**代价**：每次 delta 都扫描 `previous_token_ids`，最坏 O(n²)（n = 已生成 token 数）。对于超长思考链（>10K tokens），可改为显式状态变量优化。当前 Qwen3-1.7B 的思考链通常在 500-2000 tokens，性能可接受。

### 6.4 决策④：互斥 vs 并行（reasoning + tool choice）

当前实现中 `--enable-reasoning` 和 `--enable-auto-tool-choice` **互斥**。技术上并非不能同时支持，但：

1. **状态共享**：两者都需要 `previous_texts` / `all_previous_token_ids`（已通过 `or` 条件合并），这部分没有冲突
2. **路由冲突**：delta 路由的 `elif tool_choice_auto: ... elif should_stream_with_reasoning: ...` 是互斥分支，同时开启时只有一个生效（先检查 `tool_choice_auto`）
3. **语义冲突**：对于同时有工具调用和思考过程的模型（理论上存在），响应结构会很复杂（`reasoning_content` + `tool_calls` 同时出现）

CLI 互斥校验是在需求进一步明确前的**保守设计**，避免产生未经测试的行为。

### 6.5 决策⑤：Qwen3 模型的最小化实现

实现 Qwen3 时面临三个选项：
- A）从零实现（复制 Qwen2 全部代码，改动 QK-norm 相关部分）
- B）**最大复用 Qwen2 组件，仅实现差异部分**（选择）
- C）修改 Qwen2 加入条件分支（通过 config 判断是否 Qwen3）

选择 B 的原因：
- `Qwen2MLP`、`Qwen2Model.forward`、`Qwen2ForCausalLM` 的 `load_weights`、`prepare_inputs_for_decode` 等完全可复用
- 修改 Qwen2（方案 C）会污染现有模型，不符合"最小侵入"原则
- 只需新建 `Qwen3Attention`、`Qwen3DecoderLayer`，以及绕过双重注册的 `Qwen3Model`/`Qwen3ForCausalLM` 壳

**RMSNorm 3D bug 的根因与修复**：这是整个实现过程中最隐蔽的 bug。现象是模型生成纯乱码（"111111..."），排查步骤：

```
1. 确认权重加载正确（loss 收敛的 checkpoint）
2. 比对 HuggingFace transformers 的 Qwen3 实现：
   q = self.q_norm(q.view(bsz, q_len, self.num_heads, self.head_dim))
   k = self.k_norm(k.view(bsz, q_len, self.num_kv_heads, self.head_dim))
   # HF 的 RMSNorm 在 head_dim 维做归一化
3. 发现 vLLM 的 RMSNorm CUDA kernel 对整行做归一化：
   如果输入是 (T, num_heads * head_dim)，kernel 对 num_heads * head_dim 做 norm
   —— 这相当于对整个 q 向量做 norm，而非每个 head 分别 norm
4. 修复：flatten 到 (T * num_heads, head_dim) → norm → unflatten
```

修复后输出立即恢复正常，验证了根因判断。

### 6.6 关键数字速查

| 项目 | 值 |
|------|---|
| Qwen3 `<think>` token ID | 151667 |
| Qwen3 `</think>` token ID | 151668 |
| 新增文件数 | 5（`abs_reasoning_parsers.py`, `deepseek_r1_reasoning_parser.py`, `__init__.py`, `qwen3.py`, `registry.py` 修改） |
| 修改文件数 | 4（`protocol.py`, `cli_args.py`, `api_server.py`, `serving_chat.py`） |
| 单元测试数 | 11（6 非流式 + 3 流式 + 1 注册 + 1 错误用例） |
| E2E 测试数 | 3（非流式 thinking、流式 thinking、无 thinking） |
| 测试用 GPU | NVIDIA RTX 3060 Ti 8GB |
| 启动必需参数 | `--enforce-eager`（CUDA graph 与自定义模型不兼容） |

---

> 如需深入了解 vLLM 的 engine 层（调度、KV cache 管理）或 worker 层（模型并行），可参考 [vLLM 官方文档](https://docs.vllm.ai) 或同目录下的 `design.md`（serving 层接口设计）与 `testing.md`（测试用例详解）。
