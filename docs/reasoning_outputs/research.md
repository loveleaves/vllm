# Reasoning Outputs 技术调研报告

## 摘要

vLLM 的 reasoning outputs 功能让 OpenAI-compatible serving 层能将模型输出中的
"思考过程"（如 DeepSeek R1 的 `<think>…</think>` 内容）与最终答案分离，
通过 `message.reasoning` / `delta.reasoning` 字段单独返回给客户端。
该功能**纯粹是 serving 前端的文本后处理**，不涉及推理引擎、调度器或模型权重。

---

## 参照实现对比

### 1. vLLM 参考实现（commit `a7e3eba66`，2025-01-29）

#### 架构

```
CLI args (--enable-reasoning / --reasoning-parser)
        │
        ▼
OpenAIServingChat.__init__
  └── ReasoningParserManager.get_reasoning_parser(name)
        │  (返回 class，不是实例)
        ▼
chat_completion_stream_generator / create_chat_completion
  └── reasoning_parser = ReasoningParserClass(tokenizer)
        │
        ├── non-streaming: extract_reasoning_content(full_text) → (reasoning, content)
        └── streaming:     extract_reasoning_content_streaming(prev, curr, delta, ...) → DeltaMessage
```

#### 关键数据结构

| 结构 | 位置 | 说明 |
|------|------|------|
| `ReasoningParser` | `abs_reasoning_parsers.py` | 抽象基类，定义两个接口方法 |
| `ReasoningParserManager` | `abs_reasoning_parsers.py` | 名称→类的注册表（类变量字典） |
| `DeltaMessage.reasoning_content` | `protocol.py` | streaming delta 的推理字段 |
| `ChatMessage.reasoning_content` | `protocol.py` | non-streaming 完整响应的推理字段 |

#### 接口设计

```python
class ReasoningParser:
    def extract_reasoning_content(
        self, model_output: str, request: ChatCompletionRequest
    ) -> Tuple[Optional[str], Optional[str]]:
        # 返回 (reasoning_content, content)
        ...

    def extract_reasoning_content_streaming(
        self,
        previous_text: str, current_text: str, delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Optional[DeltaMessage]:
        # 返回包含 reasoning_content 或 content 的 delta，或 None（跳过控制 token）
        ...
```

#### DeepSeek R1 解析逻辑

- **Non-streaming**：正则 `<think>(.*?)</think>` 提取，剩余文本作为 content。
- **Streaming**：用 token ID（不用文本匹配）检测 `think_start_token_id` /
  `think_end_token_id` 是否出现在 `previous_token_ids` / `delta_token_ids` 中，
  决定当前 delta 路由到 `reasoning_content` 还是 `content`。

  状态转移（无显式状态机，通过 token id 列表推导）：

  ```
  delta 含 think_start_id  → 开始输出 reasoning（跳过 <think> token 本身）
  delta 含 think_end_id    → reasoning 结束，切换到 content
  previous 含 think_start_id 且无 think_end_id → 仍在 reasoning 中
  previous 含 think_end_id → 已切换，输出 content
  ```

#### 优缺点

| 优点 | 缺点 |
|------|------|
| 无状态（实际由 token list 推导状态），实现简单 | 每次 delta 都扫描整个 previous_token_ids，O(n) 开销 |
| 插件注册机制，新模型只需新文件 | 流式中 <think> 与文本同 delta 时有边界处理复杂性 |
| 单元测试不需要 GPU（用 opt-125m + 自定义 token）| streaming 中 delta 含 `</think>` 后的内容需拆分 |

---

### 2. ToolParser（同代码库，`vllm/entrypoints/openai/tool_parsers/`）

作为 reasoning parser 的"设计原型"参考。

- 同样是 `@ToolParserManager.register_module("name")` 装饰器注册。
- 同样区分 streaming / non-streaming 两个方法。
- 区别：tool parser 处理结构化 JSON，reasoning parser 处理自由文本标签。

**结论**：reasoning parser 的整体架构直接对标 tool parser，可以完全参照其接入模式。

---

## 当前代码库分析（基线 `fbb5bd4ce`）

### 涉及文件

```
vllm/entrypoints/openai/
├── api_server.py          ← 初始化 OpenAIServingChat，需传入新参数
├── cli_args.py            ← 解析 --reasoning-parser 参数，需验证逻辑
├── protocol.py            ← ChatMessage / DeltaMessage，需新增字段
├── serving_chat.py        ← 核心：__init__ + streaming/non-streaming 路径
└── tool_parsers/          ← 参照设计，不修改
```

### `serving_chat.py` 关键锚点

| 行号（fbb5bd4ce） | 位置 | 说明 |
|---|---|---|
| L39-84 | `__init__` | 初始化 tool_parser，在此处类比添加 reasoning_parser |
| L284-294 | streaming generator 顶部 | `tool_choice_auto` 分支，在此添加 `should_stream_with_reasoning` |
| L298-303 | tool_parser 实例化 | 在此类比实例化 reasoning_parser |
| L423-450 | delta 路由 if/elif/else | 在 `tool_choice_auto` 之后加 `elif enable_reasoning` 分支 |
| L127-175 | non-streaming 路径 | 输出后处理，在此加 reasoning 提取 |

### protocol.py 需要的修改

```python
# ChatCompletionResponseMessage（非流式）
reasoning_content: Optional[str] = None   # 新增

# DeltaMessage（流式）
reasoning_content: Optional[str] = None   # 新增
```

---

## GPU 与模型选择

**GPU**：NVIDIA GeForce RTX 3060 Ti，8 GB VRAM

| 测试层级 | 推荐模型 | VRAM 占用 | 说明 |
|---------|---------|----------|------|
| Unit Test | `facebook/opt-125m` | ~1 GB（CPU 可用）| 无需真实 reasoning 模型，手动添加 `<think>` token |
| E2E Test | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | ~3 GB | 原生支持 `<think>`，8 GB 完全足够 |

> **无需层削减**：1.5B 模型 bf16 约 3 GB，8 GB RTX 3060 Ti 有充足余量（KV cache + 系统开销约 2-3 GB）。
> 若需要更强的推理能力用于 E2E 效果验证，可尝试 `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
> 配合 `--quantization awq` 或 INT4 量化（约 4-5 GB）。

---

## 结论与选型建议

1. **架构沿用 tool_parser 模式**：注册表 + 抽象基类 + 具体实现，三文件结构清晰。
2. **流式状态推导方式**：用 token ID list 检测（参考实现已验证），无需显式状态机。
3. **非流式解析**：正则即可，DeepSeek R1 格式固定（`<think>` 始终在开头）。
4. **单元测试免 GPU**：用 `opt-125m` + `tokenizer.add_tokens` 注入特殊 token，
   与参考实现测试策略完全一致。
5. **E2E 模型**：`DeepSeek-R1-Distill-Qwen-1.5B`，无需量化，直接 bf16 运行。
