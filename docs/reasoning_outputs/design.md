# Reasoning Outputs 详细设计文档

## Motivation

vLLM 当前 serving 层（基线 `fbb5bd4ce`）将模型所有输出统一放入 `content` 字段返回。
DeepSeek R1 等推理模型会在输出中插入 `<think>…</think>` 思考块，客户端需要将其
与最终答案分离后分别展示。本功能在 serving 前端增加可插拔的 **ReasoningParser**，
让 `message.reasoning` / `delta.reasoning` 独立返回，`content` 只含最终答案。

---

## Architecture

### 模块交互图

```
┌─────────────────────────────────────────────────┐
│  CLI / api_server.py                            │
│  --reasoning-parser deepseek_r1                 │
│       │ args.reasoning_parser                   │
│       ▼                                         │
│  OpenAIServingChat.__init__                     │
│    └─ ReasoningParserManager                    │
│         .get_reasoning_parser("deepseek_r1")    │
│         → DeepSeekR1ReasoningParser (class)     │
└──────────────────┬──────────────────────────────┘
                   │ self.reasoning_parser (class ref)
                   ▼
┌─────────────────────────────────────────────────┐
│  serving_chat.py                                │
│                                                 │
│  Non-streaming path                             │
│    output.text                                  │
│      └─ parser.extract_reasoning_content()      │
│           → (reasoning_str, content_str)        │
│           → ChatMessage(reasoning=...,          │
│                         content=...)            │
│                                                 │
│  Streaming path                                 │
│    per-delta loop                               │
│      └─ parser.extract_reasoning_content_       │
│           streaming(prev, curr, delta, ids)     │
│           → DeltaMessage(reasoning=delta_r)     │
│              OR DeltaMessage(content=delta_c)   │
│              OR None  (skip control token)      │
└─────────────────────────────────────────────────┘
```

### 数据流

```
Model output tokens
  → (vllm engine, 不变)
  → RequestOutput.outputs[i].text / token_ids
  → ReasoningParser (新增，serving 层)
  → ChatMessage.reasoning + ChatMessage.content  (non-streaming)
  → DeltaMessage.reasoning / DeltaMessage.content (streaming)
  → JSON response → 客户端
```

---

## Interfaces

### 1. 新建文件：`vllm/entrypoints/openai/reasoning_parsers/`

#### `abs_reasoning_parsers.py`

```python
class ReasoningParser:
    def __init__(self, tokenizer: AnyTokenizer): ...

    @cached_property
    def vocab(self) -> Dict[str, int]: ...

    def extract_reasoning_content(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Non-streaming: 返回 (reasoning_content, content)"""
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
        """Streaming: 返回带 reasoning 或 content 的 delta，或 None 跳过"""
        raise NotImplementedError


class ReasoningParserManager:
    reasoning_parsers: Dict[str, Type] = {}

    @classmethod
    def get_reasoning_parser(cls, name: str) -> Type[ReasoningParser]: ...

    @classmethod
    def register_module(
        cls,
        name: Optional[Union[str, List[str]]] = None,
        force: bool = True,
        module: Optional[Type] = None,
    ) -> Union[Type, Callable]: ...
```

#### `deepseek_r1_reasoning_parser.py`

```python
@ReasoningParserManager.register_module("deepseek_r1")
class DeepSeekR1ReasoningParser(ReasoningParser):
    think_start_token = "<think>"
    think_end_token   = "</think>"

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        # 从词表中查找 token id，用于流式状态推导
        self.think_start_token_id = self.vocab[self.think_start_token]
        self.think_end_token_id   = self.vocab[self.think_end_token]
```

#### `__init__.py`

```python
from .abs_reasoning_parsers import ReasoningParser, ReasoningParserManager
from .deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
```

---

### 2. 修改 `protocol.py`

```python
class ChatMessage(OpenAIBaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    reasoning_content: Optional[str] = None   # ← 新增

class DeltaMessage(OpenAIBaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: List[DeltaToolCall] = Field(default_factory=list)
    reasoning_content: Optional[str] = None   # ← 新增
```

---

### 3. 修改 `cli_args.py`

新增两个 argument：

```
--enable-reasoning        action="store_true"，默认 False
--reasoning-parser NAME   从 ReasoningParserManager 已注册名称中选择
```

validate_parsed_serve_args 中增加检查：
- `--enable-reasoning` 必须配合 `--reasoning-parser`
- `--enable-auto-tool-choice` 与 `--enable-reasoning` 互斥

---

### 4. 修改 `api_server.py`

在构造 `OpenAIServingChat` 时透传新参数：

```python
state.openai_serving_chat = OpenAIServingChat(
    ...
    enable_auto_tools=args.enable_auto_tool_choice,
    tool_parser=args.tool_call_parser,
    enable_reasoning=args.enable_reasoning,        # ← 新增
    reasoning_parser=args.reasoning_parser,        # ← 新增
)
```

---

### 5. 修改 `serving_chat.py`

#### `__init__` 新增初始化（紧跟 tool_parser 初始化之后）

```python
self.enable_reasoning: bool = enable_reasoning
self.reasoning_parser: Optional[Callable[[AnyTokenizer], ReasoningParser]] = None
if self.enable_reasoning:
    try:
        self.reasoning_parser = ReasoningParserManager.get_reasoning_parser(
            reasoning_parser)
    except Exception as e:
        raise TypeError(...) from e
```

#### Streaming 路径变更

在 `chat_completion_stream_generator` 顶部：

```python
should_stream_with_reasoning = self._should_stream_with_reasoning_parsing(request)

# previous_texts / all_previous_token_ids 在 tool_choice_auto 或 reasoning 时都需要
if tool_choice_auto or should_stream_with_reasoning:
    previous_texts = [""] * num_choices
    all_previous_token_ids = [[]] * num_choices
else:
    previous_texts, all_previous_token_ids = None, None
```

实例化 reasoning parser（与 tool_parser 实例化并列）：

```python
if should_stream_with_reasoning and self.reasoning_parser:
    reasoning_parsers = [self.reasoning_parser(tokenizer)] * num_choices
else:
    reasoning_parsers = [None] * num_choices
```

delta 路由 if/elif/else 新增分支（在 `tool_choice_auto` 之后、`else` 之前）：

```python
elif self.enable_reasoning:
    reasoning_parser_i = reasoning_parsers[i]
    assert reasoning_parser_i is not None
    previous_text = previous_texts[i]
    previous_token_ids = all_previous_token_ids[i]
    current_text = previous_text + delta_text
    current_token_ids = previous_token_ids + list(output.token_ids)

    delta_message = reasoning_parser_i.extract_reasoning_content_streaming(
        previous_text, current_text, delta_text,
        previous_token_ids, current_token_ids, output.token_ids,
    )
    previous_texts[i] = current_text
    all_previous_token_ids[i] = current_token_ids
```

#### Non-streaming 路径变更

在 `if (not self.enable_auto_tools ...)` 分支之前插入优先分支：

```python
if self.enable_reasoning and self.reasoning_parser:
    try:
        rp = self.reasoning_parser(tokenizer)
    except RuntimeError as e:
        return self.create_error_response(str(e))
    reasoning_content, content = rp.extract_reasoning_content(
        output.text, request=request)
    if reasoning_content:
        message = ChatMessage(role=role, content=content,
                              reasoning_content=reasoning_content)
    else:
        message = ChatMessage(role=role, content=output.text)

elif (not self.enable_auto_tools ...):   # 原有分支不变
    ...
```

#### 新增辅助方法

```python
def _should_stream_with_reasoning_parsing(
    self, request: ChatCompletionRequest
) -> bool:
    return self.enable_reasoning and self.reasoning_parser is not None
```

---

## State Machine（流式推理状态）

流式解析不维护显式状态变量，通过检查累积 token id 列表推导当前所处阶段：

```
                    ┌──────────────────────────────────────────┐
                    │ 每个 delta 到来时，检查：                  │
                    │                                          │
  think_start_id ∈ delta_ids ──→ 开始 reasoning 阶段          │
  think_start_id ∈ prev_ids                                    │
    └─ think_end_id ∉ prev_ids  ──→ 仍在 reasoning 中         │
    └─ think_end_id ∈ delta_ids ──→ reasoning 结束，同 delta   │
                                    可能含尾部 content          │
    └─ think_end_id ∈ prev_ids  ──→ 已在 content 阶段         │
  think_start_id ∉ prev_ids                                    │
    └─ think_start_id ∉ delta_ids ──→ 纯 content              │
                    └──────────────────────────────────────────┘
```

控制 token（`<think>` 或 `</think>` 单独作为 delta）→ 返回 `None`，跳过该 chunk。

---

## Risks

| 风险 | 概率 | 缓解措施 |
|------|------|---------|
| DeepSeek-R1-1.5B tokenizer 中 `<think>` 不是单一 token | 中 | 初始化时检查 vocab，若缺失抛 RuntimeError（快速失败） |
| streaming 中 `</think>` 与正文在同一 delta | 低 | 参考实现已处理：拆分 delta_text，前半归 reasoning，后半归 content |
| tool_choice_auto 与 reasoning 同时开启 | 低 | CLI 层互斥校验，提前报错 |
| `previous_texts` 变量在非 reasoning/非 tool 路径多分配 | 无 | 条件判断 `if tool_choice_auto or should_stream_with_reasoning` |

---

## Test Plan

### Unit Tests（无 GPU）

文件：`tests/entrypoints/openai/reasoning_parsers/test_deepseekr1_reasoning_parser.py`

使用 `facebook/opt-125m` tokenizer + `tokenizer.add_tokens(["<think>", "</think>"])` 模拟。

| 测试用例 | 流式 | 预期 |
|---------|------|------|
| `<think>reasoning</think>answer` | ✗ | reasoning="reasoning", content="answer" |
| `<think>reasoning</think>answer` | ✓ | 同上（逐 token） |
| `<think>reasoning</think>` | ✗ / ✓ | reasoning="reasoning", content=None |
| 无 `<think>` | ✗ / ✓ | reasoning=None, content=原文 |
| 多行推理 | ✗ / ✓ | 换行保留 |
| `<think></think>answer` | ✗ | reasoning="", content="answer" |
| `<think></think>answer` | ✓ | reasoning=None, content="answer"（空 reasoning 跳过） |

### Integration Test（需 GPU，用 1.5B 模型）

启动 vllm server，调用 OpenAI client，验证：
- `response.choices[0].message.reasoning` 非空
- `response.choices[0].message.content` 不含 `<think>` 标签
- 流式 delta 顺序正确（reasoning deltas 先于 content deltas）
