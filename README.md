## reasoning outputs
origin breanch: vllm fbb5bd4ce

### 2026.6.4
Adds Qwen3 model implementation (qwen3.py) with QK-norm support:
- Qwen3Attention: no QKV bias + RMSNorm on Q and K (flattened 2D for
  CUDA kernel compatibility)
- Qwen3DecoderLayer: uses Qwen3Attention instead of Qwen2Attention
- Qwen3Model: bypasses Qwen2Model layer init via @support_torch_compile
  to avoid duplicate Attention registration; builds Qwen3DecoderLayer
- Qwen3ForCausalLM: registered as "Qwen3ForCausalLM" in model registry

Also adds:
- docs/reasoning_outputs/testing.md: full test report (11 unit + 3 E2E)
- tests/entrypoints/openai/reasoning_parsers/test_e2e_qwen3.py

E2E result on RTX 3060 Ti 8GB with Qwen3-1.7B + --enforce-eager:
- Non-streaming: message.reasoning_content non-empty, content tag-free
- Streaming: reasoning chunks precede content chunks
- Non-thinking mode: reasoning_content is None

### 2026.6.3
Introduces ReasoningParser plugin system (modeled after ToolParser) so
models that emit `<think>…</think>` blocks (DeepSeek-R1, Qwen3) can
return reasoning content separately from the final answer.

New files:
- vllm/entrypoints/openai/reasoning_parsers/abs_reasoning_parsers.py
  Abstract base class + ReasoningParserManager registry
- vllm/entrypoints/openai/reasoning_parsers/deepseek_r1_reasoning_parser.py
  Concrete parser registered as "deepseek_r1" and "qwen3"
- vllm/entrypoints/openai/reasoning_parsers/__init__.py

Modified files:
- protocol.py: ChatMessage/DeltaMessage gain reasoning_content field
- cli_args.py: --enable-reasoning / --reasoning-parser flags
- api_server.py: pass new params to OpenAIServingChat
- serving_chat.py: __init__ + streaming + non-streaming reasoning paths

Tests:
- tests/entrypoints/openai/reasoning_parsers/ (11 unit tests, all green)