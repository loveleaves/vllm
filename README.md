## reasoning outputs
origin breanch: vllm fbb5bd4ce

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