# Reasoning Outputs 测试文档

## 测试环境

| 项目 | 值 |
|------|----|
| GPU | NVIDIA GeForce RTX 3060 Ti (8 GB VRAM) |
| OS | Linux 5.15 (WSL2) |
| Python | 3.12.13 |
| PyTorch | 2.5.1+cu121 |
| vLLM | 0.1.dev3 (commit 448de14c3 / fbb5bd4ce base) |
| E2E 模型 | Qwen/Qwen3-1.7B |
| 单测模型 | facebook/opt-125m (注入 `<think>` token) |

---

## 测试用例清单

| 测试文件 | 测试函数 | 类型 | 测试点 | 结果 |
|---------|---------|------|--------|------|
| `test_deepseekr1_reasoning_parser.py` | `test_non_streaming_full_think_block` | Unit | 非流式：完整 `<think>…</think>` 块 | ✅ PASS |
| `test_deepseekr1_reasoning_parser.py` | `test_non_streaming_no_think_block` | Unit | 非流式：无 `<think>` 块 | ✅ PASS |
| `test_deepseekr1_reasoning_parser.py` | `test_non_streaming_unclosed_think` | Unit | 非流式：未闭合的 `<think>`（生成截断）| ✅ PASS |
| `test_deepseekr1_reasoning_parser.py` | `test_non_streaming_empty_think_block` | Unit | 非流式：空推理块 `<think></think>` | ✅ PASS |
| `test_deepseekr1_reasoning_parser.py` | `test_non_streaming_multiline_reasoning` | Unit | 非流式：多行推理内容 | ✅ PASS |
| `test_deepseekr1_reasoning_parser.py` | `test_non_streaming_only_think_no_content` | Unit | 非流式：推理后无内容 | ✅ PASS |
| `test_deepseekr1_reasoning_parser.py` | `test_streaming_full_think_block` | Unit | 流式：推理 + 正文 delta 分离 | ✅ PASS |
| `test_deepseekr1_reasoning_parser.py` | `test_streaming_no_think_block` | Unit | 流式：无 `<think>` 块时全走 content | ✅ PASS |
| `test_deepseekr1_reasoning_parser.py` | `test_streaming_reasoning_before_content` | Unit | 流式：reasoning delta 先于 content delta | ✅ PASS |
| `test_deepseekr1_reasoning_parser.py` | `test_parser_registered_under_both_names` | Unit | 注册表：`deepseek_r1` 与 `qwen3` 同时注册 | ✅ PASS |
| `test_deepseekr1_reasoning_parser.py` | `test_missing_think_tokens_raises` | Unit | 无 think token 时快速失败 | ✅ PASS |
| `test_e2e_qwen3.py` | `test_non_streaming_reasoning_field` | E2E | 非流式：`message.reasoning_content` 非空，`content` 无标签 | ✅ PASS |
| `test_e2e_qwen3.py` | `test_non_streaming_no_reasoning_when_disabled` | E2E | 非流式：关闭 thinking 时 `reasoning_content` 为 None | ✅ PASS |
| `test_e2e_qwen3.py` | `test_streaming_reasoning_then_content` | E2E | 流式：reasoning chunk 先于 content chunk | ✅ PASS |

单元测试运行命令：
```bash
.venv/bin/python -m pytest \
  tests/entrypoints/openai/reasoning_parsers/test_deepseekr1_reasoning_parser.py \
  -v --noconftest \
  -p tests.entrypoints.openai.reasoning_parsers.conftest
```

E2E 测试启动命令：
```bash
# 启动 vllm 服务（Qwen3-1.7B）
.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model ~/.cache/huggingface/hub/Qwen3-1.7B \
  --enable-reasoning \
  --reasoning-parser qwen3 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85

# 运行 E2E 测试
VLLM_TEST_MODEL=Qwen3-1.7B \
.venv/bin/python -m pytest \
  tests/entrypoints/openai/reasoning_parsers/test_e2e_qwen3.py -v
```

---

## 验收标准对照

| 验收标准（来自 PRD）| 测试方法 | 实测值 | 是否达标 |
|-------------------|---------|--------|---------|
| `<think>` 标签内文本 == `message.reasoning_content`，误差 0 字符 | `test_non_streaming_full_think_block` (unit) + E2E | Unit: 精确匹配；E2E: 非流式验证 | ✅ |
| `message.content` 不含任何 `<think>` 标签 | `test_streaming_full_think_block` (unit) + E2E | Unit/E2E 均验证通过 | ✅ |
| 普通模型 `message.reasoning_content == None` | `test_non_streaming_no_think_block` (unit) + E2E Test 3 | `None` | ✅ |
| 单元测试覆盖 ≥5 个边界条件 | Unit test suite | 11 个测试用例，6 种非流式边界 | ✅ |
| 实际模型在 8 GB 机器上返回正确 `message.reasoning_content` | E2E test（Qwen3-1.7B, RTX 3060 Ti 8GB） | 非流式+流式均返回 reasoning_content | ✅ |

---

## 已知局限

1. **Streaming 状态机无显式状态**：通过每次扫描 `previous_token_ids` 推导，最坏 O(n²)。对于长思考链（>10K tokens）可考虑改为显式 `in_reasoning` 状态位。
2. **多 `<think>` 块**：当前解析器只处理第一个 `<think>` 块，Qwen3 和 DeepSeek-R1 正常情况下不会输出多个，但若模型异常可能漏掉后续块。
3. **与 auto tool choice 互斥**：两者共用 `previous_texts` 状态，CLI 层已加互斥校验。
4. **E2E 依赖 Qwen3 thinking 模式**：Qwen3 默认不开启 thinking，需在请求中传 `enable_thinking: true` 才会输出 `<think>` 块。
