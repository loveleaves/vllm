# vLLM Reasoning Outputs 安装与开发文档

> **适用版本**：本分支基于 vLLM `fbb5bd4ce`（v0.6.6 fix），在 `reasoning_outputs` 分支上开发。
>
> **验证环境**：NVIDIA GeForce RTX 3060 Ti 8 GB · CUDA 12.4 · Python 3.12 · torch 2.5.1 · WSL2 (Linux 5.15)

---

## 目录

1. [环境要求](#1-环境要求)
2. [获取代码](#2-获取代码)
3. [安装依赖](#3-安装依赖)
4. [安装预编译 C 扩展（`vllm._C`）](#4-安装预编译-c-扩展vllm_c)
5. [下载模型](#5-下载模型)
6. [启动服务](#6-启动服务)
7. [验证安装](#7-验证安装)
8. [运行测试](#8-运行测试)
9. [故障排查](#9-故障排查)

---

## 1. 环境要求

### 硬件

| 项目 | 最低要求 | 推荐 |
|------|---------|------|
| GPU | NVIDIA 8 GB VRAM（运行 Qwen3-1.7B） | RTX 3060 Ti / A10 / A100 |
| CPU | 4 核 | 8 核以上 |
| 内存 | 16 GB RAM | 32 GB RAM |
| 磁盘 | 20 GB 空闲（模型 + venv） | 50 GB |

> **注意**：若 VRAM 不足 8 GB，可通过减小 `--gpu-memory-utilization`（如 0.7）或 `--max-model-len`（如 2048）降低显存占用，代价是最大上下文长度减小。

### 软件

| 软件 | 版本 | 说明 |
|------|------|------|
| NVIDIA Driver | ≥ 525.60 | 支持 CUDA 12.x |
| CUDA Toolkit | 12.1 – 12.4 | nvcc 用于 C 扩展编译；precompiled 路线可跳过 |
| Python | 3.10 – 3.12 | 3.12 已验证 |
| PyTorch | 2.5.1+cu121 | **版本须与 vllm 依赖完全一致** |
| git | ≥ 2.30 | |

验证 GPU 与 CUDA：

```bash
nvidia-smi
# 期望：Driver Version ≥ 525，CUDA Version ≥ 12.1

nvcc --version
# 期望：release 12.x
```

---

## 2. 获取代码

```bash
# 克隆仓库（若已有可跳过）
git clone https://github.com/loveleaves/vllm.git
cd vllm

# 切换到 reasoning_outputs 分支
git checkout reasoning_outputs

# 确认当前在正确的 commit
git log --oneline -3
# 期望输出（前三行）：
# 9f1716dae feat: add Qwen3ForCausalLM model support and complete E2E validation
# 713566dbf feat: add reasoning output support via <think> tag parsing
# 2d090b5dd init
```

**分支说明**：`reasoning_outputs` 分支在 `fbb5bd4ce`（vLLM v0.6.6 fix）基础上新增了两个提交：

| Commit | 内容 |
|--------|------|
| `713566dbf` | 核心功能：ReasoningParser 插件系统、`<think>` 解析、CLI 参数、serving 层集成 |
| `9f1716dae` | Qwen3-1.7B 模型支持（`qwen3.py`）、registry 注册、E2E 测试与文档 |

---

## 3. 安装依赖

推荐使用 venv 隔离环境：

```bash
# 在仓库根目录下创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate
```

### 3.1 安装 PyTorch（须与 CUDA 版本匹配）

```bash
# CUDA 12.1（本仓库 requirements 指定版本）
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

> 若本机 CUDA 为 12.4，torch 仍使用 cu121 编译版本即可（向上兼容）。**切勿**升级到 2.6.x，否则与 vllm 的 C 扩展 ABI 不兼容。

验证 PyTorch 可用：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望：2.5.1+cu121 True
```

### 3.2 安装 Python 依赖

```bash
pip install -r requirements-common.txt
pip install -r requirements-cuda.txt
pip install -r requirements-dev.txt     # 用于测试（可选）
```

> 若网络受限，可使用国内镜像：
> ```bash
> pip install -r requirements-common.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

---

## 4. 安装预编译 C 扩展（`vllm._C`）

vLLM 的核心功能（PagedAttention、RMSNorm CUDA kernel 等）依赖 C 扩展 `vllm/_C.abi3.so`。`pip install -e .` 默认会从源码编译该扩展（需要 nvcc 且耗时 15–30 分钟）。**推荐使用 precompiled 路线**：

### 4.1 下载 vllm-0.6.6.post1 预编译 wheel

```bash
mkdir -p /tmp/vllm_wheel && cd /tmp/vllm_wheel

# 从 PyPI 下载（约 201 MB）
pip download vllm==0.6.6.post1 \
    --no-deps \
    --platform manylinux1_x86_64 \
    --python-version 38 \
    --abi abi3 \
    -d .

# 验证文件大小（应约为 201 MB）
ls -lh vllm-0.6.6.post1-cp38-abi3-manylinux1_x86_64.whl
```

> **若 pip download 因网络中断**，使用 wget 断点续传：
> ```bash
> wget -c "https://files.pythonhosted.org/packages/.../vllm-0.6.6.post1-cp38-abi3-manylinux1_x86_64.whl" \
>     -O /tmp/vllm_wheel/vllm-0.6.6.post1.whl
> ```
> 文件 SHA256 可在 PyPI 页面 `https://pypi.org/project/vllm/0.6.6.post1/#files` 核对。

### 4.2 以 precompiled 模式安装本仓库

```bash
cd /path/to/vllm   # 回到仓库根目录

VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_LOCATION=/tmp/vllm_wheel/vllm-0.6.6.post1-cp38-abi3-manylinux1_x86_64.whl \
pip install -e . --no-build-isolation
```

这条命令会：
1. 从 wheel 中解压 `_C.abi3.so`、`_moe_C.abi3.so` 等预编译二进制文件到 `vllm/`
2. 以可编辑模式（editable）安装 Python 源码（修改 Python 文件无需重装）

> **`--no-build-isolation`**：禁止 pip 创建临时编译环境，让构建系统直接使用当前 venv 中的 torch，避免版本错位。

### 4.3 验证 C 扩展

```bash
python -c "import vllm._C; print('vllm._C OK')"
# 期望：vllm._C OK

python -c "import vllm; print(vllm.__version__)"
# 期望：0.1.dev3+g<commit_hash>
```

---

## 5. 下载模型

Reasoning output 功能已在 **Qwen3-1.7B** 上端到端验证。

### 5.1 使用 huggingface-cli（推荐）

```bash
pip install huggingface-hub

# 下载 Qwen3-1.7B（约 3.4 GB）
huggingface-cli download Qwen/Qwen3-1.7B \
    --local-dir ~/.cache/huggingface/hub/Qwen3-1.7B

# 验证
ls ~/.cache/huggingface/hub/Qwen3-1.7B/
# 期望：config.json  model.safetensors  tokenizer.json  tokenizer_config.json ...
```

### 5.2 国内镜像（镜像站）

```bash
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download Qwen/Qwen3-1.7B \
    --local-dir ~/.cache/huggingface/hub/Qwen3-1.7B
```

### 5.3 确认 config.json

```bash
python -c "
import json
with open('${HOME}/.cache/huggingface/hub/Qwen3-1.7B/config.json') as f:
    cfg = json.load(f)
print('架构:', cfg['architectures'])
print('层数:', cfg['num_hidden_layers'])
print('词表大小:', cfg['vocab_size'])
"
# 期望：
# 架构: ['Qwen3ForCausalLM']
# 层数: 28
# 词表大小: 151936
```

> **内存优化（可选）**：若 8 GB VRAM 仍不足，可将 `config.json` 中 `num_hidden_layers` 改小（如 16），减少模型层数。修改后模型权重仍完整，vLLM 加载时会自动截断多余层。**此操作改变模型质量，仅用于调试/演示。**

---

## 6. 启动服务

```bash
.venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model ~/.cache/huggingface/hub/Qwen3-1.7B \
    --enable-reasoning \
    --reasoning-parser qwen3 \
    --port 8100 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager
```

### 参数说明

| 参数 | 值 | 说明 |
|------|----|------|
| `--model` | `~/.cache/.../Qwen3-1.7B` | 模型路径（或 HuggingFace 模型 ID） |
| `--enable-reasoning` | flag | 开启 `<think>` 块解析，缺少此参数则不返回 `reasoning_content` |
| `--reasoning-parser` | `qwen3` | 指定 parser；可用值：`qwen3`、`deepseek_r1`（两者指向同一实现） |
| `--port` | `8100` | 服务端口（默认 8000，可自定义） |
| `--max-model-len` | `4096` | 最大 context 长度（降低可节省 VRAM） |
| `--gpu-memory-utilization` | `0.85` | GPU 显存利用率上限（8 GB 卡推荐 0.80–0.90） |
| `--enforce-eager` | flag | **必须**：禁用 CUDA Graph 捕获，与 Qwen3 自定义模型兼容 |

### 期望启动日志

```
INFO  Loading model...
INFO  Model architecture: Qwen3ForCausalLM
INFO  Available memory: 6.xx GiB x 1 GPU
INFO  KV cache dtype: auto, Number of GPU blocks: xxx
INFO  Running on http://0.0.0.0:8100
```

> 首次启动约需 **30–60 秒**（模型权重加载 + tokenizer 初始化）。

---

## 7. 验证安装

### 7.1 非流式请求（开启 thinking）

```bash
curl -s http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/home/cb/.cache/huggingface/hub/Qwen3-1.7B",
    "messages": [{"role": "user", "content": "9.11 和 9.9 哪个更大？"}],
    "chat_template_kwargs": {"enable_thinking": true},
    "max_tokens": 512,
    "stream": false
  }' | python -m json.tool
```

期望响应（关键字段）：

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "reasoning_content": "我需要比较 9.11 和 9.9...",
        "content": "9.9 更大。"
      },
      "finish_reason": "stop"
    }
  ]
}
```

验证要点：
- `reasoning_content` 非空，包含思考过程，**不含** `<think>`/`</think>` 标签
- `content` 是最终答案，**不含** `<think>`/`</think>` 标签

### 7.2 非流式请求（关闭 thinking）

```bash
curl -s http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/home/cb/.cache/huggingface/hub/Qwen3-1.7B",
    "messages": [{"role": "user", "content": "你好"}],
    "chat_template_kwargs": {"enable_thinking": false},
    "max_tokens": 64,
    "stream": false
  }' | python -m json.tool
```

期望响应：

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "reasoning_content": null,
        "content": "你好！有什么我可以帮助你的吗？"
      }
    }
  ]
}
```

### 7.3 流式请求

```bash
curl -s http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/home/cb/.cache/huggingface/hub/Qwen3-1.7B",
    "messages": [{"role": "user", "content": "1+1=？请思考后回答"}],
    "chat_template_kwargs": {"enable_thinking": true},
    "max_tokens": 256,
    "stream": true
  }'
```

期望：先收到若干包含 `"reasoning_content"` 字段的 chunk，再收到包含 `"content"` 字段的 chunk：

```
data: {"choices":[{"delta":{"reasoning_content":"让我来计算"},...}]}
data: {"choices":[{"delta":{"reasoning_content":"1加1等于"},...}]}
...
data: {"choices":[{"delta":{"content":"2"},...}]}
data: {"choices":[{"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

### 7.4 Python 客户端验证

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8100/v1",
    api_key="EMPTY",
)
MODEL = "/home/cb/.cache/huggingface/hub/Qwen3-1.7B"

resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "9.11 和 9.9 哪个更大？"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    max_tokens=512,
)

msg = resp.choices[0].message
print("reasoning_content:", msg.reasoning_content)
print("content:", msg.content)

assert msg.reasoning_content is not None, "未返回 reasoning_content"
assert "<think>" not in (msg.content or ""), "content 含有 <think> 标签"
print("验证通过 ✓")
```

---

## 8. 运行测试

### 8.1 单元测试（无需 GPU）

单元测试使用 `facebook/opt-125m` tokenizer（自动下载，约 250 MB）模拟 think tokens，无需 GPU。

```bash
# 确保依赖已安装
pip install pytest

# 运行 reasoning parser 单元测试（11 个用例）
.venv/bin/python -m pytest \
    tests/entrypoints/openai/reasoning_parsers/test_deepseekr1_reasoning_parser.py \
    -v --noconftest \
    -p tests.entrypoints.openai.reasoning_parsers.conftest
```

期望输出：

```
PASSED test_non_streaming_full_think_block
PASSED test_non_streaming_no_think_block
PASSED test_non_streaming_unclosed_think
PASSED test_non_streaming_empty_think_block
PASSED test_non_streaming_multiline_reasoning
PASSED test_non_streaming_only_think_no_content
PASSED test_streaming_full_think_block
PASSED test_streaming_no_think_block
PASSED test_streaming_reasoning_before_content
PASSED test_parser_registered_under_both_names
PASSED test_missing_think_tokens_raises
====== 11 passed in X.XXs ======
```

> **`--noconftest` 的原因**：项目根目录的 `conftest.py` 会尝试导入 `vllm._C` 等重量级模块。reasoning parser 测试目录内有独立的 `conftest.py`，通过 `-p` 指定使用它，以避免 GPU/CUDA 依赖影响单元测试。

### 8.2 端到端测试（需要 GPU + 运行中的服务）

确保第 6 节的服务已在 `8100` 端口启动，然后：

```bash
VLLM_TEST_MODEL=/home/cb/.cache/huggingface/hub/Qwen3-1.7B \
.venv/bin/python -m pytest \
    tests/entrypoints/openai/reasoning_parsers/test_e2e_qwen3.py \
    -v
```

期望输出：

```
PASSED test_non_streaming_reasoning_field
PASSED test_non_streaming_no_reasoning_when_disabled
PASSED test_streaming_reasoning_then_content
====== 3 passed in XX.XXs ======
```

### 8.3 测试文件结构

```
tests/entrypoints/openai/reasoning_parsers/
├── conftest.py                          # mock vllm._C，使单元测试无需 GPU
├── test_deepseekr1_reasoning_parser.py  # 11 个单元测试
└── test_e2e_qwen3.py                    # 3 个 E2E 测试
```

---

## 9. 故障排查

### 9.1 `ModuleNotFoundError: No module named 'vllm._C'`

**原因**：`pip install -e .` 未编译/提取 C 扩展。

**解决**：重新执行[第 4 节](#4-安装预编译-c-扩展vllm_c)的 precompiled 安装流程：

```bash
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_LOCATION=/tmp/vllm_wheel/vllm-0.6.6.post1-cp38-abi3-manylinux1_x86_64.whl \
pip install -e . --no-build-isolation
```

### 9.2 `ValueError: Model architectures ['Qwen3ForCausalLM'] are not supported`

**原因**：vLLM v0.6.6 原版不支持 Qwen3。本分支已添加支持，若报此错误说明未切换到 `reasoning_outputs` 分支。

```bash
git branch       # 确认当前分支
git checkout reasoning_outputs
```

### 9.3 模型推理输出乱码（全是"1"或换行符）

**原因**：Qwen3 的 QK-norm 未正确应用。本分支已修复（`qwen3.py` 中对 Q/K 先 reshape 再 RMSNorm）。若仍出现，确认使用的是 `reasoning_outputs` 分支的 `vllm/model_executor/models/qwen3.py`：

```bash
grep "reshape" vllm/model_executor/models/qwen3.py
# 期望看到：q.reshape(t * self.num_heads, self.head_dim)
```

### 9.4 CUDA Graph 相关报错

**原因**：Qwen3 自定义模型与 CUDA Graph 捕获不兼容。

**解决**：启动命令加 `--enforce-eager`（参见[第 6 节](#6-启动服务)）。

### 9.5 `RuntimeError: Tokenizer does not have '<think>' or '</think>' as single tokens`

**原因**：使用的模型 tokenizer 中 `<think>` 不是单一 special token，无法使用 `DeepSeekR1ReasoningParser`。

**解决**：
1. 确认模型为 Qwen3 或 DeepSeek-R1 系列
2. 检查 tokenizer 词表：
   ```bash
   python -c "
   from transformers import AutoTokenizer
   tok = AutoTokenizer.from_pretrained('~/.cache/huggingface/hub/Qwen3-1.7B')
   v = tok.get_vocab()
   print('<think>  ID:', v.get('<think>'))
   print('</think> ID:', v.get('</think>'))
   "
   # 期望：<think> ID: 151667，</think> ID: 151668
   ```

### 9.6 `reasoning_content` 字段未出现在响应中

**可能原因**：

| 检查点 | 命令 |
|-------|------|
| 服务端是否携带 `--enable-reasoning` | `ps aux | grep vllm` |
| 请求是否设置 `enable_thinking: true` | 查看请求 body 中 `chat_template_kwargs` |
| 模型是否真的输出了 `<think>` | 临时去掉 `--enable-reasoning`，直接看原始 `content` |

### 9.7 OOM（显存不足）

**解决方案**（按侵入程度从低到高）：

```bash
# 方案①：降低 GPU 显存利用率
--gpu-memory-utilization 0.75

# 方案②：缩短最大 context 长度
--max-model-len 2048

# 方案③：使用 8-bit 量化（需安装 bitsandbytes）
--quantization bitsandbytes --load-format bitsandbytes

# 方案④（激进）：修改 config.json 减少层数（影响质量）
python -c "
import json
with open('~/.cache/huggingface/hub/Qwen3-1.7B/config.json') as f: cfg=json.load(f)
cfg['num_hidden_layers'] = 16  # 从 28 层减到 16 层
with open('~/.cache/huggingface/hub/Qwen3-1.7B/config.json','w') as f: json.dump(cfg,f,indent=2)
"
```

### 9.8 `--enable-reasoning` 与 `--enable-auto-tool-choice` 同时使用报错

两者在本版本中**互斥**，不可同时开启。如需工具调用与思考内容并存，需等待后续版本实现。

---

## 附录：完整启动示例（一键复制）

```bash
# 1. 进入仓库目录，激活 venv
cd /path/to/vllm && source .venv/bin/activate

# 2. 启动服务
python -m vllm.entrypoints.openai.api_server \
    --model ~/.cache/huggingface/hub/Qwen3-1.7B \
    --enable-reasoning \
    --reasoning-parser qwen3 \
    --port 8100 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager &

# 3. 等待启动完成（检测到端口监听为准）
until curl -s http://localhost:8100/health >/dev/null 2>&1; do sleep 2; done
echo "服务已就绪"

# 4. 快速验证
curl -s http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$HOME"'/.cache/huggingface/hub/Qwen3-1.7B",
    "messages": [{"role":"user","content":"1+1=?"}],
    "chat_template_kwargs": {"enable_thinking": true},
    "max_tokens": 256
  }' | python -m json.tool | grep -E "reasoning_content|content"
```
