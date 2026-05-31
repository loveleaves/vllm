"""
vLLM v0.6.6 基础功能验证脚本
平台：WSL2 + RTX 3060 Ti 8GB
模型：Qwen/Qwen2.5-1.5B-Instruct
"""
import torch
from vllm import LLM, SamplingParams

# ── 1. 环境信息 ──────────────────────────────────────────────────────────────
print("=" * 60)
print("环境信息")
print("=" * 60)
print(f"PyTorch 版本  : {torch.__version__}")
print(f"CUDA 可用     : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free, _ = torch.cuda.mem_get_info()
    print(f"GPU 型号      : {torch.cuda.get_device_name(0)}")
    print(f"显存总量      : {total:.1f} GB")
    print(f"当前可用      : {free / 1024**3:.2f} GB")
print()

# ── 2. 加载模型 ──────────────────────────────────────────────────────────────
print("=" * 60)
print("加载模型：Qwen/Qwen2.5-1.5B-Instruct")
print("=" * 60)

MODEL = "/home/cb/model/Qwen2.5-1.5B-Instruct"

llm = LLM(
    model=MODEL,
    dtype="bfloat16",
    gpu_memory_utilization=0.85,
    max_model_len=2048,
    trust_remote_code=True,
)

free_after_load, _ = torch.cuda.mem_get_info()
print(f"模型加载后可用显存: {free_after_load / 1024**3:.2f} GB")
print()

# ── 3. 基础推理 ──────────────────────────────────────────────────────────────
print("=" * 60)
print("基础推理测试")
print("=" * 60)

prompts = [
    "你好，请用一句话介绍你自己。",
    "What is 3 + 5? Answer with just the number.",
    "请列出三种水果：",
]

params = SamplingParams(
    temperature=0.0,   # 贪心解码，结果确定
    max_tokens=64,
)

outputs = llm.generate(prompts, params)

for i, output in enumerate(outputs):
    prompt = output.prompt
    text   = output.outputs[0].text
    tokens = len(output.outputs[0].token_ids)
    print(f"[{i+1}] 输入 : {prompt}")
    print(f"     输出 : {text.strip()}")
    print(f"     Token : {tokens}")
    print()

# ── 4. 结论 ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("✓ 基础功能验证通过，vLLM v0.6.6 可正常运行")
print("=" * 60)
