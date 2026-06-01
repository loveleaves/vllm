import os
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
from vllm import LLM, SamplingParams

"""level 2 sleep + 分步唤醒（先 weights 后 kv_cache）"""
llm = LLM("/home/cb/model/Qwen2.5-1.5B-Instruct", enable_sleep_mode=True, max_model_len=2048)
prompt = "What is AI?"
params = SamplingParams(temperature=0, max_tokens=10)

output1 = llm.generate(prompt, params)

# Step 1：deep sleep，释放所有 GPU 内存
llm.sleep(level=2)
free_after_sleep = torch.cuda.mem_get_info()[0]
print(f"Level 2 sleep 后可用显存: {free_after_sleep / 1024**3:.2f} GB")

# Step 2：只唤醒权重（此时 KV Cache 还未分配，不能 generate）
llm.wake_up(tags=["weights"])
free_after_weights = torch.cuda.mem_get_info()[0]
print(f"唤醒 weights 后可用显存: {free_after_weights / 1024**3:.2f} GB")

# RLHF 场景：这里重新加载原始权重（验证 reload_weights 机制）
llm.collective_rpc("reload_weights")

# Step 3：唤醒 KV Cache，恢复完整推理能力
llm.wake_up(tags=["kv_cache"])

output2 = llm.generate(prompt, params)
assert output1[0].outputs[0].text == output2[0].outputs[0].text, \
    f"输出不一致！\nbefore: {output1[0].outputs[0].text}\nafter: {output2[0].outputs[0].text}"
print(f"✓ Level 2 分步唤醒正确: '{output1[0].outputs[0].text}'")