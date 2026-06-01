import os
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
from vllm import LLM, SamplingParams

"""level 1 sleep 后 wake_up，推理结果不变"""
llm = LLM("/home/cb/model/Qwen2.5-1.5B-Instruct", enable_sleep_mode=True, max_model_len=2048)
prompt = "How are you?"
params = SamplingParams(temperature=0, max_tokens=10)

# 基准推理
output1 = llm.generate(prompt, params)

# Sleep level 1
free_before = torch.cuda.mem_get_info()[0]
llm.sleep(level=1)
free_after = torch.cuda.mem_get_info()[0]
print(f"Sleep level 1 释放显存: {(free_after - free_before) / 1024**3:.2f} GB")
assert free_after > free_before, "Sleep 后显存应增加"

# Wake up
llm.wake_up()
output2 = llm.generate(prompt, params)

assert output1[0].outputs[0].text == output2[0].outputs[0].text, \
    f"输出不一致！\nbefore: {output1[0].outputs[0].text}\nafter: {output2[0].outputs[0].text}"
print(f"✓ Level 1 sleep/wake_up 正确: '{output1[0].outputs[0].text}'")