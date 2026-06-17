# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM v0.15.1 最小离线推理示例。

对标 nano-vllm/example.py，但使用真实 vLLM 的 API。两者接口很接近，
方便从 nano-vllm 过渡到 vLLM 的学习。主要差异：

  1. 入口对象都叫 ``LLM`` / ``SamplingParams``，从顶层包导入即可。
  2. nano-vllm 的 ``llm.generate`` 返回 ``list[dict]``，用 ``output["text"]`` 取文本；
     真实 vLLM 返回 ``list[RequestOutput]``，文本在 ``output.outputs[0].text``，
     原始 prompt 在 ``output.prompt``（见 vllm/outputs.py 的 RequestOutput / CompletionOutput）。
  3. 构造 chat prompt 有两种写法：
       a) 手动用 tokenizer.apply_chat_template 拼模板，再调 llm.generate（与 nano-vllm 一致）；
       b) 直接用 llm.chat(messages)，vLLM 内部自动套模板（更省事，nano-vllm 没有）。

运行前先准备好本地模型权重，例如：
    huggingface-cli download Qwen/Qwen3-0.6B --local-dir ~/model/Qwen3-0.6B
或设置环境变量 VLLM_EXAMPLE_MODEL 指向任意本地/HF 模型路径。

用法：
    python example.py
"""

import os

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams


def main():
    # 模型路径：默认读 ~/model/Qwen3-0.6B，可用 VLLM_EXAMPLE_MODEL 覆盖。
    path = os.environ.get(
        "VLLM_EXAMPLE_MODEL", os.path.expanduser("~/model/Qwen3-0.6B")
    )

    # enforce_eager=True 关闭 CUDA Graph 捕获，启动更快、显存更省，便于调试/学习；
    # 生产里去掉它可获得更高 decode 吞吐。tensor_parallel_size=1 即单卡。
    llm = LLM(model=path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)

    raw_prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]

    # ---- 写法 a：手动套 chat 模板 + generate（对标 nano-vllm/example.py） ----
    tokenizer = AutoTokenizer.from_pretrained(path)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in raw_prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    print("=" * 20, "llm.generate", "=" * 20)
    for output in outputs:
        # RequestOutput.prompt 是送入的完整 prompt，outputs[0] 是首个候选补全。
        print(f"\nPrompt:     {output.prompt!r}")
        print(f"Completion: {output.outputs[0].text!r}")

    # ---- 写法 b：llm.chat 直接传 messages，由 vLLM 内部套模板（更简洁） ----
    conversations = [[{"role": "user", "content": p}] for p in raw_prompts]
    chat_outputs = llm.chat(conversations, sampling_params)

    print("\n" + "=" * 20, "llm.chat", "=" * 20)
    for prompt, output in zip(raw_prompts, chat_outputs):
        print(f"\nQuestion: {prompt!r}")
        print(f"Answer:   {output.outputs[0].text!r}")


if __name__ == "__main__":
    main()
