"""
Sleep Mode Benchmark
====================
测量 sleep mode 各阶段的显存占用和耗时，与不开 sleep mode 进行对比。

用法：
    .venv/bin/python tests/sleep_mode/benchmark_sleep_mode.py

场景：
    1. baseline        — 不开 sleep mode，正常推理
    2. level1_sleep    — Level 1 sleep（权重 offload 到 CPU，KV cache 丢弃）
    3. level2_sleep    — Level 2 sleep（所有内存丢弃，wake_up 后 reload_weights）
    4. level2_partial  — Level 2 分步唤醒（先 weights，再 kv_cache）

每个场景独立运行（单独 subprocess），避免同进程多次初始化 CuMemAllocator 单例的问题。
"""
import os
import subprocess
import sys
import textwrap
import json
import time

# ─────────────────────────────────────────────────────────────────────────────
# 各场景的测量脚本（以字符串形式嵌入，通过 subprocess 执行）
# ─────────────────────────────────────────────────────────────────────────────

SCENARIO_BASELINE = textwrap.dedent("""
import os
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import time, json, gc
import torch
from vllm import LLM, SamplingParams

MODEL = "/home/cb/model/Qwen2.5-1.5B-Instruct"
PROMPT = "What is artificial intelligence? Please explain briefly."
PARAMS = SamplingParams(temperature=0, max_tokens=32)

result = {}

torch.cuda.empty_cache()
free_init, total = torch.cuda.mem_get_info()
result["total_gb"] = total / 1024**3
result["free_before_init_gb"] = free_init / 1024**3

t0 = time.perf_counter()
llm = LLM(MODEL, max_model_len=2048)
t1 = time.perf_counter()
result["init_time_s"] = t1 - t0

free_after_init, _ = torch.cuda.mem_get_info()
result["free_after_init_gb"] = free_after_init / 1024**3
result["used_after_init_gb"] = (total - free_after_init) / 1024**3

# warmup
_ = llm.generate(PROMPT, PARAMS)

# timed generate
t0 = time.perf_counter()
output = llm.generate(PROMPT, PARAMS)
t1 = time.perf_counter()
result["generate_time_s"] = t1 - t0
result["output"] = output[0].outputs[0].text

free_idle, _ = torch.cuda.mem_get_info()
result["free_idle_gb"] = free_idle / 1024**3
result["used_idle_gb"] = (total - free_idle) / 1024**3

print(json.dumps(result))
""")

SCENARIO_LEVEL1 = textwrap.dedent("""
import os
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import time, json, gc
import torch
from vllm import LLM, SamplingParams

MODEL = "/home/cb/model/Qwen2.5-1.5B-Instruct"
PROMPT = "What is artificial intelligence? Please explain briefly."
PARAMS = SamplingParams(temperature=0, max_tokens=32)

result = {}

torch.cuda.empty_cache()
free_init, total = torch.cuda.mem_get_info()
result["total_gb"] = total / 1024**3

t0 = time.perf_counter()
llm = LLM(MODEL, enable_sleep_mode=True, max_model_len=2048)
t1 = time.perf_counter()
result["init_time_s"] = t1 - t0

free_after_init, _ = torch.cuda.mem_get_info()
result["free_after_init_gb"] = free_after_init / 1024**3
result["used_after_init_gb"] = (total - free_after_init) / 1024**3

# warmup
_ = llm.generate(PROMPT, PARAMS)

# baseline generate (before sleep)
t0 = time.perf_counter()
output1 = llm.generate(PROMPT, PARAMS)
t1 = time.perf_counter()
result["generate_before_sleep_time_s"] = t1 - t0
result["output_before"] = output1[0].outputs[0].text

# ── SLEEP ──
free_before_sleep, _ = torch.cuda.mem_get_info()
t0 = time.perf_counter()
llm.sleep(level=1)
t1 = time.perf_counter()
result["sleep_time_s"] = t1 - t0

free_after_sleep, _ = torch.cuda.mem_get_info()
result["free_after_sleep_gb"] = free_after_sleep / 1024**3
result["used_after_sleep_gb"] = (total - free_after_sleep) / 1024**3
result["freed_by_sleep_gb"] = (free_after_sleep - free_before_sleep) / 1024**3

# ── WAKE_UP ──
t0 = time.perf_counter()
llm.wake_up()
t1 = time.perf_counter()
result["wakeup_time_s"] = t1 - t0

free_after_wakeup, _ = torch.cuda.mem_get_info()
result["free_after_wakeup_gb"] = free_after_wakeup / 1024**3
result["used_after_wakeup_gb"] = (total - free_after_wakeup) / 1024**3

# generate after wake_up
t0 = time.perf_counter()
output2 = llm.generate(PROMPT, PARAMS)
t1 = time.perf_counter()
result["generate_after_wakeup_time_s"] = t1 - t0
result["output_after"] = output2[0].outputs[0].text
result["output_match"] = (output1[0].outputs[0].text == output2[0].outputs[0].text)

print(json.dumps(result))
""")

SCENARIO_LEVEL2 = textwrap.dedent("""
import os
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import time, json, gc
import torch
from vllm import LLM, SamplingParams

MODEL = "/home/cb/model/Qwen2.5-1.5B-Instruct"
PROMPT = "What is artificial intelligence? Please explain briefly."
PARAMS = SamplingParams(temperature=0, max_tokens=32)

result = {}

torch.cuda.empty_cache()
free_init, total = torch.cuda.mem_get_info()
result["total_gb"] = total / 1024**3

t0 = time.perf_counter()
llm = LLM(MODEL, enable_sleep_mode=True, max_model_len=2048)
t1 = time.perf_counter()
result["init_time_s"] = t1 - t0

free_after_init, _ = torch.cuda.mem_get_info()
result["free_after_init_gb"] = free_after_init / 1024**3
result["used_after_init_gb"] = (total - free_after_init) / 1024**3

# warmup
_ = llm.generate(PROMPT, PARAMS)

t0 = time.perf_counter()
output1 = llm.generate(PROMPT, PARAMS)
t1 = time.perf_counter()
result["generate_before_sleep_time_s"] = t1 - t0
result["output_before"] = output1[0].outputs[0].text

# ── SLEEP level 2 ──
free_before_sleep, _ = torch.cuda.mem_get_info()
t0 = time.perf_counter()
llm.sleep(level=2)
t1 = time.perf_counter()
result["sleep_time_s"] = t1 - t0

free_after_sleep, _ = torch.cuda.mem_get_info()
result["free_after_sleep_gb"] = free_after_sleep / 1024**3
result["used_after_sleep_gb"] = (total - free_after_sleep) / 1024**3
result["freed_by_sleep_gb"] = (free_after_sleep - free_before_sleep) / 1024**3

# ── WAKE_UP all ──
t0 = time.perf_counter()
llm.wake_up()
t1 = time.perf_counter()
result["wakeup_time_s"] = t1 - t0

# reload weights from disk
t0 = time.perf_counter()
llm.collective_rpc("reload_weights")
t1 = time.perf_counter()
result["reload_weights_time_s"] = t1 - t0

free_after_wakeup, _ = torch.cuda.mem_get_info()
result["free_after_wakeup_gb"] = free_after_wakeup / 1024**3
result["used_after_wakeup_gb"] = (total - free_after_wakeup) / 1024**3

t0 = time.perf_counter()
output2 = llm.generate(PROMPT, PARAMS)
t1 = time.perf_counter()
result["generate_after_wakeup_time_s"] = t1 - t0
result["output_after"] = output2[0].outputs[0].text
result["output_match"] = (output1[0].outputs[0].text == output2[0].outputs[0].text)

print(json.dumps(result))
""")

SCENARIO_LEVEL2_PARTIAL = textwrap.dedent("""
import os
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import time, json, gc
import torch
from vllm import LLM, SamplingParams

MODEL = "/home/cb/model/Qwen2.5-1.5B-Instruct"
PROMPT = "What is artificial intelligence? Please explain briefly."
PARAMS = SamplingParams(temperature=0, max_tokens=32)

result = {}

torch.cuda.empty_cache()
free_init, total = torch.cuda.mem_get_info()
result["total_gb"] = total / 1024**3

t0 = time.perf_counter()
llm = LLM(MODEL, enable_sleep_mode=True, max_model_len=2048)
t1 = time.perf_counter()
result["init_time_s"] = t1 - t0

free_after_init, _ = torch.cuda.mem_get_info()
result["free_after_init_gb"] = free_after_init / 1024**3
result["used_after_init_gb"] = (total - free_after_init) / 1024**3

_ = llm.generate(PROMPT, PARAMS)

t0 = time.perf_counter()
output1 = llm.generate(PROMPT, PARAMS)
t1 = time.perf_counter()
result["generate_before_sleep_time_s"] = t1 - t0
result["output_before"] = output1[0].outputs[0].text

# ── SLEEP level 2 ──
free_before_sleep, _ = torch.cuda.mem_get_info()
t0 = time.perf_counter()
llm.sleep(level=2)
t1 = time.perf_counter()
result["sleep_time_s"] = t1 - t0

free_after_sleep, _ = torch.cuda.mem_get_info()
result["free_after_sleep_gb"] = free_after_sleep / 1024**3
result["used_after_sleep_gb"] = (total - free_after_sleep) / 1024**3
result["freed_by_sleep_gb"] = (free_after_sleep - free_before_sleep) / 1024**3

# ── 分步唤醒：Step 1 wake weights ──
t0 = time.perf_counter()
llm.wake_up(tags=["weights"])
t1 = time.perf_counter()
result["wakeup_weights_time_s"] = t1 - t0

free_after_weights, _ = torch.cuda.mem_get_info()
result["free_after_wakeup_weights_gb"] = free_after_weights / 1024**3
result["used_after_wakeup_weights_gb"] = (total - free_after_weights) / 1024**3

# reload weights (RLHF scenario)
t0 = time.perf_counter()
llm.collective_rpc("reload_weights")
t1 = time.perf_counter()
result["reload_weights_time_s"] = t1 - t0

# ── 分步唤醒：Step 2 wake kv_cache ──
t0 = time.perf_counter()
llm.wake_up(tags=["kv_cache"])
t1 = time.perf_counter()
result["wakeup_kvcache_time_s"] = t1 - t0

free_after_full, _ = torch.cuda.mem_get_info()
result["free_after_wakeup_full_gb"] = free_after_full / 1024**3
result["used_after_wakeup_full_gb"] = (total - free_after_full) / 1024**3

t0 = time.perf_counter()
output2 = llm.generate(PROMPT, PARAMS)
t1 = time.perf_counter()
result["generate_after_wakeup_time_s"] = t1 - t0
result["output_after"] = output2[0].outputs[0].text
result["output_match"] = (output1[0].outputs[0].text == output2[0].outputs[0].text)

print(json.dumps(result))
""")

# ─────────────────────────────────────────────────────────────────────────────
# 运行单个场景
# ─────────────────────────────────────────────────────────────────────────────

def run_scenario(name: str, script: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  Running scenario: {name}")
    print(f"{'='*60}")

    python = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "../../.venv/bin/python")
    if not os.path.exists(python):
        python = sys.executable

    env = os.environ.copy()
    t_wall_start = time.perf_counter()
    proc = subprocess.run(
        [python, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."),
    )
    t_wall = time.perf_counter() - t_wall_start

    if proc.returncode != 0:
        print(f"[ERROR] scenario '{name}' failed (exit {proc.returncode})")
        print("STDOUT:", proc.stdout[-3000:] if proc.stdout else "(empty)")
        print("STDERR:", proc.stderr[-3000:] if proc.stderr else "(empty)")
        return {"_error": True, "_wall_time_s": t_wall}

    # 取最后一行（JSON）
    lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    if not lines:
        print(f"[ERROR] no output from scenario '{name}'")
        return {"_error": True}

    try:
        data = json.loads(lines[-1])
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parse failed: {e}")
        print("Raw output:", proc.stdout[-2000:])
        return {"_error": True}

    data["_wall_time_s"] = t_wall
    print(f"  OK  (wall {t_wall:.1f}s)")
    return data

# ─────────────────────────────────────────────────────────────────────────────
# 格式化输出
# ─────────────────────────────────────────────────────────────────────────────

def fmt_gb(v) -> str:
    if v is None:
        return "  N/A  "
    return f"{v:6.2f} GB"

def fmt_s(v) -> str:
    if v is None:
        return "   N/A  "
    return f"{v:7.3f}s"

def fmt_bool(v) -> str:
    if v is None:
        return "N/A"
    return "✓" if v else "✗ MISMATCH"

def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

def row(label: str, *values):
    label_col = f"  {label:<38}"
    vals = "  ".join(f"{v:>12}" for v in values)
    print(label_col + vals)

def print_report(results: dict[str, dict]):
    scenarios = list(results.keys())
    total = results.get("baseline", {}).get("total_gb")

    def pct(gb):
        if gb is None or total is None or total == 0:
            return ""
        return f"({gb/total*100:4.1f}%)"

    def used_str(d, key):
        v = d.get(key)
        if v is None:
            return "   N/A  "
        return f"{v:5.2f}GB {pct(v)}"

    print("\n")
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                      Sleep Mode Benchmark Results                           ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")

    # ── GPU 总量 ──
    for name, d in results.items():
        if not d.get("_error") and d.get("total_gb"):
            print(f"\n  GPU Total Memory : {d['total_gb']:.2f} GB")
            break

    # ── 初始化 ──
    section("① 初始化 (LLM.__init__ + load_model + initialize_cache)")
    header = f"  {'':38}" + "  ".join(f"{'':>12}" for _ in scenarios)
    names_row = f"  {'':38}" + "  ".join(f"{n:>12}" for n in scenarios)
    print(names_row)
    print(f"  {'─'*38}" + "─" * (14 * len(scenarios)))

    def get(name, key):
        d = results.get(name, {})
        return d.get(key)

    row("init time",
        *[fmt_s(get(n, "init_time_s")) for n in scenarios])
    row("used after init",
        *[used_str(results.get(n, {}), "used_after_init_gb") for n in scenarios])
    row("free after init",
        *[used_str(results.get(n, {}), "free_after_init_gb") for n in scenarios])

    # ── 推理（sleep 前）──
    section("② 推理 (sleep 前 / baseline 正常)")
    row("generate time",
        *[fmt_s(get(n, "generate_before_sleep_time_s") or get(n, "generate_time_s"))
          for n in scenarios])

    # ── Sleep ──
    section("③ Sleep")
    row("sleep time",
        *[fmt_s(get(n, "sleep_time_s")) for n in scenarios])
    row("freed by sleep",
        *[fmt_gb(get(n, "freed_by_sleep_gb")) for n in scenarios])
    row("used after sleep",
        *[used_str(results.get(n, {}), "used_after_sleep_gb") for n in scenarios])
    row("free after sleep",
        *[used_str(results.get(n, {}), "free_after_sleep_gb") for n in scenarios])

    # ── Wake_up ──
    section("④ Wake_up")
    for n in scenarios:
        d = results.get(n, {})
        if d.get("wakeup_weights_time_s") is not None:
            # partial wakeup: two steps
            row(f"  [{n}] wake weights",
                fmt_s(d.get("wakeup_weights_time_s")))
            row(f"  [{n}] reload weights",
                fmt_s(d.get("reload_weights_time_s")))
            row(f"  [{n}] used after weights",
                used_str(d, "used_after_wakeup_weights_gb"))
            row(f"  [{n}] wake kv_cache",
                fmt_s(d.get("wakeup_kvcache_time_s")))
        elif d.get("wakeup_time_s") is not None:
            row(f"  [{n}] wake_up (full)",
                fmt_s(d.get("wakeup_time_s")))
            if d.get("reload_weights_time_s") is not None:
                row(f"  [{n}] reload weights",
                    fmt_s(d.get("reload_weights_time_s")))
    print()
    row("used after full wakeup",
        *[used_str(results.get(n, {}),
                   "used_after_wakeup_gb" if "used_after_wakeup_gb" in results.get(n, {})
                   else "used_after_wakeup_full_gb")
          for n in scenarios])

    # ── 推理（wake 后）──
    section("⑤ 推理 (wake_up 后)")
    row("generate time",
        *[fmt_s(get(n, "generate_after_wakeup_time_s") or get(n, "generate_time_s"))
          for n in scenarios])
    row("output match",
        *[fmt_bool(get(n, "output_match")) for n in scenarios])

    # ── 总计 ──
    section("⑥ 总计 (wall clock)")
    row("wall time (total)",
        *[fmt_s(get(n, "_wall_time_s")) for n in scenarios])

    # ── 生成内容 ──
    section("⑦ 生成内容")
    for n in scenarios:
        d = results.get(n, {})
        text = d.get("output_before") or d.get("output")
        if text:
            print(f"  [{n}] {repr(text[:80])}")

    print()

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    SCENARIOS = {
        "baseline":       SCENARIO_BASELINE,
        "level1":         SCENARIO_LEVEL1,
        "level2":         SCENARIO_LEVEL2,
        "level2_partial": SCENARIO_LEVEL2_PARTIAL,
    }

    results = {}
    for name, script in SCENARIOS.items():
        results[name] = run_scenario(name, script)

    print_report(results)

    # 保存 JSON 结果
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Raw results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
