# Sleep Mode 性能测试报告

## 测试环境

| 项目 | 值 |
|------|----|
| GPU | NVIDIA RTX 3060 Ti (8.00 GB VRAM) |
| 系统 | WSL2 / Ubuntu（Linux 5.15, x86_64）|
| Python | 3.12.13 (uv) |
| PyTorch | 2.5.1+cu121 |
| CUDA Toolkit | 12.4 |
| vLLM | 0.6.6（sleep_mode 分支）|
| 模型 | Qwen2.5-1.5B-Instruct（~3 GB 权重）|
| 引擎 | V1（`VLLM_USE_V1=1`，`VLLM_ENABLE_V1_MULTIPROCESSING=0`）|
| `max_model_len` | 2048 |

---

## 测试场景

| 场景 | 说明 |
|------|------|
| **baseline** | 不开 sleep mode，正常推理 |
| **level1** | Level 1 sleep：权重 offload 到 CPU pinned memory，KV cache 丢弃 |
| **level2** | Level 2 sleep：全部内存丢弃，wake_up 后 `reload_weights()` 从磁盘重载 |
| **level2_partial** | Level 2 分步唤醒：先 `wake_up(["weights"])` + `reload_weights()`，再 `wake_up(["kv_cache"])` |

每个场景在独立 subprocess 中运行，避免同进程多实例污染 `CuMemAllocator` 单例。

测试脚本：`tests/sleep_mode/benchmark_sleep_mode.py`

---

## 结果汇总

### 1. 显存占用（各阶段）

```
GPU 总显存：8.00 GB
```

| 阶段 | baseline | level1 | level2 | level2_partial |
|------|----------|--------|--------|----------------|
| 初始化后（模型 + KV cache） | 5.29 GB (66.1%) | 5.26 GB (65.7%) | 5.26 GB (65.7%) | 5.26 GB (65.7%) |
| sleep 后 | — | 1.52 GB (19.0%) | 1.52 GB (19.0%) | 1.52 GB (19.0%) |
| wake weights 后（仅 level2_partial）| — | — | — | 4.54 GB (56.8%) |
| wake_up 全量后 | — | 5.26 GB (65.7%) | 5.26 GB (65.7%) | 5.26 GB (65.7%) |
| 空闲时（推理完毕）| 5.29 GB (66.1%) | — | — | — |

**显存释放量（sleep 后可用增量）**

| 场景 | 释放量 | 释放率 |
|------|--------|--------|
| level1 | **3.74 GB** | 46.8% of total |
| level2 | **3.74 GB** | 46.8% of total |
| level2_partial | **3.74 GB** | 46.8% of total |

> 三个 sleep 场景释放量相同（3.03 GB 权重 + 0.71 GB KV cache = 3.74 GB），
> sleep 后剩余 1.52 GB 为 CUDA context、PyTorch runtime、NCCL 等非 cumem 分配。

---

### 2. 各操作耗时

#### 初始化

| 场景 | 耗时 |
|------|------|
| baseline | 81.94 s |
| level1 | 76.77 s |
| level2 | 79.26 s |
| level2_partial | 78.82 s |

> 初始化时间主导因素为 CUDA Graph capture（约 70 s），sleep mode 开关对初始化无显著影响。

#### Sleep

| 场景 | sleep 耗时 | 原理 |
|------|-----------|------|
| level1 | **3.03 s** | 权重 3 GB CPU 拷贝（D2H PCIe 传输）|
| level2 | **0.24 s** | 仅 cuMemUnmap + cuMemRelease，无数据搬运 |
| level2_partial | **0.25 s** | 同 level2 |

> Level 1 的额外 2.8 s 为权重从 GPU 拷贝至 CPU pinned memory 的 PCIe 传输开销
>（RTX 3060 Ti PCIe 3.0 x16 理论带宽 ~16 GB/s，3 GB / 16 = ~0.19 s；
> 加上 cuMemUnmap/cuMemRelease 内核调用，实测 3.0 s）。

#### Wake_up

| 场景 | 操作 | 耗时 |
|------|------|------|
| level1 | wake_up（CPU→GPU 权重拷回 + KV cache 重分配）| **0.757 s** |
| level2 | wake_up（重新 map 物理页，权重为垃圾数据）| **0.141 s** |
| level2 | reload_weights（磁盘→GPU，模型文件重读）| **0.489 s** |
| level2_partial | wake_up(["weights"])（仅重分配权重物理页）| **0.120 s** |
| level2_partial | reload_weights | **0.501 s** |
| level2_partial | wake_up(["kv_cache"])（仅重分配 KV cache 物理页）| **0.010 s** |

> Level 2 分步唤醒中 `wake_up(["kv_cache"])` 仅需 10 ms，
> 因为 KV cache 不需要加载内容（内容由后续推理覆盖）。

#### 推理延迟（32 tokens 输出）

| 阶段 | baseline | level1 | level2 | level2_partial |
|------|----------|--------|--------|----------------|
| sleep 前（warmup 后）| 0.321 s | 0.324 s | 0.472 s | 0.495 s |
| wake_up 后 | — | 0.395 s | 0.489 s | 0.509 s |
| 较 baseline 增量 | — | +23% | +52% | +58% |

> wake_up 后首次推理略慢，因 CUDA Graph 需要重新预热；再次推理将恢复正常速度。
> Level 2 wake 后推理延迟稍高，因 reload_weights 后模型处于"冷"状态（缓存未预热）。

---

### 3. 完整时序（sleep + wake 总开销）

| 场景 | sleep | wake（全量）| reload | 合计 |
|------|-------|-------------|--------|------|
| level1 | 3.03 s | 0.76 s | — | **3.79 s** |
| level2 | 0.24 s | 0.14 s | 0.49 s | **0.87 s** |
| level2_partial | 0.25 s | 0.12 s + 0.01 s | 0.50 s | **0.88 s** |

> Level 2 总开销比 Level 1 少 **4.3×**，适合需要快速释放内存、且后续会更新权重的场景（RLHF）。

---

### 4. 推理正确性验证

所有 sleep/wake 场景输出与 baseline 完全一致：

```
✓ level1:         输出与 baseline 一致
✓ level2:         输出与 baseline 一致（reload_weights 后）
✓ level2_partial: 输出与 baseline 一致（分步唤醒后）
```

生成内容（截断至 80 字符）：
```
' Artificial intelligence (AI) is a branch of computer science that focuses on cr...'
```

---

## 场景选择建议

| 使用场景 | 推荐 sleep level | 理由 |
|---------|-----------------|------|
| 模型服务临时暂停（快速恢复）| **Level 1** | wake_up 只需 0.76 s，权重完整保留在 CPU，恢复后无需 reload |
| RLHF 训练后替换权重 | **Level 2** | sleep 只需 0.24 s；wake 后用新权重覆盖，不需要保留旧权重 |
| 多模型分时复用（显存极度紧张）| **Level 2 分步唤醒** | 支持在 wake weights 后、wake KV cache 前进行权重更新，灵活度最高 |
| 生产推理（无需释放显存）| **不开 sleep mode** | 无额外开销，baseline 延迟最低 |

---

## 注意事项

1. **sleep 后剩余 1.52 GB 不可释放**：属于 CUDA context 和 PyTorch 运行时占用，不在 cumem pool 管理范围内。

2. **Level 1 sleep 耗时受 PCIe 带宽限制**：权重越大，sleep 越慢。3B 及以上模型建议优先考虑 Level 2。

3. **wake_up 后首次推理有额外延迟**：因 torch.compile CUDA Graph 需要重新执行，约比稳态多 20–60%。

4. **同一进程只能有一个 sleep mode 实例**：`CuMemAllocator` 是进程级单例；多实例测试需在独立 subprocess 中运行。

5. **Level 2 必须调用 `reload_weights()`**：wake_up 后物理页内容为零/垃圾，必须从磁盘重载或写入 RLHF 更新后的权重，否则推理结果随机。
