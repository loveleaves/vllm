# vLLM Sleep Mode 深度设计文档

> **目标读者**：不熟悉 vLLM 的开发者。读完本文后，你应能：
> 1. 理解 sleep mode 的设计动机与完整实现原理
> 2. 在旧版 vLLM 上从零实现 sleep mode
> 3. 掌握 vLLM 五层架构的核心模块和协作方式
> 4. 具备独立设计 Multi-Token Prediction (MTP) 等新特性的能力

---

## 目录

1. [为什么需要 Sleep Mode](#1-为什么需要-sleep-mode)
2. [CUDA 虚拟内存管理：实现的物理基础](#2-cuda-虚拟内存管理实现的物理基础)
3. [vLLM 五层架构总览](#3-vllm-五层架构总览)
4. [Sleep Mode 功能设计](#4-sleep-mode-功能设计)
5. [逐层源码精读](#5-逐层源码精读)
6. [PR 演进历史：设计如何从简单走向健壮](#6-pr-演进历史设计如何从简单走向健壮)
7. [从零实现：分步骤操作手册](#7-从零实现分步骤操作手册)
8. [vLLM 模块学习路线图](#8-vllm-模块学习路线图)
9. [举一反三：独立设计 Multi-Token Prediction](#9-举一反三独立设计-multi-token-prediction)
10. [核心文件速查表](#10-核心文件速查表)

---

## 1. 为什么需要 Sleep Mode

### 1.1 RLHF 训练流水线的痛点

强化学习人类反馈（RLHF）是大模型对齐训练的核心流程，典型流水线如下：

```
┌─────────────────────────────────────────────────────┐
│  RLHF 训练循环                                       │
│                                                     │
│  ① vLLM 推理  →  生成 rollout 数据                  │
│       ↓                                             │
│  ② 训练框架 (FSDP/DeepSpeed)  →  更新模型权重       │
│       ↓                                             │
│  ③ 同步新权重到 vLLM                                │
│       ↓  (回到步骤①)                               │
└─────────────────────────────────────────────────────┘
```

**核心矛盾**：步骤①和步骤②**不能同时占用 GPU**。
- 步骤①：vLLM 需要 GPU 存放模型权重（如 7B 模型 = 14GB bfloat16）+ KV Cache（通常 10-40GB）
- 步骤②：训练框架需要 GPU 存放参数 + 梯度 + 优化器状态（AdamW 约需权重 × 3 的显存）

**传统解法**：停掉 vLLM 进程 → 训练 → 重启 vLLM 进程。

**问题**：每次重启需要：
- 重新从磁盘加载模型权重（几十秒）
- 重新编译 CUDA Graph（几十秒到几分钟）
- 重新做 warmup 推理

一个 RLHF 训练循环可能每隔几分钟就要切换一次，重启开销不可接受。

### 1.2 Sleep Mode 的解决思路

**核心洞察**：GPU 显存可以分为两类：
- **物理内存**（Physical Memory）：实际占用显存芯片的内存页，影响`nvidia-smi`显示
- **虚拟地址**（Virtual Address）：CPU 和 GPU 程序中用来访问内存的指针值

普通 `cudaMalloc` 把这两者绑在一起。但 CUDA Driver API 的 `cuMem*` 系列函数允许把它们分开管理。

**Sleep Mode 的方案**：
- **初始化时**：用 `cuMem*` API 分配权重和 KV Cache，**记录每块内存的虚拟地址和物理页句柄**
- **Sleep 时**：解除虚拟地址到物理页的映射，释放物理页。PyTorch 张量对象的 `data_ptr()` **不变**，但此时访问会报错
- **Wake up 时**：重新分配物理页，映射回**同一个虚拟地址**，再把数据（如有备份）写回

结果：整个过程中，模型对象、张量的指针完全不变。CUDA Graph 里面硬编码的地址也完全不变。**完全透明**，模型代码零修改。

---

## 2. CUDA 虚拟内存管理：实现的物理基础

### 2.1 cuMem API 的三步操作

```
普通 cudaMalloc：  [预留虚拟地址] + [分配物理页] + [建立映射]  ← 三步合一
cuMem API：       三步分开，可单独控制
```

**分配（Sleep 前，初始化阶段）**：

```c
// Step 1：只预留虚拟地址范围（不占物理显存）
cuMemAddressReserve(&d_mem, aligned_size, granularity, 0, 0);

// Step 2：分配物理内存页（真正占显存，得到句柄）
CUmemAllocationProp prop = { .type = CU_MEM_ALLOCATION_TYPE_PINNED, ... };
cuMemCreate(&p_memHandle, aligned_size, &prop, 0);

// Step 3：建立虚拟地址到物理页的映射
cuMemMap(d_mem, aligned_size, 0, p_memHandle, 0);
cuMemSetAccess(d_mem, aligned_size, &accessDesc, 1);  // 设置可读写
```

此后 `d_mem` 就是一个有效的 GPU 指针，可以被 PyTorch 张量使用。

**Sleep（释放物理显存）**：

```c
// 解除映射（虚拟地址仍然预留，但访问会崩溃）
cuMemUnmap(d_mem, aligned_size);
// 释放物理页（显存还给系统）
cuMemRelease(p_memHandle);
// ⚠️ 注意：cuMemAddressFree 不调用！虚拟地址保留！
```

**Wake up（重新建立映射）**：

```c
// 重新分配物理页
cuMemCreate(&new_handle, aligned_size, &prop, 0);
// 映射回同一个虚拟地址
cuMemMap(d_mem, aligned_size, 0, new_handle, 0);
cuMemSetAccess(d_mem, aligned_size, &accessDesc, 1);
```

**关键：`d_mem` 值在整个过程中从未改变。** PyTorch 张量的 `data_ptr()` 永远是同一个地址。

### 2.2 对齐要求

cuMem API 要求分配大小按粒度对齐：

```c
size_t granularity;
cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
// NVIDIA GPU 通常是 2MB
size_t aligned_size = ((size + granularity - 1) / granularity) * granularity;
```

### 2.3 为什么其他方案不可行

作者在代码注释中明确写道（`cumem.py` 第 6-9 行）：

```python
# other approaches tried but failed:
# - cuda-python package binding
# - custom libcuda driver ctypes wrapper
# both of them failed because of cuda context mismatch.
# not sure why, they are created from a different context.
# the only successful approach is to call cuda driver API in C.
```

Python 层调用 CUDA Driver API 会遇到 CUDA context 不匹配的问题。必须写 C 扩展，在与 PyTorch 相同的 context 中调用。

---

## 3. vLLM 五层架构总览

理解 sleep mode 需要先理解 vLLM 的分层结构。每一层职责明确，sleep mode 的实现就是在每一层加入对应的 sleep/wake_up 操作。

```
┌──────────────────────────────────────────────────────────────────┐
│  Layer 5：Entrypoint（用户入口）                                  │
│  vllm/entrypoints/llm.py          ← 离线推理：LLM.generate()     │
│  vllm/entrypoints/serve/          ← 在线服务：OpenAI API server  │
│  职责：接收用户输入，返回生成结果；暴露 sleep/wake_up 公共 API    │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│  Layer 4：Engine（推理引擎）                                      │
│  vllm/v1/engine/core.py           ← EngineCore: 调度 + 执行协调  │
│  vllm/v1/engine/llm_engine.py     ← LLMEngine: 同步引擎          │
│  vllm/v1/engine/async_llm.py      ← AsyncLLM: 异步引擎           │
│  职责：调度器管理、请求生命周期、sleep 时协调调度暂停 + GPU 释放  │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│  Layer 3：Executor（执行器）                                      │
│  vllm/v1/executor/abstract.py     ← 抽象基类，定义 sleep/wake_up │
│  vllm/v1/executor/multiproc_executor.py  ← 多进程（TP > 1）      │
│  vllm/v1/executor/uniproc_executor.py    ← 单进程（TP = 1）      │
│  职责：管理一组 Worker，通过 collective_rpc 广播命令到所有 GPU   │
└──────────────────────────┬───────────────────────────────────────┘
                           │ collective_rpc("sleep") / collective_rpc("wake_up")
┌──────────────────────────▼───────────────────────────────────────┐
│  Layer 2：Worker（工作单元）                                      │
│  vllm/v1/worker/gpu_worker.py     ← 单 GPU 的所有操作            │
│  vllm/v1/worker/gpu_model_runner.py ← 模型前向推理               │
│  职责：在单 GPU 上加载模型、运行推理、分配 KV Cache；             │
│         sleep 时调用 CuMemAllocator，wake_up 时恢复并修复状态     │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│  Layer 1：Allocator + Platform（内存分配器 + 平台抽象）           │
│  vllm/device_allocator/cumem.py   ← CuMemAllocator（核心实现）   │
│  csrc/cumem_allocator.cpp         ← C 扩展（调用 CUDA Driver API）│
│  vllm/platforms/interface.py      ← 平台能力检测                  │
│  职责：管理 GPU 虚拟内存池；实际执行 unmap/remap 操作             │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│  配置层（贯穿所有层）                                             │
│  vllm/config/model.py             ← ModelConfig（含 enable_sleep_mode） │
│  vllm/config/vllm.py              ← VllmConfig（聚合所有配置）    │
│  vllm/engine/arg_utils.py         ← CLI 参数 → Config 对象        │
└──────────────────────────────────────────────────────────────────┘
```

**数据流**：用户调用 `llm.sleep()` → Engine 暂停调度器 → Engine 通知 Executor → Executor 通过 `collective_rpc` 广播到所有 Worker → Worker 调用 `CuMemAllocator.sleep()` → C 扩展调用 `cuMemUnmap + cuMemRelease`。

---

## 4. Sleep Mode 功能设计

### 4.1 三级睡眠模式

| Level | 模型权重 | 模型 Buffers | KV Cache | 调度器 | 典型场景 |
|-------|---------|-------------|---------|--------|---------|
| **0** | 保留 GPU | 保留 GPU | 保留 GPU | 暂停（PAUSED_NEW） | 仅暂停接受新请求，等 in-flight 请求处理完 |
| **1** | offload → CPU pinned memory | 保留 CPU | 丢弃 | 暂停 + 清 prefix cache | RLHF：同一个模型反复生成 rollout |
| **2** | 丢弃（不保留） | clone → CPU | 丢弃 | 暂停 + 清 prefix cache | RLHF：训练后权重已更新，旧权重无用 |

**Buffers 的特殊处理（Level 2）**：模型的 `named_buffers()` 包含不参与训练但影响推理的常量，如 RoPE 的 `cos`/`sin` 缓存、专家路由的 `expert_map` 等。这些 tensor 通常**不在** cumem 内存池中（因为它们在模型构造时就已分配），level 2 sleep 时需要手动 clone 到 CPU，wake_up 时手动恢复。

### 4.2 Tag 系统：内存的身份证

CuMemAllocator 给每一块在 cumem 池中分配的内存打一个字符串标签：

```
tag = "weights"   → Worker.load_model() 期间分配的所有张量（模型参数）
tag = "kv_cache"  → Worker.initialize_from_config() 期间分配的所有张量
tag = "default"   → 其他（临时 tensor，通常不在 pool 中）
```

**Sleep 时的 `offload_tags` 参数**：
- `offload_tags=("weights",)`（level 1）：weights 备份到 CPU，kv_cache 直接丢弃
- `offload_tags=()`（level 2）：所有 tag 的内存直接丢弃，无 CPU 备份

**Wake up 时的 `tags` 参数**：
- `tags=None`：一次性恢复所有标签
- `tags=["weights"]`：只恢复权重，不分配 KV Cache（避免 OOM，先更新权重）
- `tags=["kv_cache"]`：只恢复 KV Cache

### 4.3 分步唤醒（Partial Wake Up）：RLHF 的关键

RLHF 中更新权重的完整流程（Level 2）：

```python
# Step 1：深度睡眠，释放所有 GPU 显存
llm.sleep(level=2)
# GPU 显存: ~0 GB（只剩 CUDA runtime 自身占用）

# Step 2：训练框架接管 GPU 显存，执行反向传播 + 权重更新
trainer.train_one_step()
# GPU 显存: 被训练框架占满

# Step 3：训练完毕，只唤醒权重内存（虚拟地址已保留，重新映射物理页）
llm.wake_up(tags=["weights"])
# GPU 显存: 仅权重大小（如 7B bfloat16 = 14GB），无 KV Cache

# Step 4：训练框架将新权重写入权重内存（原地更新，地址不变）
llm.collective_rpc("reload_weights")
# GPU 显存: 仍然只有权重

# Step 5：唤醒 KV Cache，恢复完整推理能力
llm.wake_up(tags=["kv_cache"])
# GPU 显存: 权重 + KV Cache，可以推理了

output = llm.generate(...)
```

如果直接 `wake_up()`（不分步），峰值显存 = 权重 + KV Cache，容易 OOM（因为训练框架可能还没完全释放显存）。

### 4.4 调度器暂停模式（PauseMode）

Sleep 前必须先停止调度器处理请求。有三种模式（`Literal["abort", "wait", "keep"]`）：

| Mode | 行为 | 适用场景 |
|------|------|---------|
| `"abort"`（默认） | 立即中止所有 in-flight 请求（状态设为 ABORTED），等 abort 输出发送完后 sleep | 快速释放；RLHF 通常在一轮 rollout 结束后调用，此时无 in-flight 请求 |
| `"wait"` | 等所有 in-flight 请求正常完成后再 sleep | 不能丢请求的场景（仅异步引擎支持） |
| `"keep"` | 暂停调度但保留请求在队列（PAUSED_ALL），wake_up 后继续处理 | Level 0 的典型用法 |

---

## 5. 逐层源码精读

### 5.1 Layer 1a：C 扩展（`csrc/cumem_allocator.cpp`）

这是整个 sleep mode 的最底层，共 754 行 C 代码。

**导出给 Python 的三个函数**：

```c
// 1. 初始化：将 Python 的 malloc/free 回调注册为全局变量
static PyObject* py_init_module(PyObject* self, PyObject* args) {
    // 从 args 提取两个 Python callable
    g_python_malloc_callback = malloc_callback;  // 全局变量
    g_python_free_callback = free_callback;      // 全局变量
}

// 2. wake_up 时用：将物理内存重新映射到已保留的虚拟地址
static PyObject* python_create_and_map(PyObject* self, PyObject* args) {
    // 解包 (device, size, d_mem, p_memHandle)
    create_and_map(recv_device, recv_size, d_mem_ptr, p_memHandle);
}

// 3. sleep 时用：解除映射并释放物理内存
static PyObject* python_unmap_and_release(PyObject* self, PyObject* args) {
    unmap_and_release(recv_device, recv_size, d_mem_ptr, p_memHandle);
}
```

**`my_malloc`（PyTorch 分配张量时调用）**：

```c
void* my_malloc(ssize_t size, int device, CUstream stream) {
    // 1. 对齐到 granularity（通常 2MB）
    size_t alignedSize = align_up(size, granularity);
    
    // 2. 只预留虚拟地址（不占物理显存）
    CUdeviceptr d_mem;
    cuMemAddressReserve(&d_mem, alignedSize, 0, 0, 0);
    
    // 3. 分配 handle 结构体（在 CPU 堆上）
    CUmemGenericAllocationHandle* p_memHandle = malloc(sizeof(...));
    
    // 4. 调用 Python 回调，将 (device, size, d_mem, handle_ptr) 存入 Python 字典
    PyGILState_STATE gstate = PyGILState_Ensure();  // 获取 GIL！
    PyObject* arg_tuple = create_tuple_from_c_integers(device, alignedSize, d_mem, p_memHandle);
    PyObject_CallFunctionObjArgs(g_python_malloc_callback, arg_tuple, NULL);
    PyGILState_Release(gstate);
    
    // 5. 正式创建物理内存并映射到虚拟地址
    create_and_map(device, alignedSize, d_mem, p_memHandle);
    
    return (void*)d_mem;  // 返回给 PyTorch 作为 data_ptr()
}
```

**`my_free`（PyTorch GC 张量时调用）**：

```c
void my_free(void* ptr, ssize_t size, int device, CUstream stream) {
    // 1. 调用 Python 回调，从字典中查找 handle（返回元组）
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject* py_result = PyObject_CallFunctionObjArgs(g_python_free_callback, py_ptr, NULL);
    // 解包 (device, size, d_mem, p_memHandle)
    PyGILState_Release(gstate);
    
    // 2. 解除映射并释放物理内存
    unmap_and_release(device, size, d_mem, p_memHandle);
    
    // 3. 释放虚拟地址空间（free 时才真正释放虚拟地址）
    cuMemAddressFree(d_mem, size);
    free(p_memHandle);
}
```

> **单例约束的根因**：`g_python_malloc_callback` 和 `g_python_free_callback` 是 C 全局变量，只能存一个实例的回调。多个 `CuMemAllocator` 实例会互相覆盖，导致内存管理混乱。

**ROCm 的特殊处理**：ROCm 平台不支持单次大块 `cuMemCreate`，需要分成多个 chunk（默认 256MB，可通过 `VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE` 调整）。所以 ROCm 版本的 handle 是一个 handle 数组，Python 侧收到的是列表而非单整数。

### 5.2 Layer 1b：CuMemAllocator（`vllm/device_allocator/cumem.py`）

Python 层的内存池管理器，单例模式，319 行。

**核心数据结构**：

```python
@dataclasses.dataclass
class AllocationData:
    handle: HandleType          # (device, aligned_size, d_mem, p_memHandle)
    tag: str                    # "weights" / "kv_cache" / "default"
    cpu_backup_tensor: torch.Tensor | None = None  # CPU 备份（level 1 sleep 时填充）

class CuMemAllocator:
    instance: "CuMemAllocator | None" = None  # 单例

    def __init__(self):
        self.pointer_to_data: dict[int, AllocationData] = {}
        # key = GPU 虚拟地址（即 tensor.data_ptr()）
        # value = 该块内存的完整元数据

        self.current_tag: str = "default"
        # 当前 use_memory_pool 上下文的 tag

        self.allocator_and_pools: dict[str, Any] = {}
        # 保持对 allocator/pool 对象的强引用，防止 GC 回收
        # 这是修复 PyTorch 2.6 bug 的关键（见 PR #13456）

        # 必须保持强引用，否则 bound method 对象被 GC，C 层回调悬空
        # 见 PR #22724 的讨论
        self.python_malloc_callback = self._python_malloc_callback
        self.python_free_callback = self._python_free_callback
```

**`use_memory_pool` 上下文管理器**（让 PyTorch 使用自定义分配器）：

```python
@contextmanager
def use_memory_pool(self, tag: str | None = None):
    # 1. 处理 expandable_segments 不兼容问题
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    expandable_was_enabled = "expandable_segments:True" in conf
    if expandable_was_enabled:
        torch.cuda.memory._set_allocator_settings("expandable_segments:False")
        # 进入时临时禁用，退出时恢复
        # 原因：expandable_segments 与 MemPool 在 PyTorch 中存在 bug
        # 见 https://github.com/pytorch/pytorch/issues/147851

    old_tag = self.current_tag
    self.current_tag = tag  # 让 malloc callback 知道当前应打什么 tag

    try:
        # 2. 创建 CUDAPluggableAllocator（指向 cumem_allocator.so 中的 my_malloc/my_free）
        init_module(self.python_malloc_callback, self.python_free_callback)
        new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
            lib_name, "my_malloc", "my_free"
        )
        # 3. 创建 MemPool 并激活
        mem_pool = torch.cuda.memory.MemPool(new_alloc._allocator)
        with torch.cuda.memory.use_mem_pool(mem_pool):
            # 在此上下文中的所有 torch.empty/zeros/randn 等调用
            # 都会走 my_malloc，地址被记录到 pointer_to_data
            self.allocator_and_pools[tag] = (mem_pool, new_alloc)  # 强引用！
            yield

            # 4. 退出时处理"已分配后被 free"的内存（on-the-fly 量化场景）
            #    PyTorch 有 bug：pluggable allocator 下 empty_cache() 会崩溃
            #    所以手动找出 allocated_size==0 的块并 unmap
            allocations = mem_pool.snapshot()
            for alloc in allocations:
                if alloc["allocated_size"] == 0:
                    handle = self._python_free_callback(alloc["address"])
                    unmap_and_release(handle)
    finally:
        self.current_tag = old_tag
        if expandable_was_enabled:
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
```

**`sleep` 方法**：

```python
def sleep(self, offload_tags: tuple[str, ...] = ("default",)) -> None:
    total_bytes = 0
    backup_bytes = 0

    for ptr, data in self.pointer_to_data.items():
        handle = data.handle
        total_bytes += handle[1]

        if data.tag in offload_tags:
            # 1. 分配 CPU pinned memory（pin_memory 加速后续 cudaMemcpy）
            size_in_bytes = handle[1]
            cpu_backup = torch.empty(
                size_in_bytes, dtype=torch.uint8,
                device="cpu",
                pin_memory=is_pin_memory_available()
            )
            # 2. GPU → CPU 拷贝（同步）
            libcudart.cudaMemcpy(cpu_backup.data_ptr(), ptr, size_in_bytes)
            data.cpu_backup_tensor = cpu_backup
            backup_bytes += size_in_bytes

        # 3. 所有 tag 的内存都解除映射（无论是否备份）
        unmap_and_release(handle)  # → python_unmap_and_release(C) → cuMemUnmap + cuMemRelease

    # 4. 强制 Python GC 和 CUDA 缓存清理
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("sleep freed %.2f GiB, backed up %.2f GiB to CPU",
                total_bytes/1024**3, backup_bytes/1024**3)
```

**`wake_up` 方法**：

```python
def wake_up(self, tags: list[str] | None = None) -> None:
    for ptr, data in self.pointer_to_data.items():
        if tags is None or data.tag in tags:
            # 1. 重新分配物理内存并映射回原虚拟地址
            create_and_map(data.handle)
            # → python_create_and_map(C) → cuMemCreate + cuMemMap + cuMemSetAccess

            # 2. 如果有 CPU 备份，把数据搬回 GPU
            if data.cpu_backup_tensor is not None:
                size_in_bytes = data.cpu_backup_tensor.numel()
                libcudart.cudaMemcpy(
                    ptr,                              # GPU 虚拟地址（不变！）
                    data.cpu_backup_tensor.data_ptr(),
                    size_in_bytes
                )
                data.cpu_backup_tensor = None  # 释放 CPU 备份
```

> **`pointer_to_data` 字典的生命周期**：从初始化到进程退出，该字典**永远不清空**（除非张量被 GC）。sleep 和 wake_up 只改变物理内存的状态，不改变字典内容。这是设计的精髓。

### 5.3 Layer 2：Worker（`vllm/v1/worker/gpu_worker.py`）

Worker 是单 GPU 的全权代理，所有 GPU 操作都经过它。

**`_maybe_get_memory_pool_context`（核心辅助方法）**：

```python
def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
    """
    如果 enable_cumem_allocator=True，返回 cumem 内存池上下文；
    否则返回 nullcontext（什么都不做）。
    这个设计让调用方不需要判断 sleep mode 是否开启。
    """
    if not self.vllm_config.model_config.enable_cumem_allocator:
        return nullcontext()

    allocator = CuMemAllocator.get_instance()
    if tag == "weights":
        # 保证单例约束：首次进入 weights 池时，池必须为空
        assert allocator.get_current_usage() == 0, \
            "CuMem allocator can only be used for one instance per process."
    return allocator.use_memory_pool(tag=tag)
```

**`load_model`（权重进入 "weights" 池）**：

```python
def load_model(self, *, load_dummy_weights: bool = False) -> None:
    with (
        self._maybe_get_memory_pool_context(tag="weights"),  # ← 权重标记为 "weights"
        set_current_vllm_config(self.vllm_config),
        self._scoped_allocator_max_split(max_split_size_mb=20),
        # max_split_size_mb=20 减少分配碎片化，
        # 代价是更多次 cuMemCreate 调用（对模型加载影响可忽略）
    ):
        self.model_runner.load_model(load_dummy_weights=load_dummy_weights)
```

**`initialize_from_config`（KV Cache 进入 "kv_cache" 池）**：

```python
def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
    self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks

    with self._maybe_get_memory_pool_context(tag="kv_cache"):  # ← KV Cache 标记
        self.model_runner.initialize_kv_cache(kv_cache_config)

    # ⚠️ 注意：KV zero metadata（用于 KV 内存归零的辅助结构）
    # 故意在 pool 外分配，因为它不参与 sleep/wake 循环
    if kv_cache_config.needs_kv_cache_zeroing:
        self.model_runner._init_kv_zero_meta()
```

**`sleep` 方法**：

```python
def sleep(self, level: int = 1) -> None:
    free_before = torch.cuda.mem_get_info()[0]

    # Level 2：在丢弃权重前，把 buffers clone 到 CPU
    # buffers 包含 RoPE cos/sin、expert_map 等不在 cumem 池中的常量
    if level == 2:
        model = self.model_runner.model
        self._sleep_saved_buffers = {
            name: buffer.cpu().clone()
            for name, buffer in model.named_buffers()
        }

    allocator = CuMemAllocator.get_instance()
    # level 1: offload "weights"（备份到 CPU），丢弃 "kv_cache"
    # level 2: offload_tags=()，所有内存直接丢弃
    allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())

    freed_bytes = torch.cuda.mem_get_info()[0] - free_before
    logger.info("Sleep freed %s GiB", format_gib(freed_bytes))
```

**`wake_up` 方法**：

```python
def wake_up(self, tags: list[str] | None = None) -> None:
    allocator = CuMemAllocator.get_instance()
    allocator.wake_up(tags)

    # 恢复 level 2 sleep 前保存的 buffers
    if len(self._sleep_saved_buffers):
        model = self.model_runner.model
        for name, buffer in model.named_buffers():
            if name in self._sleep_saved_buffers:
                buffer.data.copy_(self._sleep_saved_buffers[name].data)
        self._sleep_saved_buffers = {}

    # KV Cache 唤醒后需要修复 FP8 scale（见下节）
    if tags is None or "kv_cache" in tags:
        self.model_runner.post_kv_cache_wake_up()
```

**`post_kv_cache_wake_up`（`gpu_model_runner.py:927`）**：

Wake up 后，新映射的物理内存内容是**未定义的**（操作系统给的"脏"内存）。对于 FP8 KV Cache 会引发灾难性后果：

```python
def post_kv_cache_wake_up(self) -> None:
    self.init_fp8_kv_scales()

def init_fp8_kv_scales(self) -> None:
    """
    修复 FP8 KV Cache wake_up 后的两个问题：
    1. KV Cache tensor 内容是随机垃圾 → zero_() 清零
    2. Attention 层的 _k_scale/_v_scale 默认值是 0.0 →
       导致所有 KV 计算结果为 0 → 输出乱码
       → 重置为 1.0
    """
    if not is_quantized_kv_cache(self.cache_config.cache_dtype):
        return  # 非 FP8 量化不需要处理

    # 清零 KV Cache
    for cache_tensor in self.kv_caches:
        cache_tensor.zero_()

    # 重置 attention 层的 FP8 scale
    for name, module in self.compilation_config.static_forward_context.items():
        if isinstance(module, (Attention, MLAAttention)):
            for attr in ("_k_scale", "k_scale"):
                if hasattr(module, attr) and isinstance(getattr(module, attr), torch.Tensor):
                    getattr(module, attr).fill_(1.0)
            for attr in ("_v_scale", "v_scale"):
                if hasattr(module, attr) and isinstance(getattr(module, attr), torch.Tensor):
                    getattr(module, attr).fill_(1.0)
```

### 5.4 Layer 3：Executor（`vllm/v1/executor/abstract.py`）

Executor 是 Worker 的管理者，sleep/wake_up 通过 `collective_rpc` 广播到所有 Worker 进程。

```python
class Executor(ABC):
    def __init__(self, ...):
        self.is_sleeping: bool = False
        self.sleeping_tags: set[str] = set()
        # 追踪当前哪些 tags 处于 sleeping 状态

    def sleep(self, level: int = 1):
        if self.is_sleeping:
            logger.warning("Executor is already sleeping.")
            return
        t0 = time.perf_counter()
        # 广播 sleep 命令到所有 Worker（TP 场景下所有 GPU 同步执行）
        self.collective_rpc("sleep", kwargs=dict(level=level))
        logger.info("It took %.6f seconds to fall asleep.", time.perf_counter() - t0)
        self.sleeping_tags = {"weights", "kv_cache"}
        self.is_sleeping = True

    def wake_up(self, tags: list[str] | None = None):
        if not self.is_sleeping:
            logger.warning("Executor is not sleeping.")
            return
        t0 = time.perf_counter()
        self.collective_rpc("wake_up", kwargs=dict(tags=tags))
        logger.info("It took %.6f seconds to wake up tags %s.",
                    time.perf_counter() - t0,
                    tags if tags is not None else self.sleeping_tags)
        if tags:
            for tag in tags:
                self.sleeping_tags.discard(tag)
        else:
            self.sleeping_tags.clear()
        if not self.sleeping_tags:
            self.is_sleeping = False
        # 注意：分步唤醒时（如先 wake_up(["weights"]) 后 wake_up(["kv_cache"])），
        # is_sleeping 直到所有 tags 都唤醒才变为 False
```

`collective_rpc` 在 `UniProcExecutor`（单进程）中直接调用 Worker 方法，在 `MultiprocExecutor`（多进程 TP）中通过 IPC 管道向每个 Worker 进程发送 RPC 消息，确保**所有 GPU 同步执行** sleep/wake_up。

### 5.5 Layer 4：EngineCore（`vllm/v1/engine/core.py`）

EngineCore 负责协调调度器与执行器，是 sleep mode 最复杂的一层。

**调度器暂停状态机**：

```python
class PauseState(enum.IntEnum):
    UNPAUSED = 0     # 正常运行
    PAUSED_NEW = 1   # 不接受新请求，in-flight 请求继续处理（wait/abort 模式中间态）
    PAUSED_ALL = 2   # 所有请求都暂停（keep 模式）
```

**`pause_scheduler`（sleep 前调用）**：

```python
def pause_scheduler(
    self, mode: PauseMode = "abort", clear_cache: bool = True
) -> Future | None:
    if mode == "abort":
        # 立即中止所有 in-flight 请求
        self.scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)

    pause_state = PauseState.PAUSED_ALL if mode == "keep" else PauseState.PAUSED_NEW
    self.scheduler.set_pause_state(pause_state)

    if clear_cache:
        # 清空 prefix cache！
        # 原因：prefix cache 中存储了指向 KV Cache 块的引用
        # sleep 后 KV Cache 已被释放，这些引用指向无效内存
        self._reset_caches()  # reset_prefix_cache + reset_mm_cache + reset_encoder_cache

    return None  # abort/keep 模式同步完成，返回 None
    # wait 模式返回 Future（异步等待 in-flight 请求完成）
```

**`sleep` 方法**：

```python
def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None | Future:
    # 1. 先暂停调度器
    clear_prefix_cache = (level >= 1)  # level 0 不需要清 cache（显存没变）
    pause_future = self.pause_scheduler(mode=mode, clear_cache=clear_prefix_cache)

    if level < 1:
        return pause_future  # level 0 只暂停调度，不动 GPU

    # 2. 处理 wait 模式（需要异步等待）
    model_executor = self.model_executor
    if pause_future is None:
        # abort/keep 模式：同步调用 executor.sleep()
        model_executor.sleep(level)
        return None
    else:
        # wait 模式：等 in-flight 请求完成后再 sleep
        future = Future()
        def pause_complete(f: Future):
            try:
                f.result()
                future.set_result(model_executor.sleep(level))
            except Exception as e:
                future.set_exception(e)
        pause_future.add_done_callback(pause_complete)
        return future
```

**`wake_up` 方法**：

```python
def wake_up(self, tags: list[str] | None = None):
    # "scheduling" 是虚拟 tag，表示 level 0 的唤醒（只恢复调度，不动 GPU 内存）
    if tags is not None and "scheduling" in tags:
        tags = [t for t in tags if t != "scheduling"]

    # 1. 先恢复 GPU 内存（如果有 GPU 操作的话）
    if tags is None or tags:
        self.model_executor.wake_up(tags)

    # 2. 恢复调度器（所有 level 都需要）
    self.resume_scheduler()  # → scheduler.set_pause_state(PauseState.UNPAUSED)

def is_sleeping(self) -> bool:
    """引擎在任何层面 sleeping 都返回 True"""
    return self.is_scheduler_paused() or self.model_executor.is_sleeping
```

### 5.6 Layer 5：Entrypoint

**离线推理 API（`vllm/entrypoints/llm.py`）**：

```python
class LLM:
    def sleep(self, level: int = 1, mode: PauseMode = "abort"):
        """
        level 0: 仅暂停调度，不动 GPU 内存
        level 1: offload 权重到 CPU，丢弃 KV Cache
        level 2: 丢弃所有 GPU 内存（权重+KV Cache）
        mode: "abort"（默认）/"wait"/"keep"
        """
        self.llm_engine.sleep(level=level, mode=mode)

    def wake_up(self, tags: list[str] | None = None):
        """
        tags=None: 恢复所有内存
        tags=["weights"]: 只恢复权重
        tags=["kv_cache"]: 只恢复 KV Cache
        tags=["scheduling"]: 从 level 0 唤醒
        """
        self.llm_engine.wake_up(tags)
```

**在线服务 API（`vllm/entrypoints/serve/sleep/api_router.py`）**：

```python
router = APIRouter()

@router.post("/sleep")
async def sleep(raw_request: Request):
    level = raw_request.query_params.get("level", "1")
    mode = raw_request.query_params.get("mode", "abort")
    await engine_client(raw_request).sleep(int(level), mode)
    return Response(status_code=200)

@router.post("/wake_up")
async def wake_up(raw_request: Request):
    tags = raw_request.query_params.getlist("tags") or None
    await engine_client(raw_request).wake_up(tags)
    return Response(status_code=200)

@router.get("/is_sleeping")
async def is_sleeping(raw_request: Request):
    result = await engine_client(raw_request).is_sleeping()
    return JSONResponse(content={"is_sleeping": result})

def attach_router(app: FastAPI):
    if not envs.VLLM_SERVER_DEV_MODE:
        return  # 仅在 VLLM_SERVER_DEV_MODE=1 时挂载，不对外暴露
    app.include_router(router)
```

### 5.7 配置层（`vllm/config/model.py` + `vllm/engine/arg_utils.py`）

```python
class ModelConfig:
    # 用户开关：--enable-sleep-mode
    enable_sleep_mode: bool = False
    # 底层开关（可独立开启 cumem 分配器，不一定要 sleep mode）
    enable_cumem_allocator: bool = False

    def __post_init__(self):
        if self.enable_sleep_mode:
            # 前提检查 1：平台必须是 CUDA 或 ROCm
            if not current_platform.is_sleep_mode_available():
                raise ValueError("Sleep mode is not supported on current platform.")
            # 自动连锁：sleep mode 依赖 cumem allocator
            if not self.enable_cumem_allocator:
                logger.info_once("Enabling cumem allocator because sleep mode requires it.")
                self.enable_cumem_allocator = True

        if self.enable_cumem_allocator:
            # 前提检查 2：C 扩展必须编译成功（CPU-only 环境无法编译）
            if not is_cumem_allocator_available():
                raise ValueError("cumem allocator is not supported on current platform.")
```

```python
# vllm/platforms/interface.py
def is_sleep_mode_available(self) -> bool:
    # CUDA 和 ROCm 都支持
    # ROCm 目前只有 mi3xx 真正支持，但无法在配置阶段静态检测
    return self._enum in (PlatformEnum.CUDA, PlatformEnum.ROCM)
```

---

## 6. PR 演进历史：设计如何从简单走向健壮

下面按时间顺序梳理 Sleep Mode 的演进，每个 PR 都解决了一个真实遇到的问题。

### 阶段一：核心功能（2025 年 1 月）

**PR #11743** `[Core] Support fully transparent sleep mode`（作者：youkaichao）

- 新增 `csrc/cumem_allocator.cpp`（310 行 C 代码）
- 新增 `vllm/device_allocator/cumem.py`（254 行）
- 修改 `vllm/config.py`：增加 `enable_sleep_mode` 字段
- 修改 `vllm/v1/worker/gpu_worker.py`：`load_model` 和 `initialize_cache` 加上 cumem pool 上下文
- 修改 `vllm/entrypoints/llm.py`：暴露 `sleep()`/`wake_up()` API

这一版只有 level 1 sleep（offload weights，discard KV cache），没有 tags 参数。

**PR #12987** `[core] add sleep and wake up endpoint and v1 support`

- 增加 HTTP 端点 `/sleep`、`/wake_up`、`/is_sleeping`
- 增加 V1 engine 支持

### 阶段二：稳定性修复（2025 年 2-4 月）

**PR #13456** `[core] fix sleep mode in pytorch 2.6`（作者：youkaichao）

PyTorch 2.6 引入了新的 GC 行为，导致 `CUDAPluggableAllocator` 和 `MemPool` 对象在 `use_memory_pool` 上下文退出后被 GC，从而导致后续 malloc/free 回调悬空崩溃。

修复：在 `CuMemAllocator` 实例上保持对 `(mem_pool, allocator)` 的强引用：
```python
self.allocator_and_pools[tag] = data  # 防止 GC
```

**PR #15500** `[core] Add tags parameter to wake_up()` （作者：erictang000）

增加 `wake_up(tags=["weights"])` 支持，实现分步唤醒。这是 RLHF 场景的核心优化。

**PR #16889** `Restore buffers when wake up from level 2 sleep`（作者：Han Zhang）

发现 level 2 sleep 后，模型的 `named_buffers()` 内容丢失（因为 buffers 不在 cumem pool 中，wake_up 后对应内存是随机内容）。

修复：sleep 前 clone buffers 到 CPU，wake_up 后 copy_ 回来：
```python
# sleep 时
if level == 2:
    self._sleep_saved_buffers = {
        name: buffer.cpu().clone()
        for name, buffer in model.named_buffers()
    }

# wake_up 时
for name, buffer in model.named_buffers():
    if name in self._sleep_saved_buffers:
        buffer.data.copy_(self._sleep_saved_buffers[name].data)
```

**PR #12695** `[Core][AMD] Migrate fully transparent sleep mode to ROCm platform`（作者：hollowman）

ROCm 的 `cuMemCreate` 不支持单次大块分配，需要分成多个 256MB chunk。涉及 `csrc/cumem_allocator.cpp` 新增 400 行 ROCm 适配代码，以及新增 `csrc/cumem_allocator_compat.h` 头文件统一 CUDA/ROCm API 差异。

### 阶段三：功能完善（2025 年下半年）

**PR #24731** `[sleep mode] save memory for on-the-fly quantization`（作者：youkaichao）

On-the-fly 量化（如 FP8）场景：模型先以高精度加载，然后在 `use_memory_pool` 上下文内原地量化。量化过程中产生的临时 tensor 被分配到 pool 里，量化完成后被 free，但 PyTorch 的 pluggable allocator 不支持 `empty_cache()`，导致这些"已 free 但未 unmap"的内存一直占着。

修复：在 `use_memory_pool` 退出时，检查 `mem_pool.snapshot()` 中 `allocated_size==0` 的块并手动 unmap。

**PR #28783** `[Bugfix][sleepmode][fp8 kv cache]: Fix FP8 KV Cache + sleep(level=2) gibberish output`

FP8 KV Cache + level 2 sleep 组合导致输出乱码。原因：wake_up 后 KV cache tensor 的内存是随机内容，且 attention 层的 `_k_scale`/`_v_scale` 默认为 0.0（因为 cumem 分配的内存未初始化），乘以 0 后所有 KV 值都是 0。

修复：增加 `post_kv_cache_wake_up()` → `init_fp8_kv_scales()`，wake_up 后清零 KV cache 并将 scale 重置为 1.0。

### 阶段四：架构重构（2026 年）

**PR #33195** `[Core] Add sleep level 0 mode with enqueue/wait pattern`（作者：jaewonlee-fb）

为 RLHF 中"仅暂停调度，不释放内存"的需求增加 level 0，同时引入 `PauseMode` 的 `"wait"` 模式（等 in-flight 请求处理完再 sleep）。

**PR #34528** `[Core] Cleanup engine pause/sleep logic`（作者：nickhill）

大规模重构 Engine 层的 sleep/pause 逻辑，将散布在各处的状态管理统一到 `PauseState` 枚举和 `pause_scheduler`/`resume_scheduler` 方法。这个 PR 同时解决了异步引擎（DP 多副本场景）的 sleep 状态同步问题。

**PR #40812** `Auto-disable expandable_segments around cumem memory pool`（作者：youkaichao）

发现 `expandable_segments:True`（PyTorch 的内存优化选项）与 `use_mem_pool` 在 PyTorch 中存在 bug (#147851)。修复：在 `use_memory_pool` 上下文中自动临时禁用 expandable_segments。

---

## 7. 从零实现：分步骤操作手册

以下是在一个**没有 sleep mode 的旧版 vLLM**（假设版本类似 2024 年底的代码）上实现 sleep mode 的完整步骤。每一步都可以单独测试，代码完全可以直接使用。

---

### Step 0：环境准备与理解

#### 0.1 安装依赖

```bash
# 安装 uv（Python 包管理器）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建 Python 3.12 虚拟环境
uv venv --python 3.12
source .venv/bin/activate   # Linux/Mac
# Windows: .venv\Scripts\activate

# 安装 lint 工具并激活 pre-commit 钩子
uv pip install -r requirements/lint.txt
pre-commit install

# 安装 vLLM（仅 Python 修改，使用预编译的 C 扩展）
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

#### 0.2 确认旧版 vLLM 的现状

在开始之前，先理解旧版代码中缺少哪些东西：

```bash
# 旧版中这些文件应该不存在
ls vllm/device_allocator/cumem.py    # 不存在
ls csrc/cumem_allocator.cpp          # 不存在

# 旧版 Worker 中应该没有 sleep/wake_up 方法
grep -n "def sleep\|def wake_up" vllm/v1/worker/gpu_worker.py  # 无输出
```

**旧版 Worker 接口（目标：向其中添加 sleep/wake_up）**：
```python
# 旧版 vllm/v1/worker/gpu_worker.py 的 Worker 类
class Worker:
    def load_model(self) -> None: ...          # 加载模型权重到 GPU
    def initialize_cache(self, kv_cache_config) -> None: ...  # 分配 KV Cache
    def execute_model(self, scheduler_output) -> None: ...    # 执行推理
    # 缺少：sleep() / wake_up()
    # 缺少：_maybe_get_memory_pool_context()
    # 缺少：_sleep_saved_buffers 字段
```

---

### Step 1：创建 C 扩展（`csrc/cumem_allocator.cpp`）

这是整个实现的基础。C 扩展实现了两件事：
1. 作为 PyTorch 的**自定义 CUDA 内存分配器**（替换 `cudaMalloc`/`cudaFree`）
2. 暴露 `python_unmap_and_release` 和 `python_create_and_map` 供 Python 直接调用

#### 1.1 创建文件 `csrc/cumem_allocator.cpp`

以下是完整代码（CUDA 版本，去掉 ROCm 条件编译以便理解）：

```cpp
// csrc/cumem_allocator.cpp
// 基于 CUDA VMM API 的自定义 PyTorch 内存分配器
// 用途：实现 vLLM sleep mode（sleep 时释放物理显存，地址不变）
#include <iostream>
#include <sys/types.h>

// 使用 CUDA Driver API（注意：不是 Runtime API）
#include <cuda.h>

extern "C" {
#define PY_SSIZE_T_CLEAN
#include <Python.h>

// ===================================================================
// 错误处理宏
// ===================================================================
char error_msg[10240];
CUresult no_error = CUresult(0);
CUresult error_code = no_error;

#define CUDA_CHECK(condition)                                            \
  do {                                                                   \
    CUresult error = condition;                                          \
    if (error != 0) {                                                    \
      error_code = error;                                                \
      char* error_string;                                                \
      cuGetErrorString(error, (const char**)&error_string);              \
      snprintf(error_msg, sizeof(error_msg), "CUDA Error: %s at %s:%d", \
               error_string, __FILE__, __LINE__);                        \
      std::cerr << error_msg << std::endl;                               \
    }                                                                    \
  } while (0)

// ===================================================================
// 核心：两个全局回调变量（这是"单例约束"的根源）
// 借用引用（borrowed reference），不负责 DECREF
// ===================================================================
static PyObject* g_python_malloc_callback = nullptr;
static PyObject* g_python_free_callback = nullptr;

// ===================================================================
// 辅助：确保 CUDA context 存在
// ===================================================================
void ensure_context(unsigned long long device) {
  CUcontext pctx;
  CUDA_CHECK(cuCtxGetCurrent(&pctx));
  if (!pctx) {
    CUDA_CHECK(cuDevicePrimaryCtxRetain(&pctx, device));
    CUDA_CHECK(cuCtxSetCurrent(pctx));
  }
}

// ===================================================================
// 辅助：将 4 个 unsigned long long 打包为 Python tuple
// ===================================================================
PyObject* create_tuple_from_c_integers(unsigned long long a,
                                       unsigned long long b,
                                       unsigned long long c,
                                       unsigned long long d) {
  PyObject* tuple = PyTuple_New(4);
  if (!tuple) return NULL;
  PyTuple_SetItem(tuple, 0, PyLong_FromUnsignedLongLong(a));
  PyTuple_SetItem(tuple, 1, PyLong_FromUnsignedLongLong(b));
  PyTuple_SetItem(tuple, 2, PyLong_FromUnsignedLongLong(c));
  PyTuple_SetItem(tuple, 3, PyLong_FromUnsignedLongLong(d));
  return tuple;
}

// ===================================================================
// 核心：create_and_map
// wake_up 时调用：重新分配物理内存，映射到已保留的虚拟地址
// ===================================================================
void create_and_map(unsigned long long device, ssize_t size,
                    CUdeviceptr d_mem,
                    CUmemGenericAllocationHandle* p_memHandle) {
  ensure_context(device);

  // 内存属性：设备端 pinned 内存
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;

  // 可选：支持 GPUDirect RDMA
  int flag = 0;
  if (cuDeviceGetAttribute(&flag,
      CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
      device) == CUDA_SUCCESS && flag) {
    prop.allocFlags.gpuDirectRDMACapable = 1;
  }

  // Step 1：分配物理内存，获得 handle
  CUDA_CHECK(cuMemCreate(p_memHandle, size, &prop, 0));
  if (error_code != 0) return;

  // Step 2：映射物理内存到虚拟地址（d_mem 地址不变！）
  CUDA_CHECK(cuMemMap(d_mem, size, 0, *p_memHandle, 0));
  if (error_code != 0) return;

  // Step 3：设置读写权限
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = device;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUDA_CHECK(cuMemSetAccess(d_mem, size, &accessDesc, 1));
}

// ===================================================================
// 核心：unmap_and_release
// sleep 时调用：解除映射，释放物理内存。虚拟地址不释放！
// ===================================================================
void unmap_and_release(unsigned long long device, ssize_t size,
                       CUdeviceptr d_mem,
                       CUmemGenericAllocationHandle* p_memHandle) {
  ensure_context(device);

  // Step 1：解除虚拟地址到物理内存的映射（地址仍保留，但访问会崩溃）
  CUDA_CHECK(cuMemUnmap(d_mem, size));
  if (error_code != 0) return;

  // Step 2：释放物理内存（显存还给系统）
  CUDA_CHECK(cuMemRelease(*p_memHandle));
  // 注意：不调用 cuMemAddressFree！虚拟地址保留供下次 wake_up 使用
}

// ===================================================================
// PyTorch 分配器钩子：my_malloc
// 每当 PyTorch 在 cumem pool 中分配 tensor 时被调用
// 签名必须与 PyTorch CUDAPluggableAllocator 接口匹配
// ===================================================================
void* my_malloc(ssize_t size, int device, CUstream stream) {
  ensure_context(device);

  // 获取分配粒度（通常是 2MB）
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;

  size_t granularity;
  CUDA_CHECK(cuMemGetAllocationGranularity(&granularity, &prop,
                                           CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  if (error_code != 0) return nullptr;

  // 对齐到粒度
  size_t alignedSize = ((size + granularity - 1) / granularity) * granularity;

  // Step 1：只预留虚拟地址（不占物理显存）
  CUdeviceptr d_mem;
  CUDA_CHECK(cuMemAddressReserve(&d_mem, alignedSize, 0, 0, 0));
  if (error_code != 0) return nullptr;

  // Step 2：在 CPU 堆上分配 handle 结构体
  CUmemGenericAllocationHandle* p_memHandle =
      (CUmemGenericAllocationHandle*)malloc(
          sizeof(CUmemGenericAllocationHandle));

  if (!g_python_malloc_callback) {
    std::cerr << "ERROR: g_python_malloc_callback not set.\n";
    return nullptr;
  }

  // Step 3：获取 GIL，调用 Python 回调记录 (device, size, d_mem, handle)
  // 这让 Python 的 pointer_to_data 字典记录下这块内存的元数据
  PyGILState_STATE gstate = PyGILState_Ensure();
  PyObject* arg_tuple = create_tuple_from_c_integers(
      (unsigned long long)device, (unsigned long long)alignedSize,
      (unsigned long long)d_mem, (unsigned long long)p_memHandle);
  PyObject* py_result =
      PyObject_CallFunctionObjArgs(g_python_malloc_callback, arg_tuple, NULL);
  Py_DECREF(arg_tuple);
  if (!py_result) {
    PyErr_Print();
    PyGILState_Release(gstate);
    return nullptr;
  }
  PyGILState_Release(gstate);

  // Step 4：正式分配物理内存并映射
  create_and_map(device, alignedSize, d_mem, p_memHandle);
  if (error_code != 0) {
    CUDA_CHECK(cuMemAddressFree(d_mem, alignedSize));
    free(p_memHandle);
    return nullptr;
  }

  // 返回虚拟地址给 PyTorch，成为 tensor.data_ptr()
  return (void*)d_mem;
}

// ===================================================================
// PyTorch 分配器钩子：my_free
// 每当 PyTorch 的 GC 释放 cumem pool 中的 tensor 时被调用
// ===================================================================
void my_free(void* ptr, ssize_t size, int device, CUstream stream) {
  if (!g_python_free_callback) {
    std::cerr << "ERROR: g_python_free_callback not set.\n";
    return;
  }

  // Step 1：获取 GIL，调用 Python 回调查找 handle（并从字典中删除）
  PyGILState_STATE gstate = PyGILState_Ensure();
  PyObject* py_ptr =
      PyLong_FromUnsignedLongLong(reinterpret_cast<unsigned long long>(ptr));
  PyObject* py_result =
      PyObject_CallFunctionObjArgs(g_python_free_callback, py_ptr, NULL);

  if (!py_result || !PyTuple_Check(py_result) || PyTuple_Size(py_result) != 4) {
    PyErr_SetString(PyExc_TypeError, "Expected a tuple of size 4");
    Py_XDECREF(py_result);
    Py_XDECREF(py_ptr);
    return;
  }

  // Step 2：解包 Python tuple 为 C 变量
  unsigned long long recv_device, recv_size, recv_d_mem, recv_p_memHandle;
  if (!PyArg_ParseTuple(py_result, "KKKK",
                        &recv_device, &recv_size,
                        &recv_d_mem, &recv_p_memHandle)) {
    Py_XDECREF(py_result);
    Py_XDECREF(py_ptr);
    return;
  }

  Py_DECREF(py_ptr);
  Py_DECREF(py_result);
  PyGILState_Release(gstate);

  // Step 3：解除映射并释放物理内存
  CUdeviceptr d_mem = (CUdeviceptr)recv_d_mem;
  CUmemGenericAllocationHandle* p_memHandle =
      (CUmemGenericAllocationHandle*)recv_p_memHandle;
  unmap_and_release(device, size, d_mem, p_memHandle);

  // Step 4：释放虚拟地址空间（free 时才真正释放虚拟地址）
  CUDA_CHECK(cuMemAddressFree(d_mem, size));
  free(p_memHandle);
}

// ===================================================================
// Python 扩展函数：py_init_module
// 注册 Python 的 malloc/free 回调到全局变量
// ===================================================================
static PyObject* py_init_module(PyObject* self, PyObject* args) {
  PyObject* malloc_callback = nullptr;
  PyObject* free_callback = nullptr;

  if (!PyArg_ParseTuple(args, "OO", &malloc_callback, &free_callback)) {
    return nullptr;
  }
  if (!PyCallable_Check(malloc_callback) || !PyCallable_Check(free_callback)) {
    PyErr_SetString(PyExc_TypeError, "Both arguments must be callables");
    return nullptr;
  }

  // 借用引用（调用方必须保持回调对象存活）
  g_python_malloc_callback = malloc_callback;
  g_python_free_callback = free_callback;

  Py_RETURN_NONE;
}

// ===================================================================
// Python 扩展函数：python_unmap_and_release
// sleep 时 Python 层直接调用（绕过 PyTorch GC）
// 接收参数：(device, size, d_mem, p_memHandle) 四元组
// ===================================================================
static PyObject* python_unmap_and_release(PyObject* self, PyObject* args) {
  if (!args || !PyTuple_Check(args) || PyTuple_Size(args) != 4) {
    PyErr_SetString(PyExc_TypeError, "Expected a tuple of size 4");
    return nullptr;
  }

  unsigned long long recv_device, recv_size, recv_d_mem, recv_p_memHandle;
  if (!PyArg_ParseTuple(args, "KKKK",
                        &recv_device, &recv_size,
                        &recv_d_mem, &recv_p_memHandle)) {
    return nullptr;
  }

  CUdeviceptr d_mem_ptr = (CUdeviceptr)recv_d_mem;
  CUmemGenericAllocationHandle* p_memHandle =
      (CUmemGenericAllocationHandle*)recv_p_memHandle;
  unmap_and_release(recv_device, recv_size, d_mem_ptr, p_memHandle);

  if (error_code != 0) {
    error_code = no_error;
    PyErr_SetString(PyExc_RuntimeError, error_msg);
    return nullptr;
  }
  Py_RETURN_NONE;
}

// ===================================================================
// Python 扩展函数：python_create_and_map
// wake_up 时 Python 层直接调用
// 接收参数：(device, size, d_mem, p_memHandle) 四元组
// ===================================================================
static PyObject* python_create_and_map(PyObject* self, PyObject* args) {
  if (!args || !PyTuple_Check(args) || PyTuple_Size(args) != 4) {
    PyErr_SetString(PyExc_TypeError, "Expected a tuple of size 4");
    return nullptr;
  }

  unsigned long long recv_device, recv_size, recv_d_mem, recv_p_memHandle;
  if (!PyArg_ParseTuple(args, "KKKK",
                        &recv_device, &recv_size,
                        &recv_d_mem, &recv_p_memHandle)) {
    return nullptr;
  }

  CUdeviceptr d_mem_ptr = (CUdeviceptr)recv_d_mem;
  CUmemGenericAllocationHandle* p_memHandle =
      (CUmemGenericAllocationHandle*)recv_p_memHandle;
  create_and_map(recv_device, recv_size, d_mem_ptr, p_memHandle);

  if (error_code != 0) {
    error_code = no_error;
    PyErr_SetString(PyExc_RuntimeError, error_msg);
    return nullptr;
  }
  Py_RETURN_NONE;
}

// ===================================================================
// Python 模块注册
// ===================================================================
static PyMethodDef module_methods[] = {
    {"init_module", (PyCFunction)py_init_module, METH_VARARGS,
     "Initialize module with python_malloc and python_free callables."},
    {"python_create_and_map", (PyCFunction)python_create_and_map, METH_VARARGS,
     "Create and map memory on the device."},
    {"python_unmap_and_release", (PyCFunction)python_unmap_and_release,
     METH_VARARGS, "Unmap and release memory on the device."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cumem_allocator_module = {
    PyModuleDef_HEAD_INIT, "cumem_allocator",
    "cumem-based allocator for CUDAPluggableAllocator", -1, module_methods};

PyMODINIT_FUNC PyInit_cumem_allocator(void) {
  PyObject* module = PyModule_Create(&cumem_allocator_module);
  if (!module) return NULL;
  return module;
}

}  // extern "C"
```

> **ROCm 注意**：真实代码（`csrc/cumem_allocator.cpp`）中还有 ROCm 的条件编译分支（通过 `#ifndef USE_ROCM`/`#else`/`#endif` 区分），ROCm 需要把大块内存拆成若干 256MB 的 chunk 分别调用 `cuMemCreate`。CUDA 版本无需此处理，上面的代码已是完整 CUDA 实现。

---

### Step 2：修改 `CMakeLists.txt`，注册 C 扩展

打开 vLLM 根目录的 `CMakeLists.txt`，在 `# Define other extension targets` 章节末尾（或在其他扩展目标定义之后）添加以下内容：

```cmake
# 在 CMakeLists.txt 中添加（参考位置：文件末尾，其他 extension target 定义之后）

#
# cumem_allocator extension（vLLM sleep mode 所需的 CUDA VMM 分配器）
#
set(VLLM_CUMEM_EXT_SRC
  "csrc/cumem_allocator.cpp")

# 为各 CUDA 架构生成代码
set_gencode_flags_for_srcs(
  SRCS "${VLLM_CUMEM_EXT_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}")

# 只有 CUDA 或 ROCm 平台才编译此扩展
if(VLLM_GPU_LANG STREQUAL "CUDA" OR VLLM_GPU_LANG STREQUAL "HIP")
  message(STATUS "Enabling cumem allocator extension.")

  if(VLLM_GPU_LANG STREQUAL "CUDA")
    # CUDA：链接 CUDA Driver 库
    list(APPEND CUMEM_LIBS CUDA::cuda_driver)
  else()
    # ROCm：链接 amdhip64 库
    find_library(AMDHIP64_LIB
      NAMES amdhip64 libamdhip64.so
      PATHS ${ROCM_PATH}/lib
      NO_DEFAULT_PATH)
    if(AMDHIP64_LIB)
      list(APPEND CUMEM_LIBS ${AMDHIP64_LIB})
    else()
      list(APPEND CUMEM_LIBS amdhip64)
    endif()
  endif()

  # 定义扩展目标（vLLM 自定义宏，输出到 vllm/ 目录）
  define_extension_target(
    cumem_allocator         # 扩展名称
    DESTINATION vllm        # 输出目录（vllm/cumem_allocator.so）
    LANGUAGE CXX            # 用 C++ 编译器
    SOURCES ${VLLM_CUMEM_EXT_SRC}
    LIBRARIES ${CUMEM_LIBS}
    USE_SABI 3.8            # 使用稳定 ABI（兼容 Python 3.8+）
    WITH_SOABI)             # 文件名包含 Python ABI 标记
endif()
```

**关键点说明**：
- `define_extension_target` 是 vLLM 自己封装的 CMake 宏（定义在 `cmake/utils.cmake`），等价于 `pybind11_add_module` + 安装规则
- `USE_SABI 3.8` 使扩展 `.so` 文件兼容 Python 3.8 及以上版本，不需要为每个 Python 版本重新编译
- 输出文件最终放在 `vllm/cumem_allocator.cpython-3xx-x86_64-linux-gnu.so`（文件名包含 Python 版本标记）

---

### Step 3：编译构建 C 扩展

#### 验证编译成功

```python
# test_c_extension.py
from vllm.cumem_allocator import init_module, python_create_and_map, python_unmap_and_release
print("✓ C extension loaded successfully")

# 验证函数签名
import inspect
print(f"  init_module: {init_module.__doc__}")
print(f"  python_create_and_map: {python_create_and_map.__doc__}")
print(f"  python_unmap_and_release: {python_unmap_and_release.__doc__}")
```

---

### Step 4：创建 Python 内存管理层（`vllm/device_allocator/cumem.py`）

先创建 `__init__.py`：

```bash
mkdir -p vllm/device_allocator
touch vllm/device_allocator/__init__.py
```

然后创建 `vllm/device_allocator/cumem.py`（完整内容）：

```python
# vllm/device_allocator/cumem.py
# CuMemAllocator：管理 cumem 内存池的 Python 层单例
import dataclasses
import gc
import os
from contextlib import contextmanager
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.system_utils import find_loaded_library

logger = init_logger(__name__)

# -----------------------------------------------------------------------
# 尝试导入 C 扩展（CPU-only 环境下不可用）
# -----------------------------------------------------------------------
cumem_available = False
libcudart: Any = None
try:
    from vllm.cumem_allocator import (
        init_module,
        python_create_and_map,
        python_unmap_and_release,
    )
    from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

    # 找到 cumem_allocator.so 的路径（CUDAPluggableAllocator 需要文件路径）
    lib_name = find_loaded_library("cumem_allocator")
    # libcudart 用于直接调用 cudaMemcpy（Python 到 C 的高速数据传输）
    libcudart = CudaRTLibrary()
    cumem_available = True
except ModuleNotFoundError:
    init_module = None
    python_create_and_map = None
    python_unmap_and_release = None
    lib_name = None

# handle 的格式：(device_id, aligned_size_bytes, d_mem_virtual_addr, p_memHandle_c_ptr)
# d_mem_virtual_addr 是 tensor.data_ptr() 的值，在 sleep/wake_up 全程不变
HandleType = tuple[int, int, int, int]


# -----------------------------------------------------------------------
# 内存分配记录（每块 cumem 内存对应一个）
# -----------------------------------------------------------------------
@dataclasses.dataclass
class AllocationData:
    handle: HandleType       # 完整元数据（传给 C 扩展用）
    tag: str                 # "weights" / "kv_cache" / "default"
    cpu_backup_tensor: torch.Tensor | None = None  # level 1 sleep 时的 CPU 备份


# -----------------------------------------------------------------------
# 封装 C 扩展调用
# -----------------------------------------------------------------------
def create_and_map(allocation_handle: HandleType) -> None:
    """wake_up 时调用：重新分配物理内存并映射回虚拟地址"""
    python_create_and_map(*allocation_handle)


def unmap_and_release(allocation_handle: HandleType) -> None:
    """sleep 时调用：解除映射，释放物理内存（虚拟地址保留）"""
    python_unmap_and_release(*allocation_handle)


# -----------------------------------------------------------------------
# 上下文管理器：创建 MemPool 并激活自定义分配器
# -----------------------------------------------------------------------
@contextmanager
def use_memory_pool_with_allocator(python_malloc_fn, python_free_func):
    """
    低层上下文：注册 Python 回调到 C 扩展，创建 CUDAPluggableAllocator 和 MemPool。
    在此上下文中的所有 torch.empty/zeros/randn 等分配操作都会调用 my_malloc。
    """
    init_module(python_malloc_fn, python_free_func)
    new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
        lib_name, "my_malloc", "my_free"
    )
    mem_pool = torch.cuda.memory.MemPool(new_alloc._allocator)
    with torch.cuda.memory.use_mem_pool(mem_pool):
        yield mem_pool, new_alloc


# -----------------------------------------------------------------------
# 单例主类
# -----------------------------------------------------------------------
class CuMemAllocator:
    """
    管理 cumem 内存池的单例类。

    工作流程：
    1. 在 use_memory_pool(tag="weights") 上下文中加载模型 → 所有权重张量被记录
    2. 在 use_memory_pool(tag="kv_cache") 上下文中分配 KV Cache → 所有 KV 张量被记录
    3. sleep(offload_tags=("weights",)) → "weights" 标签内存备份到 CPU，其余丢弃
    4. wake_up(tags=["weights"]) → 重新映射 "weights" 物理内存，搬回 GPU
    5. wake_up(tags=["kv_cache"]) → 重新映射 "kv_cache" 物理内存（内容为空）

    为什么必须是单例：
    C 扩展中用全局变量存储 malloc/free 回调，多个实例会互相覆盖，导致内存管理混乱。
    """

    instance: "CuMemAllocator | None" = None
    default_tag: str = "default"

    @staticmethod
    def get_instance() -> "CuMemAllocator":
        assert cumem_available, "cumem allocator is not available"
        if CuMemAllocator.instance is None:
            CuMemAllocator.instance = CuMemAllocator()
        return CuMemAllocator.instance

    def __init__(self):
        # 核心数据结构：GPU 虚拟地址 → 分配记录
        # key = tensor.data_ptr()（GPU 虚拟地址），value = AllocationData
        self.pointer_to_data: dict[int, AllocationData] = {}

        # 当前 use_memory_pool 上下文的 tag（malloc callback 用它来打标签）
        self.current_tag: str = CuMemAllocator.default_tag

        # 保持对 (mem_pool, allocator) 的强引用，防止 PyTorch 2.6+ 的 GC bug
        # 见 https://github.com/pytorch/pytorch/issues/146431
        # 见 vLLM PR #13456
        self.allocator_and_pools: dict[str, Any] = {}

        # 保持对 bound method 的强引用，防止回调对象被 GC 导致 C 层悬空指针
        # 见 vLLM PR #22724
        self.python_malloc_callback = self._python_malloc_callback
        self.python_free_callback = self._python_free_callback

    def _python_malloc_callback(self, allocation_handle: HandleType) -> None:
        """
        C 层 my_malloc 调用：记录新分配的内存信息。
        allocation_handle = (device, aligned_size, d_mem, p_memHandle_ptr)
        """
        py_d_mem = allocation_handle[2]  # 虚拟地址
        self.pointer_to_data[py_d_mem] = AllocationData(
            allocation_handle, self.current_tag
        )

    def _python_free_callback(self, ptr: int) -> HandleType:
        """
        C 层 my_free 调用：查找并删除内存记录，返回 handle 供 C 层解除映射。
        必须在此处同步 GPU，防止 in-flight kernel 在 unmap 后继续访问已释放地址。
        """
        data = self.pointer_to_data.pop(ptr)
        if data.cpu_backup_tensor is not None:
            data.cpu_backup_tensor = None
        # 关键：等待所有 GPU 操作完成（防止 CUDA_ERROR_ILLEGAL_ADDRESS）
        # 见 vLLM PR #43020
        torch.cuda.synchronize(data.handle[0])
        return data.handle

    @contextmanager
    def use_memory_pool(self, tag: str | None = None):
        """
        上下文管理器：在此上下文中创建的所有 GPU tensor 都使用 cumem 分配器。

        用法：
            with allocator.use_memory_pool(tag="weights"):
                model.load_weights()  # 所有权重 tensor 自动打上 "weights" 标签

        注意：与 expandable_segments 不兼容（PyTorch bug #147851）
        进入时自动禁用，退出时恢复。见 vLLM PR #40812
        """
        if tag is None:
            tag = CuMemAllocator.default_tag

        # 临时禁用 expandable_segments（与 MemPool 有 bug）
        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        expandable_was_enabled = "expandable_segments:True" in conf
        if expandable_was_enabled:
            torch.cuda.memory._set_allocator_settings("expandable_segments:False")

        old_tag = self.current_tag
        self.current_tag = tag
        try:
            with use_memory_pool_with_allocator(
                self.python_malloc_callback, self.python_free_callback
            ) as data:
                # 保持强引用！见 PR #13456
                self.allocator_and_pools[tag] = data
                yield

                # 处理 on-the-fly 量化场景：
                # 量化完成后临时 tensor 已被 free，但 PyTorch pluggable allocator
                # 不支持 empty_cache()（会崩溃，见 PyTorch bug #145168）
                # 手动找出 allocated_size==0 的块，调用 Python free callback + unmap
                # 见 vLLM PR #24731
                allocations = data[0].snapshot()
                for allocation in allocations:
                    if allocation["allocated_size"] == 0:
                        handle = self._python_free_callback(allocation["address"])
                        unmap_and_release(handle)
        finally:
            self.current_tag = old_tag
            if expandable_was_enabled:
                torch.cuda.memory._set_allocator_settings("expandable_segments:True")

    def sleep(self, offload_tags: tuple[str, ...] | str | None = None) -> None:
        """
        进入睡眠：解除所有 cumem 内存的物理映射，释放 GPU 显存。

        offload_tags：这些 tag 的内存先备份到 CPU，再解除映射（wake_up 后可以恢复）
                      其余 tag 的内存直接丢弃（wake_up 后内容为垃圾值）

        调用时机：GPU 上所有 tensor 操作已完成，无 in-flight kernel
        """
        if offload_tags is None:
            offload_tags = (CuMemAllocator.default_tag,)
        elif isinstance(offload_tags, str):
            offload_tags = (offload_tags,)

        total_bytes = 0
        backup_bytes = 0

        for ptr, data in self.pointer_to_data.items():
            handle = data.handle
            total_bytes += handle[1]

            if data.tag in offload_tags:
                # 备份到 CPU pinned memory（pin_memory 加速 cudaMemcpy）
                size_in_bytes = handle[1]
                cpu_backup_tensor = torch.empty(
                    size_in_bytes,
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=is_pin_memory_available(),
                )
                # GPU → CPU 数据拷贝（同步）
                libcudart.cudaMemcpy(cpu_backup_tensor.data_ptr(), ptr, size_in_bytes)
                data.cpu_backup_tensor = cpu_backup_tensor
                backup_bytes += size_in_bytes

            # 所有内存都解除映射（无论是否有 CPU 备份）
            unmap_and_release(handle)

        logger.info(
            "CuMemAllocator: sleep freed %.2f GiB memory in total, of which "
            "%.2f GiB is backed up in CPU and the rest %.2f GiB is discarded.",
            total_bytes / 1024**3,
            backup_bytes / 1024**3,
            (total_bytes - backup_bytes) / 1024**3,
        )

        # 强制 Python GC 和 CUDA allocator cache 清理
        gc.collect()
        torch.cuda.empty_cache()

    def wake_up(self, tags: list[str] | None = None) -> None:
        """
        从睡眠中恢复：重新映射物理内存到原虚拟地址，搬回 CPU 备份（如有）。

        tags=None：恢复所有内存
        tags=["weights"]：只恢复权重（RLHF 分步唤醒第一步）
        tags=["kv_cache"]：只恢复 KV Cache（RLHF 分步唤醒第二步）
        """
        for ptr, data in self.pointer_to_data.items():
            if tags is None or data.tag in tags:
                # 重新分配物理内存，映射回同一虚拟地址
                create_and_map(data.handle)
                # 如有 CPU 备份，搬回 GPU
                if data.cpu_backup_tensor is not None:
                    cpu_backup_tensor = data.cpu_backup_tensor
                    size_in_bytes = (
                        cpu_backup_tensor.numel() * cpu_backup_tensor.element_size()
                    )
                    # CPU → GPU 数据拷贝
                    libcudart.cudaMemcpy(
                        ptr,                              # GPU 虚拟地址（与 sleep 前完全相同！）
                        cpu_backup_tensor.data_ptr(),
                        size_in_bytes
                    )
                    data.cpu_backup_tensor = None

    def get_current_usage(self) -> int:
        """返回当前 cumem 池中总分配字节数（含 sleeping 状态）"""
        return sum(data.handle[1] for data in self.pointer_to_data.values())
```

**验证 Step 4**（独立测试，不需要完整 vLLM）：

```python
# test_cumem_basic.py
import torch
from vllm.device_allocator.cumem import CuMemAllocator

allocator = CuMemAllocator.get_instance()

# 1. 在 cumem pool 中分配 tensor
with allocator.use_memory_pool(tag="test"):
    x = torch.ones(1024, 1024, device="cuda")

addr_before = x.data_ptr()
print(f"x.mean() = {x.mean().item()}")   # 1.0
print(f"data_ptr = {hex(addr_before)}")   # 记住这个地址

# 2. Sleep：备份 x 到 CPU，释放 GPU 显存
free_before = torch.cuda.mem_get_info()[0]
allocator.sleep(offload_tags=("test",))
free_after = torch.cuda.mem_get_info()[0]
print(f"显存增加: {(free_after - free_before) / 1024**2:.1f} MB")  # 应该增加

# 3. Wake up：重映射，数据搬回 GPU
allocator.wake_up()

# 关键验证点
assert x.data_ptr() == addr_before, "地址改变了！实现有误"
assert torch.allclose(x, torch.ones(1024, 1024, device="cuda")), "数据不一致！"
print("✓ 地址不变，数据正确")
```

---

### Step 5：修改 Worker（`vllm/v1/worker/gpu_worker.py`）

Worker 是单 GPU 的全权代理，需要做 4 处修改。

#### 5.1 在 `__init__` 中添加 `_sleep_saved_buffers` 字段

找到 `Worker.__init__` 方法，在方法体末尾添加：

```python
# 在 Worker.__init__ 末尾添加
# 用于 level 2 sleep 前保存模型 buffers（buffers 不在 cumem pool 中）
self._sleep_saved_buffers: dict[str, torch.Tensor] = {}
```

> **为什么需要保存 buffers？**
> `model.named_buffers()` 包含不参与训练但影响推理的常量，例如 LLaMA 的 RoPE `cos`/`sin` 缓存（`model.layers.0.self_attn.rotary_emb.cos_cached`）。这些 tensor **在模型构造时** 就已分配，不在 cumem pool 中，因此 `allocator.sleep()` 不会处理它们。level 2 sleep 时，需要手动 clone 到 CPU 保存，wake_up 时手动恢复。

#### 5.2 添加辅助方法 `_maybe_get_memory_pool_context`

在 `Worker` 类中添加：

```python
from contextlib import AbstractContextManager, nullcontext

def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
    """
    如果 enable_cumem_allocator=True，返回打标签的 cumem 内存池上下文；
    否则返回 nullcontext（什么都不做，代码路径完全透明）。

    这个方法让调用方代码不需要判断 sleep mode 是否开启：
        with self._maybe_get_memory_pool_context("weights"):
            self.model_runner.load_model()
    """
    if not self.vllm_config.model_config.enable_cumem_allocator:
        return nullcontext()

    from vllm.device_allocator.cumem import CuMemAllocator

    allocator = CuMemAllocator.get_instance()
    if tag == "weights":
        # 单例约束检查：首次进入 weights pool 时，pool 必须为空
        # 如果非空，说明进程中已有另一个 vLLM 实例在使用，会导致冲突
        assert allocator.get_current_usage() == 0, (
            "CuMem allocator can only be used for one instance per process."
        )
    return allocator.use_memory_pool(tag=tag)
```

#### 5.3 修改 `load_model` 方法

找到 `load_model` 方法，用 `_maybe_get_memory_pool_context("weights")` 上下文包裹模型加载：

```python
def load_model(self, *, load_dummy_weights: bool = False) -> None:
    with (
        self._maybe_get_memory_pool_context(tag="weights"),  # ← 新增这一行
        # ... 其他已有的上下文管理器 ...
    ):
        self.model_runner.load_model(load_dummy_weights=load_dummy_weights)
```

> **实际代码位置**：`vllm/v1/worker/gpu_worker.py` 的 `Worker.load_model` 方法，在现有代码基础上用 `with self._maybe_get_memory_pool_context(tag="weights"):` 包裹 `self.model_runner.load_model(...)` 调用即可。

#### 5.4 修改 `initialize_from_config`（或 `initialize_cache`）方法

找到分配 KV Cache 的方法（旧版可能叫 `initialize_cache`，新版叫 `initialize_from_config`），用 `_maybe_get_memory_pool_context("kv_cache")` 包裹：

```python
def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
    self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks

    with self._maybe_get_memory_pool_context(tag="kv_cache"):  # ← 新增这一行
        self.model_runner.initialize_kv_cache(kv_cache_config)

    # 注意：KV zero metadata 故意在 pool 外分配（不参与 sleep/wake 循环）
    if kv_cache_config.needs_kv_cache_zeroing:
        self.model_runner._init_kv_zero_meta()
```

#### 5.5 添加 `sleep` 方法

在 `Worker` 类中添加：

```python
def sleep(self, level: int = 1) -> None:
    """
    释放 GPU 显存。

    level=1：权重 offload 到 CPU pinned memory，KV Cache 直接丢弃
    level=2：所有内存直接丢弃（调用方后续会更新权重，旧权重不需要保留）
    """
    from vllm.device_allocator.cumem import CuMemAllocator

    free_bytes_before = torch.cuda.mem_get_info()[0]

    # level 2：在丢弃权重前，先把 buffers 保存到 CPU
    # buffers 不在 cumem pool 中，allocator.sleep() 不会处理它们
    if level == 2:
        model = self.model_runner.model
        self._sleep_saved_buffers = {
            name: buffer.cpu().clone()
            for name, buffer in model.named_buffers()
        }

    allocator = CuMemAllocator.get_instance()
    # level 1：备份 "weights" 到 CPU，丢弃 "kv_cache"
    # level 2：所有 tag 都直接丢弃（offload_tags=空元组）
    allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())

    free_bytes_after, total = torch.cuda.mem_get_info()
    freed = free_bytes_after - free_bytes_before
    used = total - free_bytes_after
    assert freed >= 0, "Sleep 后显存使用量增加，逻辑有误"
    logger.info(
        "Sleep mode freed %.2f GiB, %.2f GiB still in use.",
        freed / 1024**3, used / 1024**3,
    )
```

#### 5.6 添加 `wake_up` 方法

```python
def wake_up(self, tags: list[str] | None = None) -> None:
    """
    恢复 GPU 显存。

    tags=None：恢复所有内存（完整唤醒）
    tags=["weights"]：只恢复权重（分步唤醒第一步）
    tags=["kv_cache"]：只恢复 KV Cache（分步唤醒第二步）
    """
    from vllm.device_allocator.cumem import CuMemAllocator

    allocator = CuMemAllocator.get_instance()
    allocator.wake_up(tags)

    # 恢复 level 2 sleep 前保存的 buffers
    if len(self._sleep_saved_buffers):
        model = self.model_runner.model
        for name, buffer in model.named_buffers():
            if name in self._sleep_saved_buffers:
                # copy_ 是原地操作，不改变 tensor 对象，不影响 CUDA Graph 中的指针
                buffer.data.copy_(self._sleep_saved_buffers[name].data)
        self._sleep_saved_buffers = {}

    # KV Cache 唤醒后需要修复 FP8 scale（见 Step 5.7）
    if tags is None or "kv_cache" in tags:
        self.model_runner.post_kv_cache_wake_up()
```

#### 5.7 在 `GPUModelRunner` 中添加 FP8 修复（`gpu_model_runner.py`）

`wake_up` 后新映射的物理内存内容是**未定义的**（操作系统随机内容）。对于 FP8 KV Cache，这会导致 scale 值为 0，进而让所有 KV 值变为 0，输出乱码。在 `gpu_model_runner.py` 的 `GPUModelRunner` 类中添加：

```python
def post_kv_cache_wake_up(self) -> None:
    """wake_up 后修复 FP8 KV Cache。非 FP8 情况下是空操作。"""
    self.init_fp8_kv_scales()

@torch.inference_mode()
def init_fp8_kv_scales(self) -> None:
    """
    FP8 KV Cache wake_up 后必须做两件事：
    1. zero_() 清零 KV Cache（新映射的物理内存内容是随机垃圾）
    2. 将 Attention 层的 _k_scale/_v_scale 重置为 1.0
       （新映射的内存中这些 tensor 的值是 0，0 * KV = 0 → 输出乱码）
    """
    from vllm.attention import Attention
    from vllm.model_executor.layers.mamba.mla.mla_attn import MLAAttention

    # 非 FP8 量化：不需要处理
    if not is_quantized_kv_cache(self.cache_config.cache_dtype):
        return

    # 清零所有 KV Cache tensor
    for cache_tensor in getattr(self, "kv_caches", []):
        if cache_tensor is not None:
            cache_tensor.zero_()

    # 重置 Attention 层的 scale
    k_attrs = ("_k_scale", "k_scale")
    v_attrs = ("_v_scale", "v_scale")
    for name, module in self.compilation_config.static_forward_context.items():
        if isinstance(module, (Attention, MLAAttention)):
            for attr in k_attrs:
                if hasattr(module, attr):
                    param = getattr(module, attr)
                    if isinstance(param, torch.Tensor):
                        param.fill_(1.0)
            for attr in v_attrs:
                if hasattr(module, attr):
                    param = getattr(module, attr)
                    if isinstance(param, torch.Tensor):
                        param.fill_(1.0)
```

---

### Step 6：修改配置层

#### 6.1 在 `ModelConfig` 中添加字段（`vllm/config/model.py` 或 `vllm/config.py`）

找到 `ModelConfig` dataclass，添加两个新字段：

```python
@dataclasses.dataclass
class ModelConfig:
    # ...已有字段...

    # 新增：sleep mode 开关（用户通过 --enable-sleep-mode 开启）
    enable_sleep_mode: bool = False

    # 新增：cumem 分配器开关（可独立于 sleep mode 开启，但 sleep mode 依赖它）
    enable_cumem_allocator: bool = False
```

在 `ModelConfig.__post_init__` 方法中添加验证逻辑（找到 `def __post_init__` 方法，在其中添加）：

```python
def __post_init__(self):
    # ...已有验证...

    # sleep mode 前提检查
    if self.enable_sleep_mode:
        # 检查 1：平台是否支持（必须是 CUDA 或 ROCm）
        if not current_platform.is_sleep_mode_available():
            raise ValueError(
                "Sleep mode is not supported on current platform. "
                "Only CUDA and ROCm are supported."
            )
        # 连锁效应：sleep mode 必须使用 cumem 分配器
        if not self.enable_cumem_allocator:
            logger.info(
                "Enabling cumem allocator because sleep mode requires it."
            )
            self.enable_cumem_allocator = True

    # cumem 分配器前提检查（CPU-only 环境下无法编译 C 扩展）
    if self.enable_cumem_allocator:
        from vllm.device_allocator.cumem import cumem_available
        if not cumem_available:
            raise ValueError(
                "cumem allocator is not available. "
                "Ensure the C extension was compiled (requires CUDA/ROCm)."
            )
```

#### 6.2 在平台接口中添加 `is_sleep_mode_available`（`vllm/platforms/interface.py`）

找到 `Platform` 基类，添加：

```python
class Platform:
    # ...已有方法...

    def is_sleep_mode_available(self) -> bool:
        """是否支持 sleep mode（基于 CUDA VMM API）"""
        # CUDA 和 ROCm 都支持
        return self._enum in (PlatformEnum.CUDA, PlatformEnum.ROCM)
```

#### 6.3 在 `EngineArgs` 中添加 CLI 参数（`vllm/engine/arg_utils.py`）

找到 `EngineArgs` dataclass，添加字段：

```python
@dataclasses.dataclass
class EngineArgs:
    # ...已有字段...
    enable_sleep_mode: bool = ModelConfig.enable_sleep_mode
    enable_cumem_allocator: bool = ModelConfig.enable_cumem_allocator
```

找到 `EngineArgs.add_cli_args` 方法，添加 argument：

```python
@staticmethod
def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # ...已有参数...

    parser.add_argument(
        "--enable-sleep-mode",
        action="store_true",
        default=EngineArgs.enable_sleep_mode,
        help="Enable sleep mode for RLHF workloads. Allows releasing GPU "
             "memory while keeping the process alive (no model reload needed).",
    )
    parser.add_argument(
        "--enable-cumem-allocator",
        action="store_true",
        default=EngineArgs.enable_cumem_allocator,
        help="Use the cumem-based CUDA memory allocator. "
             "Automatically enabled when --enable-sleep-mode is set.",
    )
    return parser
```

找到 `EngineArgs.create_engine_config` 或类似方法，确保字段传递给 `ModelConfig`：

```python
def create_model_config(self) -> ModelConfig:
    return ModelConfig(
        # ...已有参数...
        enable_sleep_mode=self.enable_sleep_mode,
        enable_cumem_allocator=self.enable_cumem_allocator,
    )
```

---

### Step 7：修改 Executor（`vllm/v1/executor/abstract.py`）

Executor 管理一组 Worker（TP 场景下是多个），通过 `collective_rpc` 广播命令。

找到 `Executor.__init__` 方法，添加状态字段：

```python
class Executor:
    def __init__(self, ...):
        # ...已有初始化...
        self.is_sleeping: bool = False          # 是否处于 sleep 状态
        self.sleeping_tags: set[str] = set()    # 当前哪些 tags 处于 sleep 状态
```

在 `Executor` 类中添加 `sleep` 和 `wake_up` 方法：

```python
import time

def sleep(self, level: int = 1) -> None:
    """
    广播 sleep 命令到所有 Worker（TP 场景下所有 GPU 同步执行）。
    """
    if self.is_sleeping:
        logger.warning("Executor is already sleeping.")
        return

    t0 = time.perf_counter()
    # collective_rpc 在单进程场景下直接调用 Worker.sleep()
    # 在多进程 TP 场景下通过 IPC 管道广播到每个 Worker 进程
    self.collective_rpc("sleep", kwargs=dict(level=level))
    logger.info(
        "It took %.6f seconds to fall asleep.",
        time.perf_counter() - t0
    )

    self.sleeping_tags = {"weights", "kv_cache"}
    self.is_sleeping = True


def wake_up(self, tags: list[str] | None = None) -> None:
    """
    广播 wake_up 命令到所有 Worker。
    支持分步唤醒：先 wake_up(["weights"])，再 wake_up(["kv_cache"])。
    """
    if not self.is_sleeping:
        logger.warning("Executor is not sleeping.")
        return

    if tags:
        for tag in tags:
            if tag not in self.sleeping_tags:
                logger.warning(
                    "Tag %s is not in sleeping tags %s", tag, self.sleeping_tags
                )
                return

    t0 = time.perf_counter()
    self.collective_rpc("wake_up", kwargs=dict(tags=tags))
    logger.info(
        "It took %.6f seconds to wake up tags %s.",
        time.perf_counter() - t0,
        tags if tags is not None else self.sleeping_tags,
    )

    if tags:
        for tag in tags:
            self.sleeping_tags.discard(tag)
    else:
        self.sleeping_tags.clear()

    # 只有所有 tags 都唤醒了，才认为整体不再 sleeping
    if not self.sleeping_tags:
        self.is_sleeping = False
```

---

### Step 8：修改 Engine Core（`vllm/v1/engine/core.py`）

Engine Core 负责协调调度器与执行器。sleep 前必须先暂停调度器；wake_up 后必须恢复调度器。

#### 8.1 添加调度器暂停/恢复方法

先定义 `PauseState` 枚举（如果旧版没有，需要创建）：

```python
import enum

class PauseState(enum.IntEnum):
    UNPAUSED = 0     # 正常运行，接受并处理所有请求
    PAUSED_NEW = 1   # 不接受新请求，当前 in-flight 请求继续处理（abort/wait 模式中间态）
    PAUSED_ALL = 2   # 所有请求都暂停（keep 模式）
```

在 `EngineCore` 类中添加暂停相关方法：

```python
def pause_scheduler(
    self, mode: str = "abort", clear_cache: bool = True
) -> None:
    """
    暂停调度器。

    mode="abort"（默认）：立即中止所有 in-flight 请求
    mode="keep"：暂停调度，in-flight 请求进入 PAUSED_ALL 状态（wake_up 后继续）
    """
    if mode == "abort":
        # 中止所有进行中的请求（状态设为 FINISHED_ABORTED）
        self.scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)

    pause_state = PauseState.PAUSED_ALL if mode == "keep" else PauseState.PAUSED_NEW
    self.scheduler.set_pause_state(pause_state)

    if clear_cache:
        # 重要：清空 prefix cache！
        # prefix cache 中存储了指向 KV Cache blocks 的引用
        # sleep 后 KV Cache 已被释放，这些引用指向无效内存
        # 不清空会导致 wake_up 后推理结果错误
        self._reset_caches()  # 清空 prefix cache + multimodal cache + encoder cache


def resume_scheduler(self) -> None:
    """恢复调度器"""
    self.scheduler.set_pause_state(PauseState.UNPAUSED)


def is_scheduler_paused(self) -> bool:
    """调度器是否处于暂停状态"""
    return self.scheduler.pause_state != PauseState.UNPAUSED
```

#### 8.2 添加 `sleep` 方法

```python
def sleep(self, level: int = 1, mode: str = "abort") -> None:
    """
    让引擎进入睡眠。

    level=0：仅暂停调度（不动 GPU 内存，适合短暂暂停）
    level=1：offload 权重到 CPU，丢弃 KV Cache
    level=2：丢弃所有 GPU 内存（权重 + KV Cache）

    mode="abort"：立即中止所有请求（RLHF 场景下一轮 rollout 结束后使用）
    mode="keep"：保留请求到队列，wake_up 后继续（level 0 的典型用法）
    """
    # Step 1：暂停调度器
    # level 0 不需要清 prefix cache（GPU 内存没变，prefix cache 仍然有效）
    clear_prefix_cache = (level >= 1)
    self.pause_scheduler(mode=mode, clear_cache=clear_prefix_cache)

    # level 0：只暂停调度，不动 GPU 内存
    if level < 1:
        return

    # level 1/2：释放 GPU 内存
    self.model_executor.sleep(level)
```

#### 8.3 添加 `wake_up` 方法

```python
def wake_up(self, tags: list[str] | None = None) -> None:
    """
    唤醒引擎。

    tags=None：完整唤醒（恢复所有 GPU 内存 + 恢复调度）
    tags=["weights"]：只唤醒权重（分步唤醒第一步，调度仍暂停）
    tags=["kv_cache"]：只唤醒 KV Cache（分步唤醒第二步，调度仍暂停）
    tags=["scheduling"]：只恢复调度（用于 level 0 的 wake_up）

    完整 RLHF 分步唤醒流程：
        sleep(level=2)
        wake_up(tags=["weights"])   # 第一步：只唤醒权重
        update_weights(...)          # 训练框架更新权重
        wake_up(tags=["kv_cache"]) # 第二步：唤醒 KV Cache，恢复推理能力
    """
    # 处理 "scheduling" 虚拟 tag（level 0 的唤醒）
    if tags is not None and "scheduling" in tags:
        tags = [t for t in tags if t != "scheduling"]

    # 恢复 GPU 内存（如果有非空的 tags 需要处理）
    if tags is None or tags:
        self.model_executor.wake_up(tags)

    # 恢复调度器（所有级别的唤醒都需要）
    self.resume_scheduler()


def is_sleeping(self) -> bool:
    """引擎在任何层面 sleeping（调度暂停或 GPU 内存释放）都返回 True"""
    return self.is_scheduler_paused() or self.model_executor.is_sleeping
```

---

### Step 9：修改 Entrypoint（`vllm/entrypoints/llm.py`）

这是用户可见的 API 层，改动最少。在 `LLM` 类中添加：

```python
from vllm.v1.engine.core import PauseMode  # 类型别名：Literal["abort", "wait", "keep"]

class LLM:
    # ...已有方法...

    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
        """
        让推理引擎进入睡眠，释放 GPU 显存。

        Args:
            level:
                0 - 仅暂停调度，不动 GPU 内存（短暂暂停，维持所有显存）
                1 - offload 模型权重到 CPU，丢弃 KV Cache（默认）
                    wake_up() 后可继续使用同一模型推理
                2 - 丢弃所有 GPU 内存（权重 + KV Cache，不备份到 CPU）
                    适合 RLHF 场景：训练后权重已更新，旧权重无用
            mode:
                "abort" - 立即中止所有 in-flight 请求（默认）
                "keep"  - 暂停但保留请求（wake_up 后继续）
                "wait"  - 等所有 in-flight 请求完成再 sleep（仅异步引擎）

        典型用法（RLHF level 2）：
            llm.sleep(level=2)
            # ... 训练框架更新权重 ...
            llm.wake_up(tags=["weights"])
            llm.collective_rpc("reload_weights")
            llm.wake_up(tags=["kv_cache"])
            output = llm.generate(...)
        """
        self.llm_engine.sleep(level=level, mode=mode)

    def wake_up(self, tags: list[str] | None = None) -> None:
        """
        唤醒引擎，恢复 GPU 显存。

        Args:
            tags: 要唤醒的内存类型。
                None         - 唤醒所有内存（完整唤醒）
                ["weights"]  - 只唤醒权重（后续可更新权重）
                ["kv_cache"] - 只唤醒 KV Cache（唤醒权重后使用）
                ["scheduling"] - 只恢复调度（level 0 的唤醒）

        注意：分步唤醒时（先 weights 后 kv_cache），两步都完成后才能 generate()
        """
        self.llm_engine.wake_up(tags)
```

---

### Step 10：在线服务 API（可选，`vllm/entrypoints/serve/`）

如果需要通过 HTTP 控制 sleep mode，创建 API 路由（只在 `VLLM_SERVER_DEV_MODE=1` 时挂载）：

创建文件 `vllm/entrypoints/serve/sleep/api_router.py`：

```python
# vllm/entrypoints/serve/sleep/api_router.py
import os
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response

router = APIRouter()


@router.post("/sleep")
async def sleep(raw_request: Request):
    """
    POST /sleep?level=1&mode=abort
    让在线服务进入睡眠。
    """
    level = int(raw_request.query_params.get("level", "1"))
    mode = raw_request.query_params.get("mode", "abort")
    await engine_client(raw_request).sleep(level, mode)
    return Response(status_code=200)


@router.post("/wake_up")
async def wake_up(raw_request: Request):
    """
    POST /wake_up
    POST /wake_up?tags=weights&tags=kv_cache
    唤醒在线服务。
    """
    tags = raw_request.query_params.getlist("tags") or None
    await engine_client(raw_request).wake_up(tags)
    return Response(status_code=200)


@router.get("/is_sleeping")
async def is_sleeping(raw_request: Request):
    """
    GET /is_sleeping
    返回 {"is_sleeping": true/false}
    """
    result = await engine_client(raw_request).is_sleeping()
    return JSONResponse(content={"is_sleeping": result})


def attach_router(app: FastAPI) -> None:
    """只在 VLLM_SERVER_DEV_MODE=1 时挂载（不对外暴露）"""
    if not os.environ.get("VLLM_SERVER_DEV_MODE"):
        return
    app.include_router(router)
```

在 `vllm/entrypoints/serve/app.py` 中引入并挂载：

```python
from vllm.entrypoints.serve.sleep.api_router import attach_router as attach_sleep_router

def build_app(args) -> FastAPI:
    app = FastAPI(...)
    # ...已有路由...
    attach_sleep_router(app)  # 挂载 sleep API（仅 dev mode）
    return app
```

---

### Step 11：边界情况核查清单

按照历史 PR 发现的 bug，逐一确认你的实现：

| # | 问题 | 检查点 | 出处 |
|---|------|--------|------|
| 1 | PyTorch GC 过早回收 allocator | `CuMemAllocator.__init__` 中 `self.allocator_and_pools[tag] = data` 保持强引用 | PR #13456 |
| 2 | bound method 被 GC 后 C 层回调悬空 | `self.python_malloc_callback = self._python_malloc_callback`（方法级别强引用）| PR #22724 |
| 3 | C 层 malloc/free 中未获取 GIL | `my_malloc`/`my_free` 开头必须有 `PyGILState_Ensure()` | 基础要求 |
| 4 | in-flight kernel 访问已释放内存 | `_python_free_callback` 中调用 `torch.cuda.synchronize(device)` | PR #43020 |
| 5 | expandable_segments 与 MemPool 冲突 | `use_memory_pool` 进入时临时禁用 `expandable_segments:True` | PR #40812 |
| 6 | FP8 KV Cache wake_up 后输出乱码 | 添加 `post_kv_cache_wake_up()` → `init_fp8_kv_scales()` | PR #28783 |
| 7 | level 2 sleep 后 buffers 丢失 | `sleep(level=2)` 前 `clone` buffers，`wake_up` 后 `copy_` 恢复 | PR #16889 |
| 8 | sleep 后 prefix cache 指向无效内存 | `pause_scheduler` 时调用 `_reset_caches()` | 设计要求 |
| 9 | on-the-fly 量化产生的临时内存泄漏 | `use_memory_pool` 退出时扫描 `snapshot()` 中 `allocated_size==0` 的块并手动 unmap | PR #24731 |
| 10 | 多实例进程中 C 全局变量被覆盖 | `_maybe_get_memory_pool_context("weights")` 时断言 `get_current_usage() == 0` | 设计要求 |

---

### Step 12：端到端测试

完整的测试套件参考 `tests/basic_correctness/test_cumem.py`，以下是最小可验证集：

```python
# tests/test_sleep_mode.py
import torch
import pytest
from vllm import LLM, SamplingParams
from vllm.device_allocator.cumem import CuMemAllocator


# ===================================================================
# 测试 1：C 扩展基础功能
# ===================================================================
def test_c_extension_loaded():
    """验证 C 扩展编译成功并可以导入"""
    from vllm.cumem_allocator import (
        init_module, python_create_and_map, python_unmap_and_release
    )
    assert callable(init_module)
    assert callable(python_create_and_map)
    assert callable(python_unmap_and_release)


# ===================================================================
# 测试 2：CuMemAllocator 基础 sleep/wake_up
# ===================================================================
def test_cumem_allocator_basic():
    """验证 sleep/wake_up 后 tensor 地址不变、数据正确"""
    allocator = CuMemAllocator.get_instance()

    with allocator.use_memory_pool(tag="test"):
        x = torch.ones(1024, 1024, device="cuda")

    addr_before = x.data_ptr()
    value_before = x.mean().item()

    # Sleep：备份到 CPU，释放 GPU 显存
    free_before = torch.cuda.mem_get_info()[0]
    allocator.sleep(offload_tags=("test",))
    free_after_sleep = torch.cuda.mem_get_info()[0]
    assert free_after_sleep > free_before, "Sleep 后显存应该增加"

    # Wake up：搬回 GPU
    allocator.wake_up()

    # 关键验证
    assert x.data_ptr() == addr_before, f"地址改变！before={hex(addr_before)}, after={hex(x.data_ptr())}"
    assert abs(x.mean().item() - value_before) < 1e-5, "数据不一致"
    print(f"✓ sleep/wake_up 正确：地址={hex(addr_before)}, 值={value_before}")


# ===================================================================
# 测试 3：CUDA Graph 兼容性
# ===================================================================
def test_cumem_with_cuda_graph():
    """CUDA Graph 捕获的地址在 sleep/wake_up 后仍有效"""
    allocator = CuMemAllocator.get_instance()

    with allocator.use_memory_pool(tag="test"):
        x = torch.ones(1024, device="cuda")
        y = torch.empty(1024, device="cuda")

    # 捕获 CUDA Graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y.copy_(x + 1)

    g.replay()
    assert torch.allclose(y, x + 1), "图执行结果不正确"

    allocator.sleep(offload_tags=("test",))
    allocator.wake_up()

    # Wake up 后图中的地址仍然有效
    g.replay()
    assert torch.allclose(y, x + 1), "wake_up 后图执行结果不正确"
    print("✓ CUDA Graph 在 sleep/wake_up 后仍然正确")


# ===================================================================
# 测试 4：端到端 level 1（offload 权重）
# ===================================================================
def test_end_to_end_level1():
    """level 1 sleep 后 wake_up，推理结果不变"""
    llm = LLM("facebook/opt-125m", enable_sleep_mode=True)
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


# ===================================================================
# 测试 5：端到端 level 2（深度睡眠）+ 分步唤醒
# ===================================================================
def test_end_to_end_level2_partial_wakeup():
    """level 2 sleep + 分步唤醒（先 weights 后 kv_cache）"""
    llm = LLM("facebook/opt-125m", enable_sleep_mode=True)
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

    # （RLHF 场景：此处可以更新权重）
    llm.collective_rpc("reload_weights")

    # Step 3：唤醒 KV Cache，恢复完整推理能力
    llm.wake_up(tags=["kv_cache"])

    output2 = llm.generate(prompt, params)
    assert output1[0].outputs[0].text == output2[0].outputs[0].text, \
        f"输出不一致！\nbefore: {output1[0].outputs[0].text}\nafter: {output2[0].outputs[0].text}"
    print(f"✓ Level 2 分步唤醒正确: '{output1[0].outputs[0].text}'")


# ===================================================================
# 运行测试
# ===================================================================
if __name__ == "__main__":
    test_c_extension_loaded()
    print("--- 测试 1 通过：C 扩展加载正常 ---\n")

    test_cumem_allocator_basic()
    print("--- 测试 2 通过：基础 sleep/wake_up ---\n")

    test_cumem_with_cuda_graph()
    print("--- 测试 3 通过：CUDA Graph 兼容 ---\n")

    test_end_to_end_level1()
    print("--- 测试 4 通过：端到端 level 1 ---\n")

    test_end_to_end_level2_partial_wakeup()
    print("--- 测试 5 通过：端到端 level 2 分步唤醒 ---\n")

    print("✅ 所有测试通过！")
```

运行测试：

```bash
# 运行所有 sleep mode 相关测试
.venv/bin/python -m pytest tests/basic_correctness/test_cumem.py -v

# 运行单个测试
.venv/bin/python -m pytest tests/basic_correctness/test_cumem.py::test_end_to_end -v -s

# 验证基础功能（不需要 GPU）
.venv/bin/python tests/test_sleep_mode.py
```

---

### 完整改动文件清单

| 步骤 | 文件 | 操作 | 关键改动 |
|------|------|------|---------|
| Step 1 | `csrc/cumem_allocator.cpp` | **新建** | CUDA VMM C 扩展（`my_malloc`/`my_free` + 3 个 Python 导出函数）|
| Step 2 | `CMakeLists.txt` | **修改** | 添加 `cumem_allocator` 扩展目标和编译规则 |
| Step 4 | `vllm/device_allocator/__init__.py` | **新建** | 空文件（Python package 标识）|
| Step 4 | `vllm/device_allocator/cumem.py` | **新建** | `CuMemAllocator` 单例类（内存池管理）|
| Step 5 | `vllm/v1/worker/gpu_worker.py` | **修改** | 添加 `_sleep_saved_buffers`、`_maybe_get_memory_pool_context()`、`sleep()`、`wake_up()`；修改 `load_model` 和 `initialize_from_config` 加上 cumem 上下文 |
| Step 5 | `vllm/v1/worker/gpu_model_runner.py` | **修改** | 添加 `post_kv_cache_wake_up()` 和 `init_fp8_kv_scales()` |
| Step 6 | `vllm/config/model.py` | **修改** | `ModelConfig` 添加 `enable_sleep_mode`、`enable_cumem_allocator` 字段及验证逻辑 |
| Step 6 | `vllm/platforms/interface.py` | **修改** | `Platform` 添加 `is_sleep_mode_available()` 方法 |
| Step 6 | `vllm/engine/arg_utils.py` | **修改** | `EngineArgs` 添加 `--enable-sleep-mode` 和 `--enable-cumem-allocator` CLI 参数 |
| Step 7 | `vllm/v1/executor/abstract.py` | **修改** | `Executor` 添加 `is_sleeping`、`sleeping_tags` 字段和 `sleep()`、`wake_up()` 方法 |
| Step 8 | `vllm/v1/engine/core.py` | **修改** | `EngineCore` 添加 `PauseState`、`pause_scheduler()`、`resume_scheduler()`、`sleep()`、`wake_up()`、`is_sleeping()` |
| Step 9 | `vllm/entrypoints/llm.py` | **修改** | `LLM` 添加 `sleep()` 和 `wake_up()` 公共 API |
| Step 10 | `vllm/entrypoints/serve/sleep/api_router.py` | **新建（可选）** | HTTP `/sleep`、`/wake_up`、`/is_sleeping` 端点 |

---

## 8. vLLM 模块学习路线图

推荐按以下顺序学习，每个阶段有明确的学习目标和验证方式。

### 阶段一：请求生命周期（第 1-2 周）

**目标**：一个 prompt 从输入到输出，完整经历哪些步骤？

**阅读顺序**：

1. **`vllm/entrypoints/llm.py`** — `LLM.generate()` 方法
   - 关注：如何把 `List[str]` 变成 `List[Request]`
   - 关键方法：`_validate_and_add_requests()`, `_run_engine()`

2. **`vllm/v1/engine/llm_engine.py`** — `LLMEngine`
   - 关注：`generate()` 的主循环，`add_request()` → `step()` 的流程
   - 关键方法：`step()` 调用 `scheduler.schedule()` 再调用 `executor.execute_model()`

3. **`vllm/v1/core/scheduler.py`** — `Scheduler.schedule()`
   - 关注：如何决定哪些请求参与下一次 forward？
   - 关键概念：running queue、waiting queue、preemption（抢占）

4. **`vllm/v1/worker/gpu_model_runner.py`** — `execute_model()`
   - 关注：`SchedulerOutput` 如何转换为模型的实际输入（input_ids, position_ids, attention_mask 等）

**动手练习**：
```python
llm = LLM("facebook/opt-125m")
# 在 LLMEngine.step() 开头加一行 print，观察每次 step 的请求数
```

### 阶段二：KV Cache 内存管理（第 3 周）

**目标**：KV Cache 如何分配、管理和回收？

5. **`vllm/v1/core/kv_cache_manager.py`** — `KVCacheManager`
   - 关注：Block 的分配/释放/共享（prefix sharing）
   - 关键概念：block table、ref counting、prefix hashing

6. **`vllm/v1/kv_cache_interface.py`** — `KVCacheSpec`, `KVCacheConfig`
   - 关注：KV Cache 的规格描述（每块多大、哪些 layer 共享等）

7. **`vllm/v1/worker/gpu_model_runner.py`** — `initialize_kv_cache()`
   - 关注：物理 tensor 如何分配，如何与 model 的 attention layer 关联

**动手练习**：
```python
# 修改 gpu_memory_utilization（0.9 → 0.5），观察 KV Cache blocks 数量变化
llm = LLM("facebook/opt-125m", gpu_memory_utilization=0.5)
```

### 阶段三：分布式执行（第 4 周）

**目标**：多卡 TP 时，请求如何被分发到所有 GPU？

8. **`vllm/v1/executor/multiproc_executor.py`** — `MultiprocExecutor`
   - 关注：Worker 进程如何启动，`collective_rpc` 如何通过 IPC 发送消息

9. **`vllm/distributed/parallel_state.py`** — 分布式状态管理
   - 关注：`init_distributed_environment()`, tensor parallel group 的创建

10. **`vllm/distributed/communication_op.py`** — 通信原语
    - 关注：`tensor_model_parallel_all_reduce()` 等操作

**动手练习**：
```python
# 启动 TP=2，观察进程数量
llm = LLM("facebook/opt-125m", tensor_parallel_size=2)
# 用 ps aux | grep python 查看有几个进程
```

### 阶段四：模型加载与权重管理（第 5 周）

**目标**：权重如何从磁盘加载到 GPU？如何就地更新权重？

11. **`vllm/model_executor/models/llama.py`** — LLaMA 模型架构
    - 关注：`LlamaForCausalLM.load_weights()` 如何处理 HuggingFace 权重名映射

12. **`vllm/model_executor/model_loader/loader.py`** — 权重加载器
    - 关注：`DefaultModelLoader`, `ShardedStateLoader` 等不同加载方式

13. **`vllm/v1/worker/gpu_worker.py`** — `reload_weights()`
    - 关注：如何在不重建模型的情况下就地更新权重

**动手练习**：在 `load_weights()` 中加 print，观察加载了多少参数。

### 阶段五：编译优化（第 6 周）

**目标**：`torch.compile` 和 CUDA Graph 如何加速推理？

14. **`vllm/compilation/`** — vLLM 的编译 pass 系统
    - 关注：`PostGradPassManager`, 各种 pass 的职责

15. **`vllm/v1/worker/gpu_model_runner.py`** — `capture_model()` (CUDA Graph 捕获)
    - 关注：为什么需要固定 batch size 捕获？如何 replay？

16. **`vllm/config/compilation.py`** — 编译配置
    - 关注：`CompilationMode` 的几种模式（eager/inductor/vllm_compile）

---

## 9. 举一反三：独立设计 Multi-Token Prediction

掌握 sleep mode 的实现方法后，我们用同样的框架来分析如何设计 Multi-Token Prediction (MTP)。

### 9.1 MTP 是什么

标准自回归解码：每次 forward 预测一个 token，N 个 token 需要 N 次 forward。

MTP（Multi-Token Prediction）：通过额外的预测头，一次 forward 同时输出多个 draft tokens。如果 draft tokens 正确（用 Speculative Decoding 验证），可以跳过多次 forward，实现加速。

```
标准解码:  [t1,t2,t3] → forward → 预测 t4
MTP:      [t1,t2,t3] → forward → 预测 t4, t5, t6  (draft tokens)
          验证 t5,t6 是否正确 → 如果正确，一次性接受
```

### 9.2 按 Sleep Mode 的方法论分析 MTP

**第一步：先读现有实现**（等价于 sleep mode 中先研究 cuMem API）

vLLM 已有 Speculative Decoding 框架（`vllm/spec_decode/`），MTP 是其一种变体。先读：
- `vllm/spec_decode/spec_decode_worker.py` — 理解 draft + verify 流程
- `vllm/spec_decode/medusa_worker.py` — Medusa（多头 MTP 的经典实现）

**第二步：确定各层需要的改动**

参考 sleep mode 的五层改动，MTP 也需要逐层分析：

| 层 | Sleep Mode 改动 | MTP 改动 |
|---|---|---|
| 配置层 | `enable_sleep_mode`, `enable_cumem_allocator` | `num_speculative_tokens`, `speculative_model` (或 `use_mtp_heads`) |
| 模型层 | `load_model` 加 cumem pool 上下文 | 在 LlamaForCausalLM 中增加 N 个额外 LM head |
| ModelRunner | `post_kv_cache_wake_up` 处理 FP8 | `execute_model` 返回 draft tokens 而非单个 token |
| Scheduler | `pause_scheduler` / `set_pause_state` | 支持"草稿验证"调度（一次接受多个 token） |
| Executor | `collective_rpc("sleep")` | 无变化（TP 广播已有） |
| Engine | 协调 sleep/pause | 协调 draft generation + verification |
| Entrypoint | `LLM.sleep()` / `LLM.wake_up()` | 透明，用户无感知 |

**第三步：核心技术决策**（等价于 sleep mode 中决定用 cuMem 而非 cudaMalloc）

MTP 的核心决策：

1. **Draft head 的架构**：
   - Medusa 风格：共享 main model 的 last hidden state，N 个独立 MLP head
   - Eagle 风格：有轻量级 draft transformer，更准确但更复杂
   - 如何选择：先用 Medusa（简单），验证端到端流程正确后再考虑 Eagle

2. **KV Cache**：
   - Draft tokens 是否需要 KV Cache？Medusa 不需要（仅用 last hidden state），Eagle 需要
   - 如果需要，参考 `kv_cache_interface.py` 的 `KVCacheSpec` 扩展

3. **验证步骤**：
   - Draft tokens 生成后，用一次 forward 并行验证所有 draft tokens
   - 接受率取决于 draft model 质量
   - 实现位置：在 `SchedulerOutput` 中增加 draft tokens 字段，在 `execute_model` 后增加 verify 步骤

**第四步：逐步实现**（等价于 sleep mode 的 9 步实现手册）

```
Step 1：在 LlamaForCausalLM 中增加 N 个 draft head（参考 Medusa 论文）
        - 每个 draft head 是一个小 MLP + LM head
        - 实现 load_weights() 中的权重映射

Step 2：修改 GPUModelRunner.execute_model() 
        - 在标准 forward 后，调用 draft heads 生成候选 tokens
        - 返回 (main_token, draft_tokens) 而非单个 token

Step 3：修改 ModelRunnerOutput（vllm/v1/outputs.py）
        - 增加 draft_token_ids 字段

Step 4：修改 Scheduler（vllm/v1/core/scheduler.py）
        - 支持批量接受 tokens（当 draft tokens 全部通过验证时）
        - 实现 speculative verification 逻辑

Step 5：修改 EngineCore 的 step() 循环
        - 增加 draft generation → verification → accept 的流程

Step 6：修改配置和 CLI 参数

Step 7：端到端测试：确保 draft=0 时输出与标准解码完全相同
```

### 9.3 MTP 实现的关键参考代码

vLLM 已有完整的 Speculative Decoding 基础设施，MTP 可以直接复用：

```python
# vllm/spec_decode/medusa_worker.py - Medusa 的 draft worker 实现
# 这就是 MTP 的一种具体实现

class MedusaWorker(SpecDecodeWorkerBase):
    def get_spec_proposals(self, execute_model_req, ...):
        # 一次 forward 生成多个 draft tokens
        ...
```

关键洞察：**MTP 的核心挑战不在于模型代码（增加几个 head 很简单），而在于调度逻辑（如何高效地批量验证和接受多个 token）**。这与 sleep mode 的核心挑战（如何保证虚拟地址不变）是类似的——核心难点往往不在功能本身，而在于与系统其他部分的集成。

---

## 10. 核心文件速查表

| 功能 | 文件 | 关键符号 |
|-----|------|---------|
| CUDA VMM C 扩展 | `csrc/cumem_allocator.cpp` | `my_malloc`, `my_free`, `create_and_map`, `unmap_and_release`, `py_init_module` |
| Python 内存池 | `vllm/device_allocator/cumem.py` | `CuMemAllocator`, `use_memory_pool()`, `sleep()`, `wake_up()`, `AllocationData` |
| Worker 实现 | `vllm/v1/worker/gpu_worker.py` | `_maybe_get_memory_pool_context()`, `load_model()`, `initialize_from_config()`, `sleep()`, `wake_up()` |
| FP8 修复 | `vllm/v1/worker/gpu_model_runner.py` | `post_kv_cache_wake_up()`, `init_fp8_kv_scales()` |
| Executor 管理 | `vllm/v1/executor/abstract.py` | `Executor.sleep()`, `Executor.wake_up()`, `collective_rpc()` |
| 调度器暂停 | `vllm/v1/core/sched/interface.py` | `PauseState`, `set_pause_state()` |
| Engine 协调 | `vllm/v1/engine/core.py` | `sleep()`, `wake_up()`, `pause_scheduler()`, `resume_scheduler()` |
| 离线 API | `vllm/entrypoints/llm.py` | `LLM.sleep()`, `LLM.wake_up()` |
| 在线 API | `vllm/entrypoints/serve/sleep/api_router.py` | `/sleep`, `/wake_up`, `/is_sleeping` |
| 配置 | `vllm/config/model.py` | `ModelConfig.enable_sleep_mode`, `enable_cumem_allocator` |
| CLI 参数 | `vllm/engine/arg_utils.py` | `EngineArgs.enable_sleep_mode` |
| 平台检测 | `vllm/platforms/interface.py` | `is_sleep_mode_available()` |
| 编译缓存 | vllm config | `enable_sleep_mode` 作为编译 cache key 因子（PR #29696） |
| 端到端测试 | `tests/basic_correctness/test_cumem.py` | `test_basic_cumem`, `test_end_to_end`, `test_deep_sleep` |

### 关键 PR 编号速查

| PR | 内容 | 状态 |
|-----|------|------|
| #11743 | 首次实现（CUDA 版本） | ✅ |
| #12987 | HTTP API | ✅ |
| #13456 | PyTorch 2.6 强引用 bug | ✅ bugfix |
| #15500 | tags 参数（分步唤醒） | ✅ |
| #16889 | Level 2 buffers 恢复 | ✅ bugfix |
| #12695 | ROCm 支持（chunked allocation） | ✅ |
| #24731 | On-the-fly 量化内存泄漏 | ✅ bugfix |
| #28783 | FP8 KV Cache 乱码修复 | ✅ bugfix |
| #33195 | Level 0 睡眠 + wait 模式 | ✅ |
| #34528 | Engine pause/sleep 逻辑重构 | ✅ |
| #40812 | expandable_segments 自动禁用 | ✅ bugfix |
| #43020 | free 回调 stream-aware 同步 | ✅ bugfix |
