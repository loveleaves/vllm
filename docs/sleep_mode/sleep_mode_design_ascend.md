# vllm-ascend Sleep Mode 深度设计文档

> **目标读者**：不熟悉 vllm-ascend 的开发者。读完本文后，你应能：
> 1. 理解昇腾 NPU 上 sleep mode 的完整实现原理，以及它与 CUDA 版本的异同
> 2. 在旧版 vllm-ascend 上从零实现 sleep mode
> 3. 理解为什么在没有 VMM（虚拟内存重映射）接口的硬件（如昆仑芯）上 wake_up 必须 recapture graph

---

## 目录

1. [核心问题：昇腾有哪些硬件接口](#1-核心问题昇腾有哪些硬件接口)
2. [CUDA vs 昇腾 VMM 接口对比](#2-cuda-vs-昇腾-vmm-接口对比)
3. [vllm-ascend 整体架构与 patch 机制](#3-vllm-ascend-整体架构与-patch-机制)
4. [逐层源码精读](#4-逐层源码精读)
5. [PR 演进历史](#5-pr-演进历史)
6. [从零实现：分步骤手册](#6-从零实现分步骤手册)
7. [为什么昆仑芯需要 recapture graph（深度分析）](#7-为什么昆仑芯需要-recapture-graph深度分析)

---

## 1. 核心问题：昇腾有哪些硬件接口

理解 vllm-ascend sleep mode 的起点，是搞清楚昇腾 NPU 的内存管理接口（CANN ACL）与 NVIDIA CUDA 的对应关系。

### CUDA 需要的关键接口

vLLM 上游的 sleep mode（CUDA 版本）依赖以下 CUDA Driver API，核心是**虚拟地址和物理内存分离管理**：

```
cuMemAddressReserve   → 预留虚拟地址空间（不占物理显存）
cuMemCreate           → 分配物理内存页，得到 Handle
cuMemMap              → 将物理页映射到虚拟地址
cuMemSetAccess        → 设置访问权限
cuMemUnmap            → 解除映射（虚拟地址保留，物理内存还给系统）
cuMemRelease          → 释放物理内存 Handle
cuMemAddressFree      → 释放虚拟地址（只在 free 时调用）
```

### 昇腾 CANN 的等价接口

昇腾 NPU 在 CANN（Compute Architecture for Neural Networks）中提供了完全对应的 ACL（Ascend Computing Language）接口：

```
aclrtReserveMemAddress   ≡ cuMemAddressReserve   → 预留虚拟地址空间
aclrtMallocPhysical      ≡ cuMemCreate           → 分配物理 HBM 内存页
aclrtMapMem              ≡ cuMemMap              → 映射到虚拟地址
（无独立访问权限设置）     ≡ cuMemSetAccess        → CANN 在 MapMem 时隐含设置
aclrtUnmapMem            ≡ cuMemUnmap            → 解除映射
aclrtFreePhysical        ≡ cuMemRelease          → 释放物理内存
aclrtReleaseMemAddress   ≡ cuMemAddressFree      → 释放虚拟地址
```

**结论：昇腾有完整的 VMM 接口，可以实现与 CUDA 完全相同的 sleep mode 语义。** 这是 vllm-ascend 能够实现"全透明"sleep mode 的硬件基础。

同样，昇腾也有对应的内存拷贝接口：

```
memcpy(dst, dst_max, src, size, ACL_MEMCPY_DEVICE_TO_HOST)  ≡  cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost)
memcpy(dst, dst_max, src, size, ACL_MEMCPY_HOST_TO_DEVICE)  ≡  cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice)
```

注意昇腾的 `memcpy` 需要额外传入 `dst_max`（目标缓冲区大小上限），这是一个安全检查参数。

---

## 2. CUDA vs 昇腾 VMM 接口对比

### 2.1 接口差异汇总表

| 功能 | CUDA Driver API | Ascend ACL API | 差异说明 |
|------|----------------|----------------|---------|
| 预留虚拟地址 | `cuMemAddressReserve(&ptr, size, 0, 0, 0)` | `aclrtReserveMemAddress(&ptr, size, 0, nullptr, 0)` | 接口基本一致 |
| 查询对齐粒度 | `cuMemGetAllocationGranularity(&g, &prop, ...)` | `aclrtMemGetAllocationGranularity(&prop, ..., &g)` | **参数顺序不同** |
| 分配物理内存 | `cuMemCreate(&handle, size, &prop, 0)` | `aclrtMallocPhysical(&handle, size, &prop, 0)` | Handle 类型不同 |
| 映射到虚拟地址 | `cuMemMap(ptr, size, 0, handle, 0)` | `aclrtMapMem(ptr, size, 0, handle, 0)` | 接口一致 |
| 设置访问权限 | `cuMemSetAccess(ptr, size, &desc, 1)` | **无独立 API，MapMem 隐含设置** | CANN 简化了流程 |
| 解除映射 | `cuMemUnmap(ptr, size)` | `aclrtUnmapMem(ptr)` | **昇腾不需要传 size** |
| 释放物理内存 | `cuMemRelease(handle)` | `aclrtFreePhysical(handle)` | 接口一致 |
| 释放虚拟地址 | `cuMemAddressFree(ptr, size)` | `aclrtReleaseMemAddress(ptr)` | **昇腾不需要传 size** |
| 内存拷贝 | `cudaMemcpy(dst, src, size, dir)` | `memcpy(dst, dst_max, src, size, dir)` | 昇腾额外需要 dst_max |
| PyTorch 插件分配器 | `torch.cuda.memory.CUDAPluggableAllocator` | `torch.npu.memory.NPUPluggableAllocator` | NPU 版本 API 对应 |
| 内存池 | `torch.cuda.memory.MemPool` | `torch.npu.memory.MemPool` | NPU 版本 API 对应 |
| 使用内存池 | `torch.cuda.memory.use_mem_pool` | `torch.npu.memory.use_mem_pool` | NPU 版本 API 对应 |

### 2.2 Handle 类型差异

```cpp
// CUDA: Handle 是一个不透明整数类型
CUmemGenericAllocationHandle handle;

// Ascend: Handle 是一个指针类型（驱动内部对象）
aclrtDrvMemHandle handle;
```

这个差异影响 Python/C 之间传递数据的方式。CUDA 版本在 Python 层存储的 handle 是一个整数（`unsigned long long`），昇腾版本存储的也是指针地址（转换为 `unsigned long long`）。两者在 Python 层的处理方式相同。

### 2.3 属性结构体差异

```cpp
// CUDA
CUmemAllocationProp prop = {};
prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
prop.location.id = device;
prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;

// Ascend
aclrtPhysicalMemProp prop = {};
prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
prop.memAttr = ACL_HBM_MEM_HUGE;          // 昇腾特有：HBM 大页内存
prop.location.id = device;
prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
prop.reserve = 0;
```

`ACL_HBM_MEM_HUGE` 是昇腾特有的内存属性，表示使用 HBM（High Bandwidth Memory）大页，这是昇腾 NPU 的标准内存类型。

---

## 3. vllm-ascend 整体架构与 patch 机制

### 3.1 vllm-ascend 的设计哲学

vllm-ascend 不是一个独立的 vLLM fork，而是基于 vLLM 主仓的**插件式扩展**。它通过两种机制修改 vLLM 行为：

1. **注册机制**：通过 `vllm_ascend.platform.NPUPlatform` 注册为 vLLM 的计算平台
2. **Patch 机制**：在运行时 monkey-patch vLLM 的某些函数/类

```
vLLM 主仓代码（不修改）
    ↑
vllm-ascend patch 层（运行时替换）
    ↑
NPU 专用实现（worker/worker.py, device_allocator/camem.py 等）
```

### 3.2 Patch 的两个阶段

```
Phase 1：平台注册时（进程启动时）
├── vllm_ascend/platform.py → NPUPlatform.pre_register_and_update()
└── vllm_ascend/patch/platform/ 下的所有 patch
    ├── patch_camem_allocator.py  ← 关键：让 vLLM 认为 cumem 可用
    ├── patch_distributed.py
    ├── patch_mamba_config.py
    └── ...

Phase 2：Worker 启动时
└── vllm_ascend/patch/worker/ 下的 patch（不涉及 sleep mode）
```

### 3.3 Sleep Mode 关键 patch：`patch_camem_allocator.py`

vLLM 主仓在 `ModelConfig.__post_init__` 中有一个校验：

```python
# vllm/config/model.py
if self.enable_cumem_allocator and not is_cumem_allocator_available():
    raise ValueError("cumem allocator is not supported on current platform.")
```

`is_cumem_allocator_available()` 的实现是尝试 import `vllm.cumem_allocator`（CUDA 编译的 C 扩展）。在昇腾环境中，这个模块不存在，会抛出 `ModuleNotFoundError`，导致 `enable_sleep_mode=True` 时报错。

`patch_camem_allocator.py` 的作用就是 monkey-patch 掉这个检查：

```python
# vllm_ascend/patch/platform/patch_camem_allocator.py
def _patched_is_cumem_allocator_available() -> bool:
    # NPUPlatform 声明了 sleep mode 支持，vllm-ascend 使用自己的 CaMemAllocator
    # 在 ModelConfig 验证阶段不需要实际导入扩展
    return True

if hasattr(model_config_module, "is_cumem_allocator_available"):
    model_config_module.is_cumem_allocator_available = _patched_is_cumem_allocator_available
```

这使得 vLLM 的配置校验通过，但真正的内存分配由 vllm-ascend 的 `CaMemAllocator` 接管。

### 3.4 Sleep Mode 的调用链（端到端）

```
用户: llm.sleep(level=1)
  │
  ▼
vllm/entrypoints/llm.py: LLM.sleep()
  │  (上游代码，不修改)
  ▼
vllm/v1/engine/core.py: EngineCore.sleep()
  │  (上游代码：pause_scheduler → model_executor.sleep())
  ▼
vllm/v1/executor/abstract.py: Executor.sleep()
  │  (上游代码：collective_rpc("sleep"))
  ▼
vllm_ascend/worker/worker.py: Worker.sleep()    ← 昇腾专用 Worker
  │  (替换了 vllm/v1/worker/gpu_worker.py)
  ▼
vllm_ascend/device_allocator/camem.py: CaMemAllocator.sleep()  ← 昇腾专用分配器
  │
  ▼
vllm_ascend/vllm_ascend_C: python_unmap_and_release()  ← C 扩展
  │
  ▼
aclrtUnmapMem() + aclrtFreePhysical()   ← ACL Driver API
```

---

## 4. 逐层源码精读

### 4.1 C 扩展（`csrc/camem_allocator.cpp`）

与 CUDA 版本结构完全相同，但使用 CANN ACL API 替换 CUDA Driver API。

**`my_malloc`**（NPU Pluggable Allocator 调用钩子）：

```cpp
void* my_malloc(ssize_t size, int device, aclrtStream stream) {
    // 1. 查询对齐粒度（注意：参数顺序与 CUDA 不同）
    aclrtPhysicalMemProp prop = {
        .handleType = ACL_MEM_HANDLE_TYPE_NONE,
        .allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED,
        .memAttr = ACL_HBM_MEM_HUGE,      // HBM 大页，昇腾特有
        .location.id = device,
        .location.type = ACL_MEM_LOCATION_TYPE_DEVICE,
    };
    size_t granularity;
    aclrtMemGetAllocationGranularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM, &granularity);

    // 2. 对齐并预留虚拟地址
    size_t alignedSize = align_up(size, granularity);
    void *d_mem;
    aclrtReserveMemAddress(&d_mem, alignedSize, 0, nullptr, 0);

    // 3. 分配 Handle 结构体（在 CPU 堆上）
    aclrtDrvMemHandle* p_memHandle = (aclrtDrvMemHandle*)malloc(sizeof(aclrtDrvMemHandle));

    // 4. 调用 Python callback，记录 (device, size, d_mem, handle_ptr) 到字典
    PyGILState_STATE gstate = PyGILState_Ensure();
    // ... 构建 tuple，调用 g_python_malloc_callback ...
    PyGILState_Release(gstate);

    // 5. 分配物理内存并映射
    create_and_map(device, alignedSize, d_mem, p_memHandle);
    return (void*)d_mem;
}
```

**`create_and_map`**（wake_up 时调用）：

```cpp
void create_and_map(unsigned long long device, ssize_t size, void* d_mem,
                    aclrtDrvMemHandle* p_memHandle) {
    ensure_context(device);
    aclrtPhysicalMemProp prop = { /* 同上 */ };

    // 分配物理 HBM 内存页
    aclrtMallocPhysical(p_memHandle, size, &prop, 0);

    // 映射到虚拟地址（CANN 隐含设置了读写权限，无需 SetAccess）
    aclrtMapMem(d_mem, size, 0, *p_memHandle, 0);
}
```

**`unmap_and_release`**（sleep 时调用）：

```cpp
void unmap_and_release(unsigned long long device, ssize_t size,
                       void* d_mem, aclrtDrvMemHandle* p_memHandle) {
    ensure_context(device);
    // 注意：昇腾的 UnmapMem 不需要传 size（与 CUDA 不同）
    aclrtUnmapMem(d_mem);
    aclrtFreePhysical(*p_memHandle);
}
```

**`my_free`**（NPU GC 张量时调用）：

```cpp
void my_free(void* ptr, ssize_t size, int device, aclrtStream stream) {
    // 调用 Python callback 查询 handle
    // ...
    unmap_and_release(device, size, d_mem, p_memHandle);

    // 释放虚拟地址（与 CUDA 不同：不需要传 size）
    aclrtReleaseMemAddress(d_mem);
    free(p_memHandle);
}
```

**模块初始化**：

```cpp
// 注意：模块名是 vllm_ascend_C，与 CUDA 版本的 cumem_allocator 不同
PyMODINIT_FUNC PyInit_vllm_ascend_C(void) {
    PyObject* module = PyModule_Create(&camem_allocator_module);
    return module;
}
```

Python 层 import 时：
```python
from vllm_ascend.vllm_ascend_C import init_module, python_create_and_map, python_unmap_and_release
```

### 4.2 CaMemAllocator（`vllm_ascend/device_allocator/camem.py`）

与 vLLM 主仓的 `CuMemAllocator` 结构基本相同，关键差异：

**内存拷贝（sleep 时）**：
```python
def sleep(self, offload_tags=("default",)):
    for ptr, data in self.pointer_to_data.items():
        handle = data.handle
        if data.tag in offload_tags:
            size_in_bytes = handle[1]
            cpu_backup_tensor = torch.empty(size_in_bytes, dtype=torch.uint8,
                                            device="cpu", pin_memory=True)
            cpu_ptr = cpu_backup_tensor.data_ptr()

            # 昇腾的 memcpy 需要 dst_max（目标缓冲区上限，安全校验）
            ACL_MEMCPY_DEVICE_TO_HOST = 2
            dest_max = cpu_ptr + size_in_bytes * 2    # 留足余量
            memcpy(cpu_ptr, dest_max, ptr, size_in_bytes, ACL_MEMCPY_DEVICE_TO_HOST)
            data.cpu_backup_tensor = cpu_backup_tensor

        unmap_and_release(handle)

    gc.collect()
    torch.npu.empty_cache()    # ← torch.npu 而非 torch.cuda
```

**内存拷贝（wake_up 时）**：
```python
def wake_up(self, tags=None):
    for ptr, data in self.pointer_to_data.items():
        if tags is None or data.tag in tags:
            create_and_map(data.handle)
            if data.cpu_backup_tensor is not None:
                size_in_bytes = data.cpu_backup_tensor.numel()
                cpu_ptr = data.cpu_backup_tensor.data_ptr()
                ACL_MEMCPY_HOST_TO_DEVICE = 1
                dest_max = ptr + size_in_bytes * 2    # 昇腾特有的安全参数
                memcpy(ptr, dest_max, cpu_ptr, size_in_bytes, ACL_MEMCPY_HOST_TO_DEVICE)
                data.cpu_backup_tensor = None
```

**内存池上下文**：
```python
@contextmanager
def use_memory_pool(self, tag=None):
    # 注意：昇腾版本直接 assert expandable_segments 未开启
    # （而 CUDA 版本是检测到了就临时禁用，更优雅）
    conf = os.environ.get("PYTORCH_NPU_ALLOC_CONF", "")
    assert "expandable_segments:True" not in conf, \
        "Expandable segments are not compatible with memory pool."

    old_tag = self.current_tag
    self.current_tag = tag
    # torch.npu.memory.NPUPluggableAllocator 对应 CUDA 版本的 CUDAPluggableAllocator
    with use_memory_pool_with_allocator(
        self.python_malloc_callback, self.python_free_callback
    ) as data:
        self.allocator_and_pools[tag] = data   # 强引用防止 GC
        yield
        self.current_tag = old_tag
```

**注意：昇腾版本缺少的优化**（与 CUDA 最新版本相比）：
1. 缺少 `torch.cuda.synchronize()` 在 free callback 中（PR #43020 才加入 CUDA 版本）
2. `use_memory_pool` 退出时没有处理 on-the-fly 量化的内存泄漏（CUDA 版本 PR #24731 添加了）
3. expandable_segments 处理是 assert 而不是自动禁用（CUDA 版本 PR #40812 优化了）

### 4.3 Worker（`vllm_ascend/worker/worker.py`）

**`load_model`**：
```python
def load_model(self) -> None:
    if self.vllm_config.model_config.enable_sleep_mode:
        allocator = CaMemAllocator.get_instance()
        assert allocator.get_current_usage() == 0, \
            "Sleep mode can only be used for one instance per process."
        context = allocator.use_memory_pool(tag="weights")
    else:
        context = nullcontext()
    with context, set_current_vllm_config(self.vllm_config):
        self.model_runner.load_model()
```

**`initialize_from_config`**（KV Cache 进内存池）：
```python
def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
    ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)
    if self.vllm_config.model_config.enable_sleep_mode:
        allocator = CaMemAllocator.get_instance()
        context = allocator.use_memory_pool(tag="kv_cache")
    else:
        context = nullcontext()
    with context:
        self.model_runner.initialize_kv_cache(kv_cache_config)
```

**`sleep`**：
```python
def sleep(self, level: int = 1) -> None:
    free_bytes_before_sleep = torch.npu.mem_get_info()[0]    # torch.npu 而非 torch.cuda

    if level == 2:
        model = self.model_runner.model
        self._sleep_saved_buffers = {
            name: buffer.cpu().clone()
            for name, buffer in model.named_buffers()
        }

    allocator = CaMemAllocator.get_instance()
    allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())

    freed_bytes = torch.npu.mem_get_info()[0] - free_bytes_before_sleep
    assert freed_bytes >= 0, "Memory usage increased after sleeping."
```

**`wake_up`（含昇腾特有的 MoE 权重转置逻辑）**：
```python
def wake_up(self, tags: list[str] | None = None) -> None:
    # 检查是否开启了 NZ 格式（昇腾专有的权重布局优化）
    nz_mode = get_ascend_config().weight_nz_mode
    if nz_mode:
        raise ValueError(
            "FRACTAL_NZ mode is enabled. This may cause model parameter precision issues "
            "in the RL scenarios. Please set weight_nz_mode=0 via --additional-config."
        )

    allocator = CaMemAllocator.get_instance()
    allocator.wake_up(tags=tags)

    # 昇腾特有：MoE 模型的权重转置
    # 背景：昇腾的 MoE 算子对权重布局有特殊要求，初始化时会做 transpose
    # wake_up 后权重已从 CPU 恢复到原始布局，需要重新 transpose
    hidden_size = self.vllm_config.model_config.hf_text_config.hidden_size
    model = self.model_runner.model
    if self.vllm_config.quant_config is None and (tags is None or "weights" in tags):
        for name, param in model.named_parameters():
            # Qwen/LLaMA MoE 的 gate_up_proj（w13）和 down_proj（w2）
            if "w2_weight" in name and param.shape[2] == hidden_size:
                parts = name.split(".")
                param_name = parts[-1]
                parent_module = model.get_submodule(".".join(parts[:-1]))
                w2_data = torch.nn.Parameter(param.transpose(1, 2), requires_grad=False)
                setattr(parent_module, param_name, w2_data)
            elif "w13_weight" in name and param.shape[1] == hidden_size:
                parts = name.split(".")
                param_name = parts[-1]
                parent_module = model.get_submodule(".".join(parts[:-1]))
                w13_data = torch.nn.Parameter(param.transpose(1, 2), requires_grad=False)
                setattr(parent_module, param_name, w13_data)

    # 恢复 level 2 sleep 保存的 buffers
    if len(self._sleep_saved_buffers):
        for name, buffer in model.named_buffers():
            if name in self._sleep_saved_buffers:
                buffer.data.copy_(self._sleep_saved_buffers[name].data)
        self._sleep_saved_buffers = {}
```

> **MoE 权重转置的必要性**（PR #4626 解释）：
> 昇腾的 MoE 算子（ATB MoE kernel）要求 w2/w13 权重以转置形式存储（`[out_features, intermediate, hidden]` 转为 `[out_features, hidden, intermediate]`）。这个转置在模型初始化时由 vllm-ascend 的模型加载器自动完成。但 sleep 时权重被原样备份到 CPU（字节级拷贝），wake_up 后恢复的是**已转置的形式**，内存地址不变，张量 shape 和 stride 也不变。这部分本不需要操作。
>
> 然而，PR #4626 发现了一个问题：在 RL 场景中，训练框架通过 `collective_rpc("reload_weights")` 将**新训练的、未转置的**权重写入相同的内存地址。此时如果不重新转置，MoE 算子会拿到格式错误的权重，产生错误输出。因此，`wake_up` 时必须在恢复 CPU 备份（level 1）或接收新权重（level 2 + reload）之后，重新执行一次 transpose 操作。

### 4.4 平台层（`vllm_ascend/platform.py`）

```python
class NPUPlatform(Platform):
    def is_sleep_mode_available(self) -> bool:
        return True    # 直接返回 True，昇腾支持 sleep mode
```

这与上游 `Platform.is_sleep_mode_available()` 的实现不同：

```python
# vllm/platforms/interface.py（上游）
def is_sleep_mode_available(self) -> bool:
    return self._enum in (PlatformEnum.CUDA, PlatformEnum.ROCM)
    # KunLun 的 _enum 既不是 CUDA 也不是 ROCM，所以返回 False！
```

---

## 5. PR 演进历史

| PR | 时间 | 内容 | 核心问题 |
|----|------|------|---------|
| `#416` / `#513` | 2025.04 | **首次实现**：C 扩展 + CaMemAllocator + Worker sleep/wake_up | 基础框架 |
| `#1084` | 2025.06 | V1 engine worker 支持（从 V0 迁移到 V1 架构） | V1 Worker 与 V0 接口不同 |
| `#1376` | 2025.06 | Level 2 sleep 的 buffers 恢复（named_buffers restore） | 直接对应上游 PR #16889 |
| `#2152` | 2025.08 | external_launcher 场景的 e2e 测试 | 多进程场景验证 |
| `#4626` | 2025.12 | **MoE 权重转置移到 wake_up**（RL 场景修复） | 训练框架 reload_weights 后需重新转置 |
| `#4019` / `#4023` | 2025.12 | Level 2 e2e 测试修复 | 测试覆盖补全 |
| `#7709` | 2026.03 | 在 sleep 函数中添加 `gc.collect` 和 `torch.npu.empty_cache` | 对应上游 PR #15248 |

**关键观察**：vllm-ascend 的 sleep mode 演进基本与上游 vLLM 的演进同步，但有两个昇腾特有的问题：
1. MoE 权重转置（`#4626`）——这是昇腾 ATB kernel 特有的权重布局要求
2. NZ 格式检测（`wake_up` 中的 `if nz_mode: raise`）——FRACTAL_NZ 是昇腾专有的内存格式优化，与 sleep mode 不兼容

---

## 6. 从零实现：分步骤手册

以下是在一个旧版 vllm-ascend（没有 sleep mode 支持的版本）上从零实现的完整步骤。

### 前置条件检查

```bash
# 确认 CANN 版本支持 VMM 接口（需要 CANN >= 8.0）
python3 -c "import acl; print(dir(acl.rt))" | grep -i "ReserveMemAddress\|MallocPhysical"
# 如果能看到这些函数，说明 VMM 接口可用

# 确认 torch_npu 支持 NPUPluggableAllocator
python3 -c "import torch_npu; print(dir(torch.npu.memory))" | grep -i "Pluggable\|MemPool\|use_mem_pool"
```

### Step 1：实现 C 扩展（`csrc/camem_allocator.cpp`）

参考 `vllm_ascend/csrc/camem_allocator.cpp` 实现，关键点：
- 包含 `acl/acl.h`（而非 `cuda.h`）
- 所有 `cuMem*` 调用改为对应的 `aclrt*` 调用
- `my_free` 的 `aclrtUnmapMem` 不需要传 size 参数
- 模块名 `PyInit_vllm_ascend_C`（不是 `PyInit_cumem_allocator`）

**关键验证**（Step 1 完成后）：
```python
# 验证 C 扩展加载
from vllm_ascend.vllm_ascend_C import init_module, python_create_and_map, python_unmap_and_release
print("C extension OK")
```

### Step 2：修改 CMakeLists.txt / setup.py

在构建脚本中将 `csrc/camem_allocator.cpp` 编译为 Python 扩展 `vllm_ascend_C`：

```cmake
# CMakeLists.txt 中添加
if(ASCEND_FOUND)
    Python_add_library(vllm_ascend_C MODULE
        csrc/camem_allocator.cpp
    )
    target_link_libraries(vllm_ascend_C PRIVATE ascendcl)
    install(TARGETS vllm_ascend_C
            LIBRARY DESTINATION ${Python_SITEARCH}/vllm_ascend)
endif()
```

### Step 3：实现 `CaMemAllocator`（`vllm_ascend/device_allocator/camem.py`）

以上游 `vllm/device_allocator/cumem.py` 为模板，替换：
- `torch.cuda.*` → `torch.npu.*`
- `CuMemAllocator` → `CaMemAllocator`
- `libcudart.cudaMemcpy(dst, src, size)` → `memcpy(dst, dst_max, src, size, direction)` 其中 `dst_max = dst + size * 2`
- `CudaRTLibrary()` 和 `libcudart` 不再需要（直接用 `from acl.rt import memcpy`）

**验证 Step 3**（不依赖 vllm）：
```python
import torch
import torch_npu
from vllm_ascend.device_allocator.camem import CaMemAllocator

allocator = CaMemAllocator.get_instance()
with allocator.use_memory_pool(tag="test"):
    x = torch.ones(1024, 1024, device="npu:0")

print(f"data_ptr before sleep: {hex(x.data_ptr())}")
ptr_before = x.data_ptr()

allocator.sleep(offload_tags=("test",))
free_after_sleep = torch.npu.mem_get_info()[0]
print(f"Free NPU memory increased: {free_after_sleep}")

allocator.wake_up()
print(f"data_ptr after wake_up: {hex(x.data_ptr())}")  # 应该相同
assert x.data_ptr() == ptr_before, "Virtual address changed!"
assert x.mean().item() == 1.0, "Data not restored!"
print("CaMemAllocator basic test PASSED")
```

### Step 4：添加 patch（`vllm_ascend/patch/platform/patch_camem_allocator.py`）

```python
# patch_camem_allocator.py
import vllm.config.model as model_config_module

def _patched_is_cumem_allocator_available() -> bool:
    return True  # 昇腾用自己的 CaMemAllocator

if hasattr(model_config_module, "is_cumem_allocator_available"):
    model_config_module.is_cumem_allocator_available = _patched_is_cumem_allocator_available
```

在 `vllm_ascend/patch/platform/__init__.py` 中注册：
```python
import vllm_ascend.patch.platform.patch_camem_allocator  # noqa
```

### Step 5：修改 Platform（`vllm_ascend/platform.py`）

```python
class NPUPlatform(Platform):
    def is_sleep_mode_available(self) -> bool:
        return True  # 昇腾支持 VMM，sleep mode 可用
```

### Step 6：修改 Worker（`vllm_ascend/worker/worker.py`）

在现有 Worker 类的 `__init__` 中初始化：
```python
if vllm_config.model_config and vllm_config.model_config.enable_sleep_mode:
    self._sleep_saved_buffers: dict[str, torch.Tensor] = {}
```

修改 `load_model`：
```python
def load_model(self) -> None:
    if self.vllm_config.model_config.enable_sleep_mode:
        allocator = CaMemAllocator.get_instance()
        assert allocator.get_current_usage() == 0
        context = allocator.use_memory_pool(tag="weights")
    else:
        context = nullcontext()
    with context, set_current_vllm_config(self.vllm_config):
        self.model_runner.load_model()
```

修改 `initialize_from_config`（或 `initialize_cache`）：
```python
def initialize_from_config(self, kv_cache_config):
    if self.vllm_config.model_config.enable_sleep_mode:
        context = CaMemAllocator.get_instance().use_memory_pool(tag="kv_cache")
    else:
        context = nullcontext()
    with context:
        self.model_runner.initialize_kv_cache(kv_cache_config)
```

添加 `sleep` 和 `wake_up` 方法（参见第 4.3 节的完整实现）。

### Step 7：处理 NZ 格式兼容性

检查 vllm-ascend 的 model loader 是否有 NZ 格式相关逻辑：

```bash
grep -rn "NZ\|nz_mode\|FRACTAL_NZ\|weight_nz" vllm_ascend/ | grep -v "__pycache__"
```

如果有，在 `wake_up` 中添加检测：
```python
def wake_up(self, tags=None):
    nz_mode = get_ascend_config().weight_nz_mode
    if nz_mode:
        raise ValueError("weight_nz_mode=1 is incompatible with sleep mode. Use --additional-config '{\"weight_nz_mode\": 0}'")
    # ... 其余逻辑
```

### Step 8：处理 MoE 权重转置（如果使用 MoE 模型）

如果你的模型是 Qwen-MoE、DeepSeek、Mixtral 等 MoE 架构，需要在 `wake_up` 后重新做权重转置（参见第 4.3 节的详细实现）。

### Step 9：测试

参考 `tests/e2e/singlecard/test_camem.py`：

```python
def test_sleep_level1():
    """level 1 sleep/wake_up 正确性"""
    free, total = torch.npu.mem_get_info()
    baseline = total - free
    llm = LLM("Qwen/Qwen3-0.6B", enable_sleep_mode=True)
    prompt = "Hello"
    params = SamplingParams(temperature=0, max_tokens=5)
    out1 = llm.generate(prompt, params)

    llm.sleep(level=1)
    free_after, _ = torch.npu.mem_get_info()
    assert (total - free_after - baseline) < 1 * GiB_bytes  # 释放了大量内存

    llm.wake_up()
    out2 = llm.generate(prompt, params)
    assert out1[0].outputs[0].text == out2[0].outputs[0].text  # 输出相同

def test_sleep_level2_with_reload():
    """level 2 + reload_weights 正确性"""
    llm = LLM("Qwen/Qwen3-0.6B", enable_sleep_mode=True)
    prompt = "Hello"
    params = SamplingParams(temperature=0, max_tokens=5)
    out1 = llm.generate(prompt, params)

    llm.sleep(level=2)
    llm.wake_up(tags=["weights"])
    llm.collective_rpc("reload_weights")  # 重新加载相同权重
    llm.wake_up(tags=["kv_cache"])
    out2 = llm.generate(prompt, params)
    assert out1[0].outputs[0].text == out2[0].outputs[0].text
```

---

## 7. 为什么昆仑芯需要 recapture graph（深度分析）

这是本文最核心的技术分析。理解这个问题，需要先理解 CUDA Graph / NPU Graph 与内存地址的关系。

### 7.1 Graph Replay 的本质：地址的承诺

CUDA Graph（以及昇腾的 NPU Graph）的工作原理是：

**Capture 阶段**：将一系列 GPU 命令（kernel 调用、内存操作）录制为一个"图"。录制时，GPU 命令中的所有地址（输入 tensor、输出 tensor、权重 tensor、KV cache）被**硬编码**进图的命令流中。

```
Graph 命令流（示意）:
    GEMM(input=0xA000, weight=0xB000, output=0xC000)
    LayerNorm(input=0xC000, gamma=0xD000, output=0xE000)
    ...
```

**Replay 阶段**：把录制好的命令流重新提交给 GPU 执行。GPU 直接使用命令流中的硬编码地址，**不会重新解析 Python 对象**。

这就是 Graph 加速的来源：省掉了 Python-GPU 的接口开销（kernel 启动、参数解析）。

**关键前提**：Replay 要求所有硬编码地址的**数据有效**。具体说：

1. 这些地址指向的虚拟内存范围必须被映射到有效的物理内存
2. 对于权重地址，物理内存中的数据必须是正确的权重数据

### 7.2 为什么 CUDA/昇腾 sleep mode 不需要 recapture

CUDA 和昇腾都有 VMM 接口，实现的是"**虚拟地址固定，物理内存可替换**"的内存模型：

```
Sleep 前：
  虚拟地址 0xB000 ──映射──▶ 物理页 P1（存储权重数据）

Sleep 时（unmap + free）：
  虚拟地址 0xB000 ──映射──▶ （无效，但虚拟地址保留）
  物理页 P1 → 释放，数据备份到 CPU RAM

Wake up 时（alloc + remap）：
  虚拟地址 0xB000 ──映射──▶ 物理页 P2（新分配）
  从 CPU 恢复数据到 P2
```

Graph 命令流中硬编码的是 `0xB000`（虚拟地址），而不是物理地址。Wake up 后，`0xB000` 再次被映射到物理内存，Graph 可以直接 replay，不需要重新 capture。

**这就是"透明"的含义**：从 Graph 的角度看，什么都没发生过。

### 7.3 昆仑芯的问题：没有 VMM 接口

昆仑芯的 XPU 驱动（XTCL）没有提供对应的虚拟内存分离管理接口。它的内存分配是"传统"模式：

```
xpuMalloc(size) → 返回一个指针，物理内存已绑定，无法分离
xpuFree(ptr)    → 释放物理内存 + 虚拟地址（两步合一）
```

当用户尝试实现 sleep mode 时，没有 VMM 接口意味着必须用**高层 Python 操作**来替代：

```python
# 没有 VMM 接口时，只能这样做 sleep：
def sleep_without_vmm(model):
    # 将权重数据拷贝到 CPU
    cpu_backup = {name: param.cpu().clone() for name, param in model.named_parameters()}
    # 释放 NPU 上的权重内存（此时指针失效！）
    for param in model.parameters():
        param.data = torch.empty(0, device="cpu")  # 让 data 指向空 CPU tensor
    # 强制释放 XPU 内存
    torch.xpu.empty_cache()
    return cpu_backup

# wake_up 时必须重新分配内存：
def wake_up_without_vmm(model, cpu_backup):
    for name, param in model.named_parameters():
        # 重新分配 XPU 内存（得到新的指针！）
        param.data = cpu_backup[name].to("xpu")
        # 新的指针与 Graph 录制时的指针不同！
```

**这就是问题的根源**：

```
Graph 录制时（地址A）：  GEMM(weight=0xB000, ...)
Wake up 后（地址B）：   weight 重新分配到 0xF000
                       但 Graph 里面还是 0xB000 → 野指针！
```

所以必须**重新 capture graph**，让 Graph 录制新的地址 `0xF000`。

### 7.4 昆仑芯实现的两种可选方案

**方案 A：每次 wake_up 后 recapture graph（你已尝试的方案）**

```
sleep()
  → 备份权重到 CPU，param.data = empty
  → 释放 XPU 内存

wake_up()
  → 从 CPU 恢复权重，param.data = tensor.to("xpu")
  → 重新 capture graph（耗时！）
  → 才能继续推理
```

**缺点**：recapture 耗时数十秒，接近重启进程的代价。如果是反复 sleep/wake 的场景，每次都付出这个代价是不可接受的。

**方案 B：强制 eager 模式（禁用 graph）**

```python
# 在 sleep 模式下，不使用 graph capture
vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
vllm_config.model_config.enforce_eager = True
```

**缺点**：放弃了 graph 加速，推理速度下降（通常 15-30%）。

**方案 C：实现"持久化 graph"（理论上可行，实现复杂）**

核心思想：Sleep 时不释放 XPU 内存，而是将权重数据置零（清空内容但保留地址）；Wake up 时在原地写回数据。

```python
def sleep_level1_no_vmm(model):
    cpu_backup = {}
    for name, param in model.named_parameters():
        # 备份数据到 CPU
        cpu_backup[name] = param.data.cpu().clone()
        # 将参数归零（XPU 内存地址不变，但内容清空）
        param.data.zero_()
    # 零张量可能允许 XPU 系统回收物理内存（取决于驱动）
    torch.xpu.empty_cache()
    return cpu_backup

def wake_up_no_vmm(model, cpu_backup):
    for name, param in model.named_parameters():
        # 在原地写回（copy_ 不改变地址）
        param.data.copy_(cpu_backup[name].to("xpu"))
    # 无需 recapture graph！地址没变！
```

**问题**：`zero_()` 后系统是否真的会释放物理内存，取决于驱动实现。大多数硬件驱动不会因为内容归零而释放物理内存，这个方案只有当操作系统支持"零页折叠"（zero-page folding）时才有效。

**方案 D：XPU 驱动添加 VMM 支持（根本解决方案）**

向昆仑芯提需求，在 XTCL 驱动中添加类似 `cuMemAddressReserve`/`cuMemCreate`/`cuMemMap` 的接口族。一旦有了这些接口，就可以完全复制 vllm-ascend 的 C 扩展实现，做到真正透明的 sleep mode。

### 7.5 为什么昇腾没有这个问题的总结

| 方面 | 昇腾（CANN ACL） | 昆仑芯（XTCL） |
|------|-----------------|---------------|
| VMM 接口 | ✅ `aclrtReserveMemAddress` 等完整套件 | ❌ 无等价接口 |
| 虚拟地址稳定性 | ✅ sleep/wake 后地址不变 | ❌ wake 后地址改变 |
| Graph recapture | ❌ 不需要 | ✅ 必须 |
| 实现难度 | 中（移植 C 扩展） | 高（需要妥协方案） |
| 性能 | ✅ 与 CUDA 相同 | ⚠️ 降级或额外开销 |

### 7.6 最优实用建议（昆仑芯场景）

如果昆仑芯不支持 VMM，且要求 RLHF sleep 场景下必须保留 graph 加速，最可行的短期方案是：

**方案 B+**：Sleep 时保留 graph（不 recapture），但释放大部分内存。

```python
def sleep_kunlun(model, level=1):
    """
    妥协方案：
    - Graph 相关的内存（通常较小，约 2-5GB）：保留
    - 模型权重（最大头，通常 10-50GB）：offload 到 CPU
    - KV Cache：释放
    """
    if level >= 1:
        # 将权重 offload 到 CPU（in-place，保留 XPU 地址）
        # 原理：先把数据拷贝出去，再 clone 一个空 tensor 到同地址
        for param in model.parameters():
            cpu_copy = param.data.cpu()
            # 这里的问题：XPU tensor 无法 in-place 替换
            # 需要驱动支持 "pin virtual address" 语义
            ...
```

**结论**：没有 VMM 接口，就没有真正透明的 sleep mode。在 RLHF 场景中，推荐以下降级顺序：

1. **最优**：推动驱动添加 VMM 接口
2. **次优**：sleep 时强制 eager + recapture（accept 数十秒 wake_up 开销）
3. **备选**：完全不使用 graph（enforce_eager=True），牺牲推理性能