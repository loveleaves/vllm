# CuMemAllocator：管理 cumem 内存池的 Python 层单例
import dataclasses
import gc
import os
from contextlib import contextmanager
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.utils import is_pin_memory_available
from vllm.utils import find_loaded_library

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
                # MemPool.snapshot() 在 PyTorch 2.6+ 才可用；2.5.x 的 pluggable
                # allocator 不缓存 free block，无需手动清理，直接跳过。
                if hasattr(data[0], "snapshot"):
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