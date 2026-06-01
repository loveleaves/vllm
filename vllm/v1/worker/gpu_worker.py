"""A GPU worker class."""
import gc
import os
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.distributed

import vllm.envs as envs
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.platforms import current_platform
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType, get_dtype_size
from vllm.v1.core.scheduler import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.scheduler import SchedulerOutput
from contextlib import AbstractContextManager, nullcontext


class Worker:

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
    ):

        # TODO: use WorkerBase.__init__(self, vllm_config=vllm_config)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config

        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        torch_profiler_trace_dir)
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir, use_gzip=True))
        else:
            self.profiler = None
        
        # 用于 level 2 sleep 前保存模型 buffers（buffers 不在 cumem pool 中）
        self._sleep_saved_buffers: dict[str, torch.Tensor] = {}

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

    def initialize(self):
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            gc.collect()
            torch.cuda.empty_cache()
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.parallel_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Construct the model runner
        self.model_runner = GPUModelRunner(self.vllm_config, self.device)

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        with (
            self._maybe_get_memory_pool_context(tag="weights")
        ):
            self.model_runner.load_model()

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

    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        _, total_gpu_memory = torch.cuda.mem_get_info()
        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()
        torch.cuda.synchronize()

        free_gpu_memory, _ = torch.cuda.mem_get_info()
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        assert self.init_gpu_memory > free_gpu_memory, (
            "Error in memory profiling. "
            f"Initial free memory {self.init_gpu_memory}, current free memory"
            f" {free_gpu_memory}. This happens when the GPU memory was "
            "not properly cleaned up before initializing the vLLM instance.")

        # Get the peak memory allocation recorded by torch
        peak_memory = torch.cuda.memory_stats()["allocated_bytes.all.peak"]

        # Check for any memory left around that may have been allocated on the
        # gpu outside of `torch`. NCCL operations, for example, can use a few
        # GB during a forward pass
        torch.cuda.empty_cache()
        torch_allocated_bytes = torch.cuda.memory_stats(
        )["allocated_bytes.all.current"]
        total_allocated_bytes = torch.cuda.mem_get_info(
        )[1] - torch.cuda.mem_get_info()[0]
        non_torch_allocations = total_allocated_bytes - torch_allocated_bytes
        if non_torch_allocations > 0:
            peak_memory += non_torch_allocations
        available_kv_cache_memory = (
            total_gpu_memory * self.cache_config.gpu_memory_utilization -
            peak_memory)

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        cache_block_size = _get_cache_block_size(self.cache_config,
                                                 self.model_config,
                                                 self.parallel_config)
        num_gpu_blocks = int(available_kv_cache_memory // cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        return num_gpu_blocks, 0

    def initialize_cache(self, num_gpu_blocks: int) -> None:
        """Allocate GPU and CPU KV cache with the specified number of blocks."""
        if num_gpu_blocks <= 0:
            raise ValueError("No available memory for the cache blocks. "
                             "Try increasing `gpu_memory_utilization` when "
                             "initializing the engine.")

        max_seq_len = self.cache_config.block_size * num_gpu_blocks
        max_model_len = self.model_config.max_model_len
        if max_model_len > max_seq_len:
            raise ValueError(
                f"The model's max seq len ({max_model_len}) "
                "is larger than the maximum number of tokens that can be "
                f"stored in KV cache ({max_seq_len}). Try increasing "
                "`gpu_memory_utilization` or decreasing `max_model_len` when "
                "initializing the engine.")

        with self._maybe_get_memory_pool_context(tag="kv_cache"):
            self.model_runner.initialize_kv_cache(num_gpu_blocks)

    def reload_weights(self) -> None:
        """level 2 wake_up 后重新加载权重（RLHF 场景：新权重已写入虚拟地址）。"""
        self.model_runner.reload_weights()

    def compile_or_warm_up_model(self) -> None:
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model()
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput:
        output = self.model_runner.execute_model(scheduler_output)
        return output if self.rank == 0 else None

    def profile(self, is_start: bool = True):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return


def init_worker_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_distributed_environment(parallel_config.world_size, rank,
                                 distributed_init_method, local_rank)

    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:  # noqa: SIM102
        if not current_platform.has_device_capability(80):
            capability = current_platform.get_device_capability()
            gpu_name = current_platform.get_device_name()

            if capability is None:
                compute_str = "does not have a compute capability"
            else:
                version_str = capability.as_version_str()
                compute_str = f"has compute capability {version_str}"

            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU {compute_str}. "
                "You can use float16 instead by explicitly setting the"
                "`dtype` flag in CLI, for example: --dtype=half.")


def _get_cache_block_size(
    cache_config: CacheConfig,
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
) -> int:
    head_size = model_config.get_head_size()
    num_heads = model_config.get_num_kv_heads(parallel_config)
    num_attention_layers = model_config.get_num_layers_by_block_type(
        parallel_config, LayerBlockType.attention)

    key_cache_block = cache_config.block_size * num_heads * head_size
    value_cache_block = key_cache_block
    total = num_attention_layers * (key_cache_block + value_cache_block)
    if cache_config.cache_dtype == "auto":
        dtype = model_config.dtype
    else:
        dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
    dtype_size = get_dtype_size(dtype)
    return dtype_size * total
