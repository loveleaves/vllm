import torch
from vllm.device_allocator.cumem import CuMemAllocator

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