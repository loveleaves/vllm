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