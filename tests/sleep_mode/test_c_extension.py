# test_c_extension.py
from vllm.cumem_allocator import init_module, python_create_and_map, python_unmap_and_release
print("✓ C extension loaded successfully")

# 验证函数签名
import inspect
print(f"  init_module: {init_module.__doc__}")
print(f"  python_create_and_map: {python_create_and_map.__doc__}")
print(f"  python_unmap_and_release: {python_unmap_and_release.__doc__}")