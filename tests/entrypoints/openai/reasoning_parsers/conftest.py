"""Inject stubs for compiled C extensions before any vllm import.

The reasoning-parser unit tests only exercise pure-Python serving code.
They don't need vllm._C (CUDA kernels), but vllm/platforms/cuda.py
unconditionally imports it.  Pre-populating sys.modules with a MagicMock
lets the test suite run without a compiled vllm._C.so.
"""
import sys
import types
from unittest.mock import MagicMock

for _mod in ("vllm._C", "vllm._moe_C", "vllm.cumem_allocator"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
