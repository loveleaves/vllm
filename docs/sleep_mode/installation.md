# vLLM 0.6.6 安装与开发指南

## 环境信息

| 组件 | 版本 |
|------|------|
| Python | 3.12.13（uv 管理） |
| PyTorch | 2.5.1+cu121 |
| CUDA Toolkit | 12.4（系统）|
| xformers | 0.0.28.post3 |
| transformers | 4.57.6（需 `<5.0`）|
| GPU | RTX 3060 Ti 8GB |
| 系统 | WSL2 / Ubuntu |

---

## 一、首次安装

### 1. 创建虚拟环境

必须使用 **uv 管理的 Python**（含完整开发头文件），不能用系统 Python（缺少 cmake 所需的 `Python.h`）。

```bash
# 安装 uv 管理的 Python 3.12
uv python install 3.12

# 用 uv Python 创建 venv（--seed 预置 pip）
uv venv .venv --python ~/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12 --seed --clear
```

### 2. 安装构建工具

```bash
uv pip install --python .venv/bin/python \
  "cmake>=3.26" ninja packaging "setuptools>=61" "setuptools-scm>=8" wheel jinja2
```

### 3. 安装 PyTorch（CUDA 12.1 构建，兼容系统 CUDA 12.4+）

```bash
uv pip install --python .venv/bin/python \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

### 4. 安装运行时依赖

```bash
# 通用依赖
uv pip install --python .venv/bin/python -r requirements-common.txt

# CUDA 专属依赖（ray、xformers、nvidia-ml-py）
uv pip install --python .venv/bin/python \
  "ray[default]>=2.9" "nvidia-ml-py>=12.560.30" "xformers==0.0.28.post3"

# transformers 必须 < 5.0（5.x 移除了 all_special_tokens_extended）
uv pip install --python .venv/bin/python "transformers>=4.45.2,<5.0"
```

### 5. 从源码编译 CUDA 扩展

> 编译约需 30-60 分钟，使用 8 核并行。

```bash
PATH=".venv/bin:/usr/local/cuda-12.4/bin:$PATH" \
  MAX_JOBS=8 \
  .venv/bin/python setup.py build_ext --inplace
```

编译成功后将 `.so` 文件复制到 vllm 包目录：

```bash
BUILD=build/temp.linux-x86_64-cpython-312

cp $BUILD/_C.abi3.so                              vllm/_C.abi3.so
cp $BUILD/_moe_C.abi3.so                          vllm/_moe_C.abi3.so
cp $BUILD/cumem_allocator.abi3.so                 vllm/cumem_allocator.abi3.so
cp $BUILD/vllm-flash-attn/vllm_flash_attn_c.abi3.so  vllm/vllm_flash_attn/vllm_flash_attn_c.abi3.so
```

### 6. 安装 vllm Python 包（可编辑模式）

利用已编译的 `.so` 创建 fake wheel，跳过重复编译：

```bash
# 创建包含已编译 .so 的 fake wheel
python -c "
import zipfile, os
os.chdir('$(pwd)')
with zipfile.ZipFile('/tmp/vllm_prebuilt.whl', 'w', zipfile.ZIP_DEFLATED) as whl:
    whl.write('vllm/_C.abi3.so')
    whl.write('vllm/_moe_C.abi3.so')
    whl.write('vllm/cumem_allocator.abi3.so')
    whl.write('vllm/vllm_flash_attn/vllm_flash_attn_c.abi3.so')
"

# 可编辑安装
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_LOCATION=/tmp/vllm_prebuilt.whl \
PATH=".venv/bin:/usr/local/cuda-12.4/bin:$PATH" \
uv pip install --python .venv/bin/python --no-build-isolation -e .
```

### 7. 安装 ModelScope（国内网络下载模型）

> WSL 下 Python 的 SSL 与 HuggingFace 不兼容，改用 ModelScope。

```bash
uv pip install --python .venv/bin/python modelscope
```

---

## 二、验证安装

### 普通推理验证

运行 `test.py`：

```bash
.venv/bin/python test.py
```

`test.py` 内容（使用 ModelScope 加载模型）：

```python
from modelscope import snapshot_download
from vllm import LLM, SamplingParams

model_path = snapshot_download("qwen/Qwen2.5-1.5B-Instruct")

llm = LLM(
    model=model_path,
    dtype="bfloat16",
    gpu_memory_utilization=0.85,
    max_model_len=2048,
)

outputs = llm.generate(
    ["你好，请介绍一下自己。", "What is machine learning?"],
    SamplingParams(temperature=0.7, max_tokens=100)
)
for o in outputs:
    print(o.outputs[0].text)
    print("---")
```

### Sleep mode 功能验证

需要使用 V1 引擎（V0 引擎不含 sleep 实现）：

```bash
# 基础 allocator 测试（无需完整模型，快速验证 C 扩展和 cumem 机制）
.venv/bin/python tests/sleep_mode/test_cumem_basic.py

# 端到端 level 1 测试（offload 权重到 CPU，wake_up 后推理结果一致）
VLLM_USE_V1=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  .venv/bin/python tests/sleep_mode/test_end_to_end_level1.py

# 端到端 level 2 测试（分步唤醒 + reload_weights）
VLLM_USE_V1=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  .venv/bin/python tests/sleep_mode/test_end_to_end_level2_partial_wakeup.py
```

> **注意**：端到端测试每个文件独立运行；不建议在同一个 Python 进程中连续初始化多个 `LLM(enable_sleep_mode=True)` 实例（`CuMemAllocator` 是进程级单例）。

---

## 三、修改代码后的重新构建

### 只修改了 Python 代码

无需重新编译，直接运行即可（可编辑模式下修改立即生效）：

```bash
.venv/bin/python test.py
```

### 修改了 C++/CUDA 源码（`csrc/` 目录）

重新编译并更新 `.so`：

```bash
# 增量编译（cmake 自动检测变更，只重编修改的文件）
PATH=".venv/bin:/usr/local/cuda-12.4/bin:$PATH" \
  MAX_JOBS=8 \
  ninja -C build/temp.linux-x86_64-cpython-312 -j8 install 2>&1 | tail -5

# ninja install 因权限失败时，手动复制
BUILD=build/temp.linux-x86_64-cpython-312
cp $BUILD/_C.abi3.so                              vllm/_C.abi3.so
cp $BUILD/_moe_C.abi3.so                          vllm/_moe_C.abi3.so
cp $BUILD/cumem_allocator.abi3.so                 vllm/cumem_allocator.abi3.so
cp $BUILD/vllm-flash-attn/vllm_flash_attn_c.abi3.so  vllm/vllm_flash_attn/vllm_flash_attn_c.abi3.so
```

> **说明**：cmake 已配置好 build.ninja，增量编译只重新编译修改的 `.cu`/`.cpp` 文件，比全量编译快很多。

### 添加了新的 C++ 文件或修改了 CMakeLists.txt

需要重新 cmake 配置：

```bash
# 删除旧 build 目录
rm -rf build/temp.linux-x86_64-cpython-312

# 重新全量编译
PATH=".venv/bin:/usr/local/cuda-12.4/bin:$PATH" \
  MAX_JOBS=8 \
  .venv/bin/python setup.py build_ext --inplace

# 复制 .so
BUILD=build/temp.linux-x86_64-cpython-312
cp $BUILD/_C.abi3.so                              vllm/_C.abi3.so
cp $BUILD/_moe_C.abi3.so                          vllm/_moe_C.abi3.so
cp $BUILD/cumem_allocator.abi3.so                 vllm/cumem_allocator.abi3.so
cp $BUILD/vllm-flash-attn/vllm_flash_attn_c.abi3.so  vllm/vllm_flash_attn/vllm_flash_attn_c.abi3.so
```

> **注意**：如果 cmake 报错 `CMakeCache.txt directory is different`（项目路径发生变化时，如从 `/mnt/d/...` 迁移到 `/home/...`），需要额外清理 `.deps` 中的旧缓存：
>
> ```bash
> # 删除 stale subbuild 缓存（保留已下载的源码）
> rm -rf .deps/cutlass-subbuild .deps/vllm-flash-attn-subbuild
> # 再重新运行 setup.py build_ext --inplace
> ```

---

## 四、开发调试

### 4.1 日志级别控制

```bash
# 详细日志（DEBUG 级别）
VLLM_LOGGING_LEVEL=DEBUG .venv/bin/python test.py

# 只看 WARNING 以上
VLLM_LOGGING_LEVEL=WARNING .venv/bin/python test.py
```

### 4.2 禁用 CUDA Graph（简化调试）

CUDA graph capture 会合并内核调用，导致错误堆栈不清晰。调试时关闭：

```python
llm = LLM(
    model=model_path,
    enforce_eager=True,   # 禁用 cudagraph，报错时有完整堆栈
    ...
)
```

或环境变量：

```bash
VLLM_USE_RAY_SPMD_WORKER=0 VLLM_ENFORCE_EAGER=1 .venv/bin/python test.py
```

### 4.3 调试 C++ / CUDA 扩展

编译 Debug 版本（带符号，不优化）：

```bash
PATH=".venv/bin:/usr/local/cuda-12.4/bin:$PATH" \
  CMAKE_BUILD_TYPE=Debug \
  MAX_JOBS=8 \
  .venv/bin/python setup.py build_ext --inplace
```

使用 `cuda-gdb` 调试 CUDA kernel：

```bash
cuda-gdb --args .venv/bin/python test.py
(cuda-gdb) break layernorm_kernels.cu:42
(cuda-gdb) run
```

### 4.4 GPU 内存分析

```python
# 在 LLM 初始化前后对比显存
import torch
torch.cuda.reset_peak_memory_stats()

llm = LLM(model=model_path, ...)

print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print(f"Current GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

或使用 `nvidia-smi` 实时监控：

```bash
# 另开终端
watch -n 0.5 nvidia-smi
```

### 4.5 性能 Profiling

```python
# 用 torch.profiler 分析推理性能
import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    with record_function("vllm_generate"):
        outputs = llm.generate(prompts, sampling_params)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("trace.json")  # 用 chrome://tracing 可视化
```

### 4.6 单元测试

```bash
# 运行指定测试文件
.venv/bin/python -m pytest tests/test_embedded_commit.py -v

# 运行量化相关测试
.venv/bin/python -m pytest tests/quantization/ -v -k "cpu_offload"

# 跳过需要多 GPU 的测试
.venv/bin/python -m pytest tests/ -v -m "not multi_gpu"
```

### 4.7 常见错误排查

| 错误 | 原因 | 解决 |
|------|------|------|
| `ImportError: undefined symbol` | `.so` 与 torch 版本不匹配 | 重新编译 `setup.py build_ext --inplace` |
| `AttributeError: all_special_tokens_extended` | transformers >= 5.0 | `uv pip install "transformers<5.0"` |
| `CUDA OOM` | 显存不足 | 降低 `gpu_memory_utilization` 或 `max_model_len` |
| `Unable to find python matching` | 系统 Python 缺少开发头文件 | 用 uv 管理的 Python 创建 venv |
| SSL 下载模型失败（WSL） | WSL Python SSL 与 HuggingFace 不兼容 | 改用 ModelScope：`snapshot_download(...)` |
| `cuptiActivityEnableDriverApi` 缺失 | torch 版本要求 CUDA > 系统版本 | 降级 torch 至 `2.5.1+cu121` |
| `CMakeCache.txt directory is different` | 项目目录迁移后 `.deps/*-subbuild` 缓存路径过期 | `rm -rf .deps/cutlass-subbuild .deps/vllm-flash-attn-subbuild` 后重新编译 |
| `.so` 文件缺失但 `.o` 文件存在 | 编译过程被中断，链接步骤未执行 | `ninja -C build/temp.linux-x86_64-cpython-312 <target_name>` 只重链对应目标 |
| `sleep()` 后显存释放接近 0 | `enable_cumem_allocator` 未正确启用（验证代码放错类） | 确认 sleep mode 验证逻辑在 `ModelConfig.__init__` 末尾，不在其他类的 `__post_init__` 中 |
| `max_seq_len > max tokens in KV cache` | V1 引擎 + torch.compile 占用更多显存 | 初始化时设置 `max_model_len=2048` |
| `TypeError: load_model() got unexpected keyword argument 'load_dummy_weights'` | V1 `GPUModelRunner.load_model()` 不接受参数 | Worker 的 `load_model` 调用改为 `self.model_runner.load_model()`（无参数）|

---

## 五、快速参考

```bash
# 激活环境
source .venv/bin/activate

# 增量编译 CUDA 扩展（已有 build 目录时）
PATH=".venv/bin:/usr/local/cuda-12.4/bin:$PATH" MAX_JOBS=8 \
  ninja -C build/temp.linux-x86_64-cpython-312 -j8 && \
  BUILD=build/temp.linux-x86_64-cpython-312 && \
  cp $BUILD/_C.abi3.so vllm/ && \
  cp $BUILD/_moe_C.abi3.so vllm/ && \
  cp $BUILD/cumem_allocator.abi3.so vllm/

# 运行测试
.venv/bin/python test.py

# 查看 GPU 状态
nvidia-smi
```
