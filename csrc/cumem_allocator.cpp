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