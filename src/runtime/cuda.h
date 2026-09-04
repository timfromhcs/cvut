// SPDX-License-Identifier: Apache-2.0
#ifndef CUDA_H
#define CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
  #define CUDAAPI __stdcall
  #if defined(NVCUDA_EXPORTS)
    #define NVCUDA_API __declspec(dllexport)
  #else
    #define NVCUDA_API __declspec(dllimport)
  #endif
#else
  #define CUDAAPI
  #define NVCUDA_API __attribute__((visibility("default")))
#endif

typedef enum cudaError_enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_PROFILER_DISABLED = 5,
    CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,
    CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,
    CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,
    CUDA_ERROR_STUB_LIBRARY = 34,
    CUDA_ERROR_DEVICE_UNAVAILABLE = 46,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_DEVICE_NOT_LICENSED = 102,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
    CUDA_ERROR_MAP_FAILED = 205,
    CUDA_ERROR_UNMAP_FAILED = 206,
    CUDA_ERROR_ARRAY_IS_MAPPED = 207,
    CUDA_ERROR_ALREADY_MAPPED = 208,
    CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
    CUDA_ERROR_ALREADY_ACQUIRED = 210,
    CUDA_ERROR_NOT_MAPPED = 211,
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
    CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,
    CUDA_ERROR_ECC_UNCORRECTABLE = 214,
    CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,
    CUDA_ERROR_INVALID_PTX = 218,
    CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,
    CUDA_ERROR_NVLINK_UNCORRECTABLE = 220,
    CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,
    CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222,
    CUDA_ERROR_JIT_COMPILATION_DISABLED = 223,
    CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY = 224,
    CUDA_ERROR_INVALID_SOURCE = 300,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
    CUDA_ERROR_OPERATING_SYSTEM = 304,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_ILLEGAL_STATE = 401,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_ILLEGAL_ADDRESS = 700,
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
    CUDA_ERROR_LAUNCH_TIMEOUT = 702,
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
    CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
    CUDA_ERROR_ASSERT = 710,
    CUDA_ERROR_TOO_MANY_PEERS = 711,
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
    CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
    CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
    CUDA_ERROR_MISALIGNED_ADDRESS = 716,
    CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
    CUDA_ERROR_INVALID_PC = 718,
    CUDA_ERROR_LAUNCH_FAILED = 719,
    CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
    CUDA_ERROR_NOT_PERMITTED = 800,
    CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_SYSTEM_NOT_READY = 802,
    CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803,
    CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,
    CUDA_ERROR_MPS_CONNECTION_FAILED = 805,
    CUDA_ERROR_MPS_RPC_FAILURE = 806,
    CUDA_ERROR_MPS_SERVER_NOT_FOUND = 807,
    CUDA_ERROR_MPS_MAX_CLIENTS_REACHED = 808,
    CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED = 809,
    CUDA_ERROR_MPS_CLIENT_TERMINATED = 810,
    CUDA_ERROR_CDP_NOT_SUPPORTED = 811,
    CUDA_ERROR_CDP_RESERVED_EXCEPTION = 812,
    CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900,
    CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901,
    CUDA_ERROR_STREAM_CAPTURE_MERGE = 902,
    CUDA_ERROR_STREAM_CAPTURE_UNMATCHED_WITHATION = 903,
    CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904,
    CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905,
    CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906,
    CUDA_ERROR_CAPTURED_EVENT = 907,
    CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908,
    CUDA_ERROR_TIMEOUT = 909,
    CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910,
    CUDA_ERROR_UNKNOWN = 999
} CUresult;

typedef int CUdevice;
typedef uintptr_t CUdeviceptr;
typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef struct CUstream_st* CUstream;
typedef struct CUevent_st* CUevent;

typedef enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96
} CUdevice_attribute;

// Core Initialization & Driver Info
NVCUDA_API CUresult CUDAAPI cuInit(unsigned int Flags);
NVCUDA_API CUresult CUDAAPI cuDriverGetVersion(int* driverVersion);

// Device Management
NVCUDA_API CUresult CUDAAPI cuDeviceGetCount(int* count);
NVCUDA_API CUresult CUDAAPI cuDeviceGet(CUdevice* device, int ordinal);
NVCUDA_API CUresult CUDAAPI cuDeviceGetName(char* name, int len, CUdevice dev);
NVCUDA_API CUresult CUDAAPI cuDeviceTotalMem_v2(size_t* bytes, CUdevice dev);
NVCUDA_API CUresult CUDAAPI cuDeviceTotalMem(size_t* bytes, CUdevice dev);
NVCUDA_API CUresult CUDAAPI cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev);
NVCUDA_API CUresult CUDAAPI cuDeviceComputeCapability(int* major, int* minor, CUdevice dev);

// Context Management
NVCUDA_API CUresult CUDAAPI cuCtxCreate_v2(CUcontext* pctx, unsigned int flags, CUdevice dev);
NVCUDA_API CUresult CUDAAPI cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev);
NVCUDA_API CUresult CUDAAPI cuCtxDestroy_v2(CUcontext ctx);
NVCUDA_API CUresult CUDAAPI cuCtxDestroy(CUcontext ctx);
NVCUDA_API CUresult CUDAAPI cuCtxPushCurrent_v2(CUcontext ctx);
NVCUDA_API CUresult CUDAAPI cuCtxPushCurrent(CUcontext ctx);
NVCUDA_API CUresult CUDAAPI cuCtxPopCurrent_v2(CUcontext* pctx);
NVCUDA_API CUresult CUDAAPI cuCtxPopCurrent(CUcontext* pctx);
NVCUDA_API CUresult CUDAAPI cuCtxSetCurrent(CUcontext ctx);
NVCUDA_API CUresult CUDAAPI cuCtxGetCurrent(CUcontext* pctx);
NVCUDA_API CUresult CUDAAPI cuCtxGetDevice(CUdevice* device);
NVCUDA_API CUresult CUDAAPI cuCtxSynchronize(void);
NVCUDA_API CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev);
NVCUDA_API CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice dev);
NVCUDA_API CUresult CUDAAPI cuDevicePrimaryCtxReset(CUdevice dev);
NVCUDA_API CUresult CUDAAPI cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active);

// Memory Management
NVCUDA_API CUresult CUDAAPI cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize);
NVCUDA_API CUresult CUDAAPI cuMemAlloc(CUdeviceptr* dptr, size_t bytesize);
NVCUDA_API CUresult CUDAAPI cuMemFree_v2(CUdeviceptr dptr);
NVCUDA_API CUresult CUDAAPI cuMemFree(CUdeviceptr dptr);
NVCUDA_API CUresult CUDAAPI cuMemGetAddressRange_v2(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr);
NVCUDA_API CUresult CUDAAPI cuMemGetAddressRange(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr);
NVCUDA_API CUresult CUDAAPI cuMemAllocPitch_v2(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes);
NVCUDA_API CUresult CUDAAPI cuMemAllocPitch(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes);
NVCUDA_API CUresult CUDAAPI cuMemAllocHost_v2(void** pp, size_t bytesize);
NVCUDA_API CUresult CUDAAPI cuMemAllocHost(void** pp, size_t bytesize);
NVCUDA_API CUresult CUDAAPI cuMemFreeHost(void* p);
NVCUDA_API CUresult CUDAAPI cuMemHostAlloc(void** pp, size_t bytesize, unsigned int Flags);
NVCUDA_API CUresult CUDAAPI cuMemHostGetDevicePointer_v2(CUdeviceptr* pdptr, void* p, unsigned int Flags);
NVCUDA_API CUresult CUDAAPI cuMemHostGetDevicePointer(CUdeviceptr* pdptr, void* p, unsigned int Flags);

// Memory Copies
NVCUDA_API CUresult CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount);
NVCUDA_API CUresult CUDAAPI cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount);
NVCUDA_API CUresult CUDAAPI cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
NVCUDA_API CUresult CUDAAPI cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
NVCUDA_API CUresult CUDAAPI cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);
NVCUDA_API CUresult CUDAAPI cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);

NVCUDA_API CUresult CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream);
NVCUDA_API CUresult CUDAAPI cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream);
NVCUDA_API CUresult CUDAAPI cuMemcpyDtoHAsync_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
NVCUDA_API CUresult CUDAAPI cuMemcpyDtoHAsync(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
NVCUDA_API CUresult CUDAAPI cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
NVCUDA_API CUresult CUDAAPI cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);

NVCUDA_API CUresult CUDAAPI cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N);
NVCUDA_API CUresult CUDAAPI cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N);
NVCUDA_API CUresult CUDAAPI cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N);
NVCUDA_API CUresult CUDAAPI cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N);

// Module & Function Management
NVCUDA_API CUresult CUDAAPI cuModuleLoad(CUmodule* module, const char* fname);
NVCUDA_API CUresult CUDAAPI cuModuleLoadData(CUmodule* module, const void* image);
NVCUDA_API CUresult CUDAAPI cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int numOptions, void* options, void** optionValues);
NVCUDA_API CUresult CUDAAPI cuModuleLoadFatBinary(CUmodule* module, const void* fatCubin);
NVCUDA_API CUresult CUDAAPI cuModuleUnload(CUmodule hmod);
NVCUDA_API CUresult CUDAAPI cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name);
NVCUDA_API CUresult CUDAAPI cuModuleGetGlobal_v2(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name);
NVCUDA_API CUresult CUDAAPI cuModuleGetGlobal(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name);

// Execution Control
NVCUDA_API CUresult CUDAAPI cuLaunchKernel(
    CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void** kernelParams,
    void** extra
);

// Stream Management
NVCUDA_API CUresult CUDAAPI cuStreamCreate(CUstream* phStream, unsigned int Flags);
NVCUDA_API CUresult CUDAAPI cuStreamCreateWithPriority(CUstream* phStream, unsigned int flags, int priority);
NVCUDA_API CUresult CUDAAPI cuStreamDestroy_v2(CUstream hStream);
NVCUDA_API CUresult CUDAAPI cuStreamDestroy(CUstream hStream);
NVCUDA_API CUresult CUDAAPI cuStreamSynchronize(CUstream hStream);
NVCUDA_API CUresult CUDAAPI cuStreamQuery(CUstream hStream);
NVCUDA_API CUresult CUDAAPI cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags);

// Event Management
NVCUDA_API CUresult CUDAAPI cuEventCreate(CUevent* phEvent, unsigned int Flags);
NVCUDA_API CUresult CUDAAPI cuEventDestroy_v2(CUevent hEvent);
NVCUDA_API CUresult CUDAAPI cuEventDestroy(CUevent hEvent);
NVCUDA_API CUresult CUDAAPI cuEventRecord(CUevent hEvent, CUstream hStream);
NVCUDA_API CUresult CUDAAPI cuEventSynchronize(CUevent hEvent);
NVCUDA_API CUresult CUDAAPI cuEventQuery(CUevent hEvent);
NVCUDA_API CUresult CUDAAPI cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd);

// Error Handling
NVCUDA_API CUresult CUDAAPI cuGetErrorString(CUresult error, const char** pStr);
NVCUDA_API CUresult CUDAAPI cuGetErrorName(CUresult error, const char** pStr);

#ifdef __cplusplus
}
#endif

#endif // CUDA_H
