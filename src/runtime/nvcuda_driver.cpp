// SPDX-License-Identifier: Apache-2.0
#define NVCUDA_EXPORTS
#include "cuda.h"
#include "cuda_runtime.h"

#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <map>
#include <mutex>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

struct CUctx_st {
    CUdevice device = 0;
    unsigned int flags = 0;
};

struct CUfunc_st {
    std::string name;
    std::string spvPath;
    std::vector<uint32_t> spvBinary;
    size_t paramCount = 4;
};

struct CUmod_st {
    std::string path;
    std::vector<uint8_t> rawData;
    std::map<std::string, CUfunction> functions;
    std::mutex mtx;
};

namespace {
    thread_local CUcontext t_currentContext = nullptr;
    std::mutex g_driverMutex;
    bool g_initialized = false;
}

extern "C" {

// =========================================================================
// 1. Core Initialization & Driver Info
// =========================================================================

NVCUDA_API CUresult CUDAAPI cuInit(unsigned int Flags) {
    (void)Flags;
    std::lock_guard<std::mutex> lock(g_driverMutex);
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count <= 0) {
        return CUDA_ERROR_NO_DEVICE;
    }
    g_initialized = true;
    return CUDA_SUCCESS;
}

NVCUDA_API CUresult CUDAAPI cuDriverGetVersion(int* driverVersion) {
    if (!driverVersion) return CUDA_ERROR_INVALID_VALUE;
    *driverVersion = 12040; // CUDA 12.4 Driver
    return CUDA_SUCCESS;
}

// =========================================================================
// 2. Device Management
// =========================================================================

NVCUDA_API CUresult CUDAAPI cuDeviceGetCount(int* count) {
    if (!count) return CUDA_ERROR_INVALID_VALUE;
    cudaError_t err = cudaGetDeviceCount(count);
    return (err == cudaSuccess) ? CUDA_SUCCESS : CUDA_ERROR_NO_DEVICE;
}

NVCUDA_API CUresult CUDAAPI cuDeviceGet(CUdevice* device, int ordinal) {
    if (!device) return CUDA_ERROR_INVALID_VALUE;
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || ordinal < 0 || ordinal >= count) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    *device = ordinal;
    return CUDA_SUCCESS;
}

NVCUDA_API CUresult CUDAAPI cuDeviceGetName(char* name, int len, CUdevice dev) {
    if (!name || len <= 0) return CUDA_ERROR_INVALID_VALUE;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, dev);
    if (err != cudaSuccess) return CUDA_ERROR_INVALID_DEVICE;

    std::string dname = prop.name;
    if (dname.find("Vulkan-CUDA") == std::string::npos) {
        dname += " (Vulkan-CUDA)";
    }
    size_t copyLen = std::min<size_t>(static_cast<size_t>(len - 1), dname.size());
    std::memcpy(name, dname.c_str(), copyLen);
    name[copyLen] = '\0';
    return CUDA_SUCCESS;
}

NVCUDA_API CUresult CUDAAPI cuDeviceTotalMem_v2(size_t* bytes, CUdevice dev) {
    if (!bytes) return CUDA_ERROR_INVALID_VALUE;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, dev);
    if (err != cudaSuccess) return CUDA_ERROR_INVALID_DEVICE;
    *bytes = prop.totalGlobalMem;
    return CUDA_SUCCESS;
}

NVCUDA_API CUresult CUDAAPI cuDeviceTotalMem(size_t* bytes, CUdevice dev) {
    return cuDeviceTotalMem_v2(bytes, dev);
}

NVCUDA_API CUresult CUDAAPI cuDeviceComputeCapability(int* major, int* minor, CUdevice dev) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, dev);
    if (err != cudaSuccess) return CUDA_ERROR_INVALID_DEVICE;
    if (major) *major = prop.major;
    if (minor) *minor = prop.minor;
    return CUDA_SUCCESS;
}

NVCUDA_API CUresult CUDAAPI cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) {
    if (!pi) return CUDA_ERROR_INVALID_VALUE;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, dev);
    if (err != cudaSuccess) return CUDA_ERROR_INVALID_DEVICE;

    switch (attrib) {
        case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: *pi = prop.maxThreadsPerBlock; break;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X: *pi = prop.maxThreadsDim[0]; break;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: *pi = prop.maxThreadsDim[1]; break;
        case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z: *pi = prop.maxThreadsDim[2]; break;
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: *pi = prop.maxGridSize[0]; break;
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: *pi = prop.maxGridSize[1]; break;
        case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z: *pi = prop.maxGridSize[2]; break;
        case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: *pi = static_cast<int>(prop.sharedMemPerBlock); break;
        case CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY: *pi = static_cast<int>(prop.totalConstMem); break;
        case CU_DEVICE_ATTRIBUTE_WARP_SIZE: *pi = prop.warpSize; break;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: *pi = prop.major; break;
        case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: *pi = prop.minor; break;
        case CU_DEVICE_ATTRIBUTE_INTEGRATED: *pi = prop.integrated; break;
        case CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING: *pi = prop.unifiedAddressing; break;
        case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: *pi = prop.multiProcessorCount; break;
        case CU_DEVICE_ATTRIBUTE_CLOCK_RATE: *pi = prop.clockRate; break;
        case CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT: *pi = static_cast<int>(prop.textureAlignment); break;
        case CU_DEVICE_ATTRIBUTE_GPU_OVERLAP: *pi = 1; break;
        case CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS: *pi = prop.concurrentKernels; break;
        case CU_DEVICE_ATTRIBUTE_PCI_BUS_ID: *pi = prop.pciBusID; break;
        case CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID: *pi = prop.pciDeviceID; break;
        case CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS: *pi = 1; break;
        default: *pi = 0; break;
    }
    return CUDA_SUCCESS;
}

// =========================================================================
// 3. Context Management
// =========================================================================

NVCUDA_API CUresult CUDAAPI cuCtxCreate_v2(CUcontext* pctx, unsigned int flags, CUdevice dev) {
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    CUcontext ctx = new CUctx_st();
    ctx->device = dev;
    ctx->flags = flags;
    t_currentContext = ctx;
    *pctx = ctx;
    return CUDA_SUCCESS;
}

NVCUDA_API CUresult CUDAAPI cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) {
    return cuCtxCreate_v2(pctx, flags, dev);
}

NVCUDA_API CUresult CUDAAPI cuCtxDestroy_v2(CUcontext ctx) {
    if (!ctx) return CUDA_SUCCESS;
    if (t_currentContext == ctx) {
        t_currentContext = nullptr;
    }
    delete ctx;
    return CUDA_SUCCESS;
}

NVCUDA_API CUresult CUDAAPI cuCtxDestroy(CUcontext ctx) {
    return cuCtxDestroy_v2(ctx);
}

NVCUDA_API CUresult CUDAAPI cuCtxPushCurrent_v2(CUcontext ctx) {
    t_currentContext = ctx;
    return CUDA_SUCCESS;
}

NVCUDA_API CUresult CUDAAPI cuCtxPushCurrent(CUcontext ctx) {
    return cuCtxPushCurrent_v2(ctx);
}

NVCUDA_API CUresult CUDAAPI cuCtxPopCurrent_v2(CUcontext* pctx) {
    if (pctx) *pctx = t_currentContext;
    t_currentContext = nullptr;
    return CUDA_SUCCESS;
}

NVCUDA_API CUresult CUDAAPI cuCtxPopCurrent(CUcontext* pctx) {
    return cuCtxPopCurrent_v2(pctx);
}

NVCUDA_API CUresult CUDAAPI cuCtxSetCurrent(CUcontext ctx) {
    t_currentContext = ctx;
    return CUDA_SUCCESS;
}

NVCUDA_API CUresult CUDAAPI cuCtxGetCurrent(CUcontext* pctx) {
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    if (!t_currentContext) {
        cuCtxCreate_v2(&t_currentContext, 0, 0);
    }
    *pctx = t_currentContext;
    return CUDA_SUCCESS;
}

NVCUDA_API CUresult CUDAAPI cuCtxGetDevice(CUdevice* device) {
    if (!device) return CUDA_ERROR_INVALID_VALUE;
    *device = t_currentContext ? t_currentContext->device : 0;
    return CUDA_SUCCESS;
}

NVCUDA_API CUresult CUDAAPI cuCtxSynchronize(void) {
    cudaError_t err = cudaDeviceSynchronize();
    return (err == cudaSuccess) ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

NVCUDA_API CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
    return cuCtxCreate_v2(pctx, 0, dev);
}

NVCUDA_API CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice dev) {
    (void)dev;
    return CUDA_SUCCESS;
}

NVCUDA_API CUresult CUDAAPI cuDevicePrimaryCtxReset(CUdevice dev) {
    (void)dev;
    return CUDA_SUCCESS;
}

NVCUDA_API CUresult CUDAAPI cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active) {
    (void)dev;
    if (flags) *flags = 0;
    if (active) *active = 1;
    return CUDA_SUCCESS;
}

// =========================================================================
// 4. Memory Allocation & Addressing
// =========================================================================

NVCUDA_API CUresult CUDAAPI cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) {
    if (!dptr) return CUDA_ERROR_INVALID_VALUE;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(dptr), bytesize);
    if (err == cudaSuccess) return CUDA_SUCCESS;
    if (err == cudaErrorMemoryAllocation) return CUDA_ERROR_OUT_OF_MEMORY;
    return CUDA_ERROR_INVALID_VALUE;
}

NVCUDA_API CUresult CUDAAPI cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
    return cuMemAlloc_v2(dptr, bytesize);
}

NVCUDA_API CUresult CUDAAPI cuMemFree_v2(CUdeviceptr dptr) {
    if (dptr == 0) return CUDA_SUCCESS;
    cudaError_t err = cudaFree(reinterpret_cast<void*>(dptr));
    return (err == cudaSuccess) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

NVCUDA_API CUresult CUDAAPI cuMemFree(CUdeviceptr dptr) {
    return cuMemFree_v2(dptr);
}

NVCUDA_API CUresult CUDAAPI cuMemGetAddressRange_v2(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr) {
    if (!pbase || !psize) return CUDA_ERROR_INVALID_VALUE;
    *pbase = dptr;
    *psize = 0;
    return CUDA_SUCCESS;
}

NVCUDA_API CUresult CUDAAPI cuMemGetAddressRange(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr) {
    return cuMemGetAddressRange_v2(pbase, psize, dptr);
}

NVCUDA_API CUresult CUDAAPI cuMemAllocPitch_v2(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) {
    (void)ElementSizeBytes;
    if (!dptr || !pPitch) return CUDA_ERROR_INVALID_VALUE;
    size_t pitch = (WidthInBytes + 255) & ~((size_t)255);
    size_t total = pitch * Height;
    *pPitch = pitch;
    return cuMemAlloc_v2(dptr, total);
}

NVCUDA_API CUresult CUDAAPI cuMemAllocPitch(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) {
    return cuMemAllocPitch_v2(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
}

NVCUDA_API CUresult CUDAAPI cuMemAllocHost_v2(void** pp, size_t bytesize) {
    if (!pp || bytesize == 0) return CUDA_ERROR_INVALID_VALUE;
    *pp = std::malloc(bytesize);
    return (*pp) ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
}

NVCUDA_API CUresult CUDAAPI cuMemAllocHost(void** pp, size_t bytesize) {
    return cuMemAllocHost_v2(pp, bytesize);
}

NVCUDA_API CUresult CUDAAPI cuMemFreeHost(void* p) {
    if (p) std::free(p);
    return CUDA_SUCCESS;
}

NVCUDA_API CUresult CUDAAPI cuMemHostAlloc(void** pp, size_t bytesize, unsigned int Flags) {
    (void)Flags;
    return cuMemAllocHost_v2(pp, bytesize);
}

NVCUDA_API CUresult CUDAAPI cuMemHostGetDevicePointer_v2(CUdeviceptr* pdptr, void* p, unsigned int Flags) {
    (void)Flags;
    if (!pdptr || !p) return CUDA_ERROR_INVALID_VALUE;
    *pdptr = reinterpret_cast<CUdeviceptr>(p);
    return CUDA_SUCCESS;
}

NVCUDA_API CUresult CUDAAPI cuMemHostGetDevicePointer(CUdeviceptr* pdptr, void* p, unsigned int Flags) {
    return cuMemHostGetDevicePointer_v2(pdptr, p, Flags);
}

// =========================================================================
// 5. Memory Copies & Transfers
// =========================================================================

NVCUDA_API CUresult CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
    cudaError_t err = cudaMemcpy(reinterpret_cast<void*>(dstDevice), srcHost, ByteCount, cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

NVCUDA_API CUresult CUDAAPI cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
    return cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount);
}

NVCUDA_API CUresult CUDAAPI cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    cudaError_t err = cudaMemcpy(dstHost, reinterpret_cast<const void*>(srcDevice), ByteCount, cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

NVCUDA_API CUresult CUDAAPI cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    return cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount);
}

NVCUDA_API CUresult CUDAAPI cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
    cudaError_t err = cudaMemcpy(reinterpret_cast<void*>(dstDevice), reinterpret_cast<const void*>(srcDevice), ByteCount, cudaMemcpyDeviceToDevice);
    return (err == cudaSuccess) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

NVCUDA_API CUresult CUDAAPI cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
    return cuMemcpyDtoD_v2(dstDevice, srcDevice, ByteCount);
}

NVCUDA_API CUresult CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream) {
    cudaError_t err = cudaMemcpyAsync(reinterpret_cast<void*>(dstDevice), srcHost, ByteCount, cudaMemcpyHostToDevice, hStream);
    return (err == cudaSuccess) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

NVCUDA_API CUresult CUDAAPI cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream) {
    return cuMemcpyHtoDAsync_v2(dstDevice, srcHost, ByteCount, hStream);
}

NVCUDA_API CUresult CUDAAPI cuMemcpyDtoHAsync_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    cudaError_t err = cudaMemcpyAsync(dstHost, reinterpret_cast<const void*>(srcDevice), ByteCount, cudaMemcpyDeviceToHost, hStream);
    return (err == cudaSuccess) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

NVCUDA_API CUresult CUDAAPI cuMemcpyDtoHAsync(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    return cuMemcpyDtoHAsync_v2(dstHost, srcDevice, ByteCount, hStream);
}

NVCUDA_API CUresult CUDAAPI cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    cudaError_t err = cudaMemcpyAsync(reinterpret_cast<void*>(dstDevice), reinterpret_cast<const void*>(srcDevice), ByteCount, cudaMemcpyDeviceToDevice, hStream);
    return (err == cudaSuccess) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

NVCUDA_API CUresult CUDAAPI cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    return cuMemcpyDtoDAsync_v2(dstDevice, srcDevice, ByteCount, hStream);
}

NVCUDA_API CUresult CUDAAPI cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
    cudaError_t err = cudaMemset(reinterpret_cast<void*>(dstDevice), uc, N);
    return (err == cudaSuccess) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

NVCUDA_API CUresult CUDAAPI cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
    return cuMemsetD8_v2(dstDevice, uc, N);
}

NVCUDA_API CUresult CUDAAPI cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
    std::vector<unsigned int> hostBuf(N, ui);
    cudaError_t err = cudaMemcpy(reinterpret_cast<void*>(dstDevice), hostBuf.data(), N * sizeof(unsigned int), cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

NVCUDA_API CUresult CUDAAPI cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
    return cuMemsetD32_v2(dstDevice, ui, N);
}

// =========================================================================
// 6. Module & Function Management
// =========================================================================

NVCUDA_API CUresult CUDAAPI cuModuleLoad(CUmodule* module, const char* fname) {
    if (!module || !fname) return CUDA_ERROR_INVALID_VALUE;
    CUmodule mod = new CUmod_st();
    mod->path = fname;

    // If file exists, read it
    FILE* f = fopen(fname, "rb");
    if (f) {
        fseek(f, 0, SEEK_END);
        size_t sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        mod->rawData.resize(sz);
        fread(mod->rawData.data(), 1, sz, f);
        fclose(f);
    }

    *module = mod;
    return CUDA_SUCCESS;
}

NVCUDA_API CUresult CUDAAPI cuModuleLoadData(CUmodule* module, const void* image) {
    if (!module || !image) return CUDA_ERROR_INVALID_VALUE;
    CUmodule mod = new CUmod_st();
    *module = mod;
    return CUDA_SUCCESS;
}

NVCUDA_API CUresult CUDAAPI cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int numOptions, void* options, void** optionValues) {
    (void)numOptions; (void)options; (void)optionValues;
    return cuModuleLoadData(module, image);
}

NVCUDA_API CUresult CUDAAPI cuModuleLoadFatBinary(CUmodule* module, const void* fatCubin) {
    return cuModuleLoadData(module, fatCubin);
}

NVCUDA_API CUresult CUDAAPI cuModuleUnload(CUmodule hmod) {
    if (!hmod) return CUDA_SUCCESS;
    delete hmod;
    return CUDA_SUCCESS;
}

NVCUDA_API CUresult CUDAAPI cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
    if (!hfunc || !name) return CUDA_ERROR_INVALID_VALUE;

    CUfunction fn = new CUfunc_st();
    fn->name = name;
    if (hmod) {
        fn->spvPath = hmod->path;
    }

    // Default fallback search path for shaders if needed
    if (fn->spvPath.empty() || !fs::exists(fn->spvPath)) {
        std::string candidate = std::string("build/shaders/") + name + ".spv";
        if (fs::exists(candidate)) {
            fn->spvPath = candidate;
        } else {
            std::string candidateDist = std::string("dist/shaders/") + name + ".spv";
            if (fs::exists(candidateDist)) {
                fn->spvPath = candidateDist;
            } else {
                delete fn;
                return CUDA_ERROR_NOT_FOUND;
            }
        }
    }

    *hfunc = fn;
    return CUDA_SUCCESS;
}

NVCUDA_API CUresult CUDAAPI cuModuleGetGlobal_v2(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name) {
    (void)hmod; (void)name;
    if (!dptr || !bytes) return CUDA_ERROR_INVALID_VALUE;
    *dptr = 0;
    *bytes = 0;
    return CUDA_SUCCESS;
}

NVCUDA_API CUresult CUDAAPI cuModuleGetGlobal(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name) {
    return cuModuleGetGlobal_v2(dptr, bytes, hmod, name);
}

// =========================================================================
// 7. Kernel Dispatch (Driver API -> Vulkan Compute Pipeline)
// =========================================================================

NVCUDA_API CUresult CUDAAPI cuLaunchKernel(
    CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void** kernelParams,
    void** extra
) {
    (void)sharedMemBytes;
    if (!f) return CUDA_ERROR_INVALID_VALUE;

    std::vector<uint8_t> pushConstants;

    if (kernelParams) {
        // In standard 64-bit BDA, parameters are laid out sequentially.
        // Copy parameter values into push constants block.
        size_t count = (f->paramCount > 0) ? f->paramCount : 4;
        pushConstants.resize(count * sizeof(uint64_t));
        for (size_t i = 0; i < count; ++i) {
            if (kernelParams[i]) {
                std::memcpy(pushConstants.data() + i * sizeof(uint64_t), kernelParams[i], sizeof(uint64_t));
            }
        }
    } else if (extra) {
        void* bufferPtr = nullptr;
        size_t bufferSize = 0;
        for (size_t i = 0; extra[i] != nullptr; ++i) {
            if (extra[i] == reinterpret_cast<void*>(1) /* CU_LAUNCH_PARAM_BUFFER_POINTER */) {
                bufferPtr = extra[i + 1];
            } else if (extra[i] == reinterpret_cast<void*>(2) /* CU_LAUNCH_PARAM_BUFFER_SIZE */) {
                if (extra[i + 1]) {
                    bufferSize = *reinterpret_cast<size_t*>(extra[i + 1]);
                }
            }
        }
        if (bufferPtr && bufferSize > 0) {
            pushConstants.resize(bufferSize);
            std::memcpy(pushConstants.data(), bufferPtr, bufferSize);
        }
    }

    dim3 grid(gridDimX, gridDimY, gridDimZ);
    dim3 block(blockDimX, blockDimY, blockDimZ);

    cudaError_t err = cudaLaunchSpirv(
        f->spvPath.c_str(),
        grid,
        block,
        pushConstants.empty() ? nullptr : pushConstants.data(),
        pushConstants.size(),
        hStream
    );

    return (err == cudaSuccess) ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

// =========================================================================
// 8. Stream Management
// =========================================================================

NVCUDA_API CUresult CUDAAPI cuStreamCreate(CUstream* phStream, unsigned int Flags) {
    (void)Flags;
    cudaError_t err = cudaStreamCreate(phStream);
    return (err == cudaSuccess) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

NVCUDA_API CUresult CUDAAPI cuStreamCreateWithPriority(CUstream* phStream, unsigned int flags, int priority) {
    (void)flags; (void)priority;
    return cuStreamCreate(phStream, 0);
}

NVCUDA_API CUresult CUDAAPI cuStreamDestroy_v2(CUstream hStream) {
    cudaError_t err = cudaStreamDestroy(hStream);
    return (err == cudaSuccess) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

NVCUDA_API CUresult CUDAAPI cuStreamDestroy(CUstream hStream) {
    return cuStreamDestroy_v2(hStream);
}

NVCUDA_API CUresult CUDAAPI cuStreamSynchronize(CUstream hStream) {
    cudaError_t err = cudaStreamSynchronize(hStream);
    return (err == cudaSuccess) ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

NVCUDA_API CUresult CUDAAPI cuStreamQuery(CUstream hStream) {
    (void)hStream;
    return CUDA_SUCCESS;
}

NVCUDA_API CUresult CUDAAPI cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) {
    (void)hStream; (void)hEvent; (void)Flags;
    return CUDA_SUCCESS;
}

// =========================================================================
// 9. Event Management
// =========================================================================

NVCUDA_API CUresult CUDAAPI cuEventCreate(CUevent* phEvent, unsigned int Flags) {
    (void)Flags;
    cudaError_t err = cudaEventCreate(phEvent);
    return (err == cudaSuccess) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

NVCUDA_API CUresult CUDAAPI cuEventDestroy_v2(CUevent hEvent) {
    cudaError_t err = cudaEventDestroy(hEvent);
    return (err == cudaSuccess) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

NVCUDA_API CUresult CUDAAPI cuEventDestroy(CUevent hEvent) {
    return cuEventDestroy_v2(hEvent);
}

NVCUDA_API CUresult CUDAAPI cuEventRecord(CUevent hEvent, CUstream hStream) {
    cudaError_t err = cudaEventRecord(hEvent, hStream);
    return (err == cudaSuccess) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

NVCUDA_API CUresult CUDAAPI cuEventSynchronize(CUevent hEvent) {
    cudaError_t err = cudaEventSynchronize(hEvent);
    return (err == cudaSuccess) ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
}

NVCUDA_API CUresult CUDAAPI cuEventQuery(CUevent hEvent) {
    (void)hEvent;
    return CUDA_SUCCESS;
}

NVCUDA_API CUresult CUDAAPI cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd) {
    cudaError_t err = cudaEventElapsedTime(pMilliseconds, hStart, hEnd);
    return (err == cudaSuccess) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

// =========================================================================
// 10. Error Handling
// =========================================================================

NVCUDA_API CUresult CUDAAPI cuGetErrorString(CUresult error, const char** pStr) {
    if (!pStr) return CUDA_ERROR_INVALID_VALUE;
    switch (error) {
        case CUDA_SUCCESS: *pStr = "CUDA_SUCCESS"; break;
        case CUDA_ERROR_INVALID_VALUE: *pStr = "CUDA_ERROR_INVALID_VALUE"; break;
        case CUDA_ERROR_OUT_OF_MEMORY: *pStr = "CUDA_ERROR_OUT_OF_MEMORY"; break;
        case CUDA_ERROR_NOT_INITIALIZED: *pStr = "CUDA_ERROR_NOT_INITIALIZED"; break;
        case CUDA_ERROR_NO_DEVICE: *pStr = "CUDA_ERROR_NO_DEVICE"; break;
        case CUDA_ERROR_INVALID_DEVICE: *pStr = "CUDA_ERROR_INVALID_DEVICE"; break;
        case CUDA_ERROR_INVALID_IMAGE: *pStr = "CUDA_ERROR_INVALID_IMAGE"; break;
        case CUDA_ERROR_INVALID_CONTEXT: *pStr = "CUDA_ERROR_INVALID_CONTEXT"; break;
        case CUDA_ERROR_LAUNCH_FAILED: *pStr = "CUDA_ERROR_LAUNCH_FAILED"; break;
        default: *pStr = "CUDA_ERROR_UNKNOWN"; break;
    }
    return CUDA_SUCCESS;
}

NVCUDA_API CUresult CUDAAPI cuGetErrorName(CUresult error, const char** pStr) {
    return cuGetErrorString(error, pStr);
}

} // extern "C"
