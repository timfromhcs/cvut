// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(CUDART_EXPORTS)
#    define CUDART_API __declspec(dllexport)
#  else
#    define CUDART_API __declspec(dllimport)
#  endif
#else
#  define CUDART_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorLaunchFailure = 4,
    cudaErrorPriorLaunchFailure = 5,
    cudaErrorLaunchTimeout = 6,
    cudaErrorLaunchOutOfResources = 7,
    cudaErrorInvalidDeviceFunction = 8,
    cudaErrorInvalidConfiguration = 9,
    cudaErrorInvalidDevice = 10,
    cudaErrorInvalidValue2 = 11,
    cudaErrorInvalidPitchValue = 12,
    cudaErrorInvalidSymbol = 13,
    cudaErrorMapBufferObjectFailed = 14,
    cudaErrorUnmapBufferObjectFailed = 15,
    cudaErrorArrayIsMapped = 16,
    cudaErrorAlreadyMapped = 17,
    cudaErrorNoDevice = 18,
    cudaErrorAlreadyAcquired = 19,
    cudaErrorNotMapped = 20,
    cudaErrorNotMappedAsArray = 21,
    cudaErrorNotMappedAsPointer = 22,
    cudaErrorECCUncorrectable = 23,
    cudaErrorUnsupportedLimit = 24,
    cudaErrorDeviceAlreadyInUse = 25,
    cudaErrorPeerAccessUnsupported = 26,
    cudaErrorInvalidPtx = 27,
    cudaErrorInvalidGraphicsContext = 28,
    cudaErrorNvlinkUncorrectable = 29,
    cudaErrorJitCompilerNotFound = 30,
    cudaErrorInvalidSource = 31,
    cudaErrorFileNotFound = 32,
    cudaErrorSharedObjectSymbolNotFound = 33,
    cudaErrorSharedObjectInitFailed = 34,
    cudaErrorOperatingSystem = 35,
    cudaErrorInvalidResourceHandle = 36,
    cudaErrorIllegalState = 37,
    cudaErrorSymbolNotFound = 38,
    cudaErrorNotReady = 39,
    cudaErrorIllegalAddress = 40,
    cudaErrorLaunchOutOfMemory = 41,
    cudaErrorNotSupported = 801,
    cudaErrorUnknown = 999
} cudaError_t;

typedef enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
} cudaMemcpyKind;

struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;

struct CUevent_st;
typedef struct CUevent_st* cudaEvent_t;

typedef struct cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    size_t totalConstMem;
    int major;
    int minor;
    size_t textureAlignment;
    int multiProcessorCount;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int concurrentKernels;
    int eccEnabled;
    int pciBusID;
    int pciDeviceID;
    int pciDomainID;
    int tccDriver;
    int asyncEngineCount;
    int unifiedAddressing;
    int memoryClockRate;
    int memoryBusWidth;
    int l2CacheSize;
    int maxThreadsPerMultiProcessor;
} cudaDeviceProp;

struct dim3 {
    unsigned int x, y, z;
#ifdef __cplusplus
    dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
#endif
};
#ifndef __cplusplus
typedef struct dim3 dim3;
#endif

// Memory API
CUDART_API cudaError_t cudaMalloc(void** devPtr, size_t size);
CUDART_API cudaError_t cudaFree(void* devPtr);
CUDART_API cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
CUDART_API cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream);
CUDART_API cudaError_t cudaMemset(void* devPtr, int value, size_t count);
CUDART_API cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream);
CUDART_API cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags);
CUDART_API cudaError_t cudaFreeHost(void* ptr);

// Stream API
CUDART_API cudaError_t cudaStreamCreate(cudaStream_t* pStream);
CUDART_API cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags);
CUDART_API cudaError_t cudaStreamDestroy(cudaStream_t stream);
CUDART_API cudaError_t cudaStreamSynchronize(cudaStream_t stream);
CUDART_API cudaError_t cudaStreamQuery(cudaStream_t stream);

// Device API
CUDART_API cudaError_t cudaGetDevice(int* device);
CUDART_API cudaError_t cudaSetDevice(int device);
CUDART_API cudaError_t cudaGetDeviceCount(int* count);
CUDART_API cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device);
CUDART_API cudaError_t cudaDeviceSynchronize(void);
CUDART_API cudaError_t cudaDeviceReset(void);

// Event API
CUDART_API cudaError_t cudaEventCreate(cudaEvent_t* event);
CUDART_API cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);
CUDART_API cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
CUDART_API cudaError_t cudaEventSynchronize(cudaEvent_t event);
CUDART_API cudaError_t cudaEventQuery(cudaEvent_t event);
CUDART_API cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end);
CUDART_API cudaError_t cudaEventDestroy(cudaEvent_t event);

// Error Handling
CUDART_API const char* cudaGetErrorString(cudaError_t error);
CUDART_API const char* cudaGetErrorName(cudaError_t error);
CUDART_API cudaError_t cudaGetLastError(void);
CUDART_API cudaError_t cudaPeekAtLastError(void);

// Kernel Launch API
CUDART_API cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
CUDART_API cudaError_t cudaLaunchSpirv(const char* spvPath, dim3 gridDim, dim3 blockDim, const void* pushConstants, size_t pushConstantsSize, cudaStream_t stream);
CUDART_API cudaError_t cudaGetVulkanContext(void** pInstance, void** pPhysicalDevice, void** pDevice, void** pQueue, uint32_t* pQueueFamily);
// CVUT extension: resolve a device pointer to its allocation base and size.
// Returns cudaErrorInvalidValue for unknown/freed pointers.
CUDART_API cudaError_t cudaGetAllocRange(void* devPtr, void** pBase, size_t* pSize);

#ifdef __cplusplus
}
#endif
