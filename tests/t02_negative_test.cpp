// SPDX-License-Identifier: Apache-2.0
// T02: Negative tests for invalid runtime/driver API inputs.
//
// These cases must NOT require a physical GPU: every assertion below exercises
// argument-validation paths that run before Vulkan initialization. This keeps
// the test meaningful on headless CI runners (SKIP/BLOCKED GPU work stays in
// the E2E binaries, never silently passing here).
#include "../src/runtime/cuda_runtime.h"
#include "../src/runtime/cuda.h"

#include <cstdio>
#include <cstddef>

static int g_failures = 0;

#define CHECK_EQ(expr, expected, label)                                     \
    do {                                                                    \
        long long _got = static_cast<long long>(expr);                      \
        long long _want = static_cast<long long>(expected);                 \
        if (_got != _want) {                                                \
            std::printf("FAIL %-48s got %lld want %lld\n", label, _got, _want); \
            ++g_failures;                                                   \
        } else {                                                            \
            std::printf("ok   %s\n", label);                                \
        }                                                                   \
    } while (0)

int main() {
    void* p = reinterpret_cast<void*>(0x1234);

    // Probe: does this machine initialize a Vulkan device? Tier-2 checks
    // below require a live device; without one they report SKIP (never PASS).
    bool hasGpu = (cudaDeviceSynchronize() == cudaSuccess);
    std::printf("device probe: %s\n", hasGpu ? "GPU available" : "no GPU (tier-2 SKIP)");

    // --- Runtime API null/zero validation (pre-init paths) ---
    CHECK_EQ(cudaMalloc(nullptr, 100), cudaErrorInvalidValue, "cudaMalloc(null,100)");
    CHECK_EQ(cudaMalloc(&p, 0), cudaSuccess, "cudaMalloc(&p,0)");
    CHECK_EQ(p == nullptr, true, "cudaMalloc(0) nulls pointer");
    CHECK_EQ(cudaFree(nullptr), cudaSuccess, "cudaFree(null)");

    char buf[8] = {};
    CHECK_EQ(cudaMemcpy(nullptr, buf, 8, cudaMemcpyHostToDevice), cudaErrorInvalidValue, "H2D null dst");
    CHECK_EQ(cudaMemcpy(buf, nullptr, 8, cudaMemcpyHostToDevice), cudaErrorInvalidValue, "H2D null src");
    CHECK_EQ(cudaMemcpy(nullptr, nullptr, 0, cudaMemcpyHostToDevice), cudaSuccess, "count 0 ok");

    CHECK_EQ(cudaMemset(nullptr, 0, 10), cudaErrorInvalidValue, "cudaMemset(null,10)");
    CHECK_EQ(cudaMemset(p, 0, 0), cudaSuccess, "cudaMemset count 0 ok");

    CHECK_EQ(cudaStreamCreate(nullptr), cudaErrorInvalidValue, "cudaStreamCreate(null)");
    CHECK_EQ(cudaGetDeviceCount(nullptr), cudaErrorInvalidValue, "cudaGetDeviceCount(null)");
    CHECK_EQ(cudaGetDevice(nullptr), cudaErrorInvalidValue, "cudaGetDevice(null)");
    CHECK_EQ(cudaSetDevice(99), cudaErrorInvalidDevice, "cudaSetDevice(99)");
    CHECK_EQ(cudaSetDevice(-1), cudaErrorInvalidDevice, "cudaSetDevice(-1)");

    CHECK_EQ(cudaEventCreate(nullptr), cudaErrorInvalidValue, "cudaEventCreate(null)");
    CHECK_EQ(cudaEventRecord(nullptr, nullptr), cudaErrorInvalidValue, "cudaEventRecord(null)");
    CHECK_EQ(cudaEventQuery(nullptr), cudaErrorInvalidValue, "cudaEventQuery(null)");
    CHECK_EQ(cudaEventSynchronize(nullptr), cudaErrorInvalidValue, "cudaEventSynchronize(null)");

    // Raw function-pointer launch is explicitly unsupported (never fake success).
    CHECK_EQ(cudaLaunchKernel(nullptr, dim3(1), dim3(1), nullptr, 0, nullptr),
             cudaErrorInvalidDeviceFunction, "cudaLaunchKernel(null func)");
    int cookie = 0;
    CHECK_EQ(cudaLaunchKernel(&cookie, dim3(1), dim3(1), nullptr, 0, nullptr),
             cudaErrorNotSupported, "cudaLaunchKernel(raw func)->NotSupported");
    CHECK_EQ(cudaLaunchSpirv(nullptr, dim3(1), dim3(1), nullptr, 0, nullptr),
             cudaErrorInvalidValue, "cudaLaunchSpirv(null path)");
    CHECK_EQ(cudaLaunchSpirv("nope.spv", dim3(0, 1, 1), dim3(1), nullptr, 0, nullptr),
             cudaErrorInvalidConfiguration, "cudaLaunchSpirv(zero grid)");
    char pc[200] = {};
    CHECK_EQ(cudaLaunchSpirv("nope.spv", dim3(1), dim3(1), pc, sizeof(pc), nullptr),
             cudaErrorInvalidValue, "cudaLaunchSpirv(oversize push)");

    // Error strings cover the codes CVUT returns.
    CHECK_EQ(cudaGetErrorString(cudaErrorNotSupported) != nullptr, true, "errstr NotSupported");
    CHECK_EQ(cudaGetErrorString(cudaErrorNotReady) != nullptr, true, "errstr NotReady");

    // --- Driver API validation (pre-GPU paths) ---
    CHECK_EQ(cuDeviceGet(nullptr, 0), CUDA_ERROR_INVALID_VALUE, "cuDeviceGet(null)");
    CHECK_EQ(cuDeviceGetCount(nullptr), CUDA_ERROR_INVALID_VALUE, "cuDeviceGetCount(null)");
    CHECK_EQ(cuCtxCreate(nullptr, 0, 0), CUDA_ERROR_INVALID_VALUE, "cuCtxCreate(null)");
    CHECK_EQ(cuCtxPushCurrent(nullptr), CUDA_ERROR_INVALID_CONTEXT, "cuCtxPushCurrent(null)");
    CUcontext popped = reinterpret_cast<CUcontext>(0x1);
    CHECK_EQ(cuCtxPopCurrent(&popped), CUDA_ERROR_INVALID_CONTEXT, "cuCtxPopCurrent(empty)");
    CHECK_EQ(popped == nullptr, true, "pop(empty) nulls out-param");
    CHECK_EQ(cuCtxSetCurrent(nullptr), CUDA_SUCCESS, "cuCtxSetCurrent(null) detaches");

    CUdeviceptr base = 0;
    size_t size = 0;
    CHECK_EQ(cuMemGetAddressRange(nullptr, nullptr, 0), CUDA_ERROR_INVALID_VALUE, "addrRange(null,null)");
    CHECK_EQ(cuMemGetAddressRange(&base, &size, 0), CUDA_ERROR_INVALID_VALUE, "addrRange(null dptr)");

    CUdeviceptr pitchPtr = 0;
    size_t pitch = 0;
    CHECK_EQ(cuMemAllocPitch(nullptr, nullptr, 64, 64, 4), CUDA_ERROR_INVALID_VALUE, "pitch(null,null)");
    CHECK_EQ(cuMemAllocPitch(&pitchPtr, &pitch, 0, 64, 4), CUDA_ERROR_INVALID_VALUE, "pitch(zero W)");
    CHECK_EQ(cuMemAllocPitch(&pitchPtr, &pitch, 64, 0, 4), CUDA_ERROR_INVALID_VALUE, "pitch(zero H)");
    CHECK_EQ(cuMemAllocPitch(&pitchPtr, &pitch, (size_t)-1, 2, 4), CUDA_ERROR_OUT_OF_MEMORY, "pitch(overflow)");

    CHECK_EQ(cuMemsetD32(0, 0, 0), CUDA_SUCCESS, "memsetD32 zero N ok");
    CHECK_EQ(cuMemsetD32(0, 0, 4), CUDA_ERROR_INVALID_VALUE, "memsetD32 null dptr");

    CUdeviceptr gptr = 0;
    size_t gsize = 0;
    CHECK_EQ(cuModuleGetGlobal(&gptr, &gsize, nullptr, "sym"), CUDA_ERROR_NOT_FOUND, "getGlobal->NOT_FOUND");
    CHECK_EQ(cuModuleLoad(nullptr, nullptr), CUDA_ERROR_INVALID_VALUE, "moduleLoad(null)");
    CHECK_EQ(cuModuleGetFunction(nullptr, nullptr, nullptr), CUDA_ERROR_INVALID_VALUE, "getFunction(null)");

    // --- Tier-2: require a live Vulkan device; SKIP (not PASS) when headless ---
    if (hasGpu) {
        CHECK_EQ(cudaFree(reinterpret_cast<void*>(0xdead)), cudaErrorInvalidValue, "cudaFree(garbage)");
        CHECK_EQ(cuMemGetAddressRange(&base, &size, 0xdead), CUDA_ERROR_INVALID_VALUE, "addrRange(unknown)");
        CUresult q = cuStreamQuery(nullptr);
        CHECK_EQ(q == CUDA_SUCCESS || q == CUDA_ERROR_LAUNCH_FAILED, true, "cuStreamQuery(null) valid code");
    } else {
        std::printf("SKIP   tier-2 device checks (no GPU)\n");
    }

    if (g_failures == 0) {
        std::printf("TEST_PASSED: NEGATIVE_INPUTS_VALIDATED\n");
        return 0;
    }
    std::printf("TEST_FAILED: %d negative assertions failed\n", g_failures);
    return 1;
}
