// SPDX-License-Identifier: Apache-2.0
#include "../src/runtime/cuda.h"

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdint>

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << " Starting Real-World CUDA Driver API E2E Verification        " << std::endl;
    std::cout << "============================================================" << std::endl;

    // 1. Initialize CUDA Driver API
    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        std::cerr << "cuInit failed: " << res << std::endl;
        return 1;
    }
    std::cout << "[DRIVER_E2E] 1. cuInit(0) returned CUDA_SUCCESS." << std::endl;

    // 2. Query Driver Version
    int driverVersion = 0;
    res = cuDriverGetVersion(&driverVersion);
    assert(res == CUDA_SUCCESS);
    std::cout << "[DRIVER_E2E] 2. CUDA Driver Version: " << driverVersion / 1000 << "." << (driverVersion % 100) / 10 << std::endl;

    // 3. Device Introspection
    int deviceCount = 0;
    res = cuDeviceGetCount(&deviceCount);
    assert(res == CUDA_SUCCESS && deviceCount > 0);
    std::cout << "[DRIVER_E2E] 3. Detected " << deviceCount << " compute-capable device(s)." << std::endl;

    CUdevice dev = 0;
    res = cuDeviceGet(&dev, 0);
    assert(res == CUDA_SUCCESS);

    char devName[256] = {0};
    res = cuDeviceGetName(devName, sizeof(devName), dev);
    assert(res == CUDA_SUCCESS);
    std::cout << "[DRIVER_E2E] 4. Active Device [0]: " << devName << std::endl;

    size_t totalMem = 0;
    res = cuDeviceTotalMem_v2(&totalMem, dev);
    assert(res == CUDA_SUCCESS);
    std::cout << "[DRIVER_E2E] 5. Device Global Memory: " << totalMem / (1024 * 1024) << " MiB." << std::endl;

    // 4. Create Driver Context
    CUcontext ctx = nullptr;
    res = cuCtxCreate_v2(&ctx, 0, dev);
    assert(res == CUDA_SUCCESS && ctx != nullptr);
    std::cout << "[DRIVER_E2E] 6. Created CUDA Driver Context." << std::endl;

    // 5. Memory Allocation via cuMemAlloc_v2 (with 256-byte alignment validation)
    const size_t NUM_ELEMENTS = 1048576; // 1M floats = 4 MB
    const size_t BUFFER_SIZE = NUM_ELEMENTS * sizeof(float);

    CUdeviceptr d_a = 0, d_b = 0, d_c = 0;
    res = cuMemAlloc_v2(&d_a, BUFFER_SIZE);
    assert(res == CUDA_SUCCESS && d_a != 0);
    assert((d_a % 256) == 0); // Deterministic INV-02 alignment

    res = cuMemAlloc_v2(&d_b, BUFFER_SIZE);
    assert(res == CUDA_SUCCESS && d_b != 0);
    assert((d_b % 256) == 0);

    res = cuMemAlloc_v2(&d_c, BUFFER_SIZE);
    assert(res == CUDA_SUCCESS && d_c != 0);
    assert((d_c % 256) == 0);

    std::cout << "[DRIVER_E2E] 7. Allocated 3x 4MB buffers via cuMemAlloc_v2 (256-byte aligned)." << std::endl;

    // 6. Host Memory Preparation and cuMemcpyHtoD_v2 Transfers
    std::vector<float> h_a(NUM_ELEMENTS);
    std::vector<float> h_b(NUM_ELEMENTS);
    std::vector<float> h_gold(NUM_ELEMENTS);

    for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
        h_a[i] = static_cast<float>(i % 500) * 1.5f + 0.5f;
        h_b[i] = static_cast<float>((i + 13) % 250) * 2.25f - 1.0f;
        h_gold[i] = h_a[i] + h_b[i];
    }

    res = cuMemcpyHtoD_v2(d_a, h_a.data(), BUFFER_SIZE);
    assert(res == CUDA_SUCCESS);

    res = cuMemcpyHtoD_v2(d_b, h_b.data(), BUFFER_SIZE);
    assert(res == CUDA_SUCCESS);
    std::cout << "[DRIVER_E2E] 8. Transferred input buffers via cuMemcpyHtoD_v2." << std::endl;

    // 7. Module Loading and Kernel Dispatch via cuLaunchKernel
    CUmodule mod = nullptr;
    res = cuModuleLoad(&mod, "build/shaders/vector_add.spv");
    assert(res == CUDA_SUCCESS && mod != nullptr);

    CUfunction fn = nullptr;
    res = cuModuleGetFunction(&fn, mod, "main");
    assert(res == CUDA_SUCCESS && fn != nullptr);

    struct KernelPushConstants {
        uint64_t a;
        uint64_t b;
        uint64_t c;
        uint32_t n;
    } pushParams;

    pushParams.a = static_cast<uint64_t>(d_a);
    pushParams.b = static_cast<uint64_t>(d_b);
    pushParams.c = static_cast<uint64_t>(d_c);
    pushParams.n = static_cast<uint32_t>(NUM_ELEMENTS);

    uint64_t p_a = pushParams.a;
    uint64_t p_b = pushParams.b;
    uint64_t p_c = pushParams.c;
    uint32_t p_n = pushParams.n;

    void* kernelArgs[] = { &p_a, &p_b, &p_c, &p_n };

    unsigned int blockSize = 256;
    unsigned int gridSize = (static_cast<unsigned int>(NUM_ELEMENTS) + blockSize - 1) / blockSize;

    std::cout << "[DRIVER_E2E] 9. Dispatching cuLaunchKernel with grid(" << gridSize << ",1,1) block(" << blockSize << ",1,1)..." << std::endl;
    res = cuLaunchKernel(
        fn,
        gridSize, 1, 1,
        blockSize, 1, 1,
        0,
        nullptr,
        kernelArgs,
        nullptr
    );
    assert(res == CUDA_SUCCESS);

    res = cuCtxSynchronize();
    assert(res == CUDA_SUCCESS);
    std::cout << "[DRIVER_E2E] 10. Kernel execution synchronized." << std::endl;

    // 8. Device to Host Memory Transfer via cuMemcpyDtoH_v2
    std::vector<float> h_c(NUM_ELEMENTS, 0.0f);
    res = cuMemcpyDtoH_v2(h_c.data(), d_c, BUFFER_SIZE);
    assert(res == CUDA_SUCCESS);

    // 9. Strict Bit-Level / Numerical Parity Verification
    float max_diff = 0.0f;
    for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
        float diff = std::abs(h_c[i] - h_gold[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-5f) {
            std::cerr << "Mismatch at index " << i << ": computed=" << h_c[i] << " gold=" << h_gold[i] << " diff=" << diff << std::endl;
            assert(false);
        }
    }
    std::cout << "[DRIVER_E2E] 11. Verified exact numerical parity (max_diff=" << max_diff << " across 1M elements)." << std::endl;

    // 10. Clean Driver Resource Teardown
    cuMemFree_v2(d_a);
    cuMemFree_v2(d_b);
    cuMemFree_v2(d_c);
    cuModuleUnload(mod);
    cuCtxDestroy_v2(ctx);

    std::cout << "============================================================" << std::endl;
    std::cout << " [E2E_CUDA_SUCCESS: DRIVER_DISPATCH_VERIFIED]               " << std::endl;
    std::cout << "============================================================" << std::endl;

    return 0;
}
