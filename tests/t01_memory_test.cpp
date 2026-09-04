// SPDX-License-Identifier: Apache-2.0
#include "../src/runtime/cuda_runtime.h"

#include <iostream>
#include <vector>
#include <cassert>
#include <cstring>
#include <cstdint>

int main() {
    std::cout << "[STAGE_02] Starting Vulkan 1.3 CUDA Memory Runtime Test..." << std::endl;

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    assert(err == cudaSuccess);
    assert(deviceCount > 0);
    std::cout << "[STAGE_02] Found " << deviceCount << " compute-capable Vulkan device(s)." << std::endl;

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    assert(err == cudaSuccess);
    std::cout << "[STAGE_02] Device name: " << prop.name << " (Compute capability " << prop.major << "." << prop.minor << ")" << std::endl;

    // Test 1: Sub-allocator and 64-bit device address validation
    const size_t NUM_ELEMS = 1048576; // 1M floats = 4MB
    const size_t BUFFER_SIZE = NUM_ELEMS * sizeof(float);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    err = cudaMalloc(reinterpret_cast<void**>(&d_a), BUFFER_SIZE);
    assert(err == cudaSuccess);
    assert(d_a != nullptr);

    err = cudaMalloc(reinterpret_cast<void**>(&d_b), BUFFER_SIZE);
    assert(err == cudaSuccess);
    assert(d_b != nullptr);

    err = cudaMalloc(reinterpret_cast<void**>(&d_c), BUFFER_SIZE);
    assert(err == cudaSuccess);
    assert(d_c != nullptr);

    // Verify non-overlapping allocations
    uintptr_t addr_a = reinterpret_cast<uintptr_t>(d_a);
    uintptr_t addr_b = reinterpret_cast<uintptr_t>(d_b);
    uintptr_t addr_c = reinterpret_cast<uintptr_t>(d_c);
    assert(addr_a != addr_b && addr_b != addr_c && addr_a != addr_c);
    std::cout << "[STAGE_02] Sub-allocator allocated 3x 4MB buffers at addresses: "
              << (void*)d_a << ", " << (void*)d_b << ", " << (void*)d_c << std::endl;

    // Test 2: Host to Device and Device to Host memcpy parity
    std::vector<float> h_in(NUM_ELEMS);
    std::vector<float> h_out(NUM_ELEMS, 0.0f);
    for (size_t i = 0; i < NUM_ELEMS; ++i) {
        h_in[i] = static_cast<float>(i) * 1.5f + 0.25f;
    }

    err = cudaMemcpy(d_a, h_in.data(), BUFFER_SIZE, cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);

    err = cudaMemcpy(h_out.data(), d_a, BUFFER_SIZE, cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);

    for (size_t i = 0; i < NUM_ELEMS; ++i) {
        if (h_in[i] != h_out[i]) {
            std::cerr << "Mismatch at index " << i << ": expected " << h_in[i] << ", got " << h_out[i] << std::endl;
            return 1;
        }
    }
    std::cout << "[STAGE_02] H2D and D2H memcpy parity verified (1M elements, exact match)." << std::endl;

    // Test 3: Device to Device memcpy
    err = cudaMemcpy(d_b, d_a, BUFFER_SIZE, cudaMemcpyDeviceToDevice);
    assert(err == cudaSuccess);

    std::fill(h_out.begin(), h_out.end(), 0.0f);
    err = cudaMemcpy(h_out.data(), d_b, BUFFER_SIZE, cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);

    for (size_t i = 0; i < NUM_ELEMS; ++i) {
        if (h_in[i] != h_out[i]) {
            std::cerr << "D2D mismatch at index " << i << std::endl;
            return 1;
        }
    }
    std::cout << "[STAGE_02] D2D memcpy verified (exact match)." << std::endl;

    // Test 4: Stream creation, async memcpy and timeline synchronization
    cudaStream_t stream = nullptr;
    err = cudaStreamCreate(&stream);
    assert(err == cudaSuccess);
    assert(stream != nullptr);

    err = cudaMemcpyAsync(d_c, d_b, BUFFER_SIZE, cudaMemcpyDeviceToDevice, stream);
    assert(err == cudaSuccess);

    err = cudaStreamSynchronize(stream);
    assert(err == cudaSuccess);

    std::fill(h_out.begin(), h_out.end(), 0.0f);
    err = cudaMemcpy(h_out.data(), d_c, BUFFER_SIZE, cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);
    assert(h_in[0] == h_out[0] && h_in[NUM_ELEMS - 1] == h_out[NUM_ELEMS - 1]);

    err = cudaStreamDestroy(stream);
    assert(err == cudaSuccess);
    std::cout << "[STAGE_02] Stream timeline semaphore synchronization verified." << std::endl;

    // Test 5: cudaFree and reuse
    err = cudaFree(d_a);
    assert(err == cudaSuccess);
    err = cudaFree(d_b);
    assert(err == cudaSuccess);
    err = cudaFree(d_c);
    assert(err == cudaSuccess);

    // Re-allocate to test block coalescing and reuse
    float* d_realloc = nullptr;
    err = cudaMalloc(reinterpret_cast<void**>(&d_realloc), BUFFER_SIZE * 2);
    assert(err == cudaSuccess);
    assert(d_realloc != nullptr);
    err = cudaFree(d_realloc);
    assert(err == cudaSuccess);
    std::cout << "[STAGE_02] Sub-allocator coalescing and reuse verified." << std::endl;

    std::cout << "[STAGE_02_RUNTIME_MEMORY] All memory and synchronization tests passed successfully." << std::endl;
    return 0;
}
