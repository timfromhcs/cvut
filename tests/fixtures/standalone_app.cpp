// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>

int main() {
    std::cout << "[STANDALONE_APP] Starting standalone external verification..." << std::endl;

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount <= 0) {
        std::cerr << "Failed to query devices: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        std::cerr << "Failed to query properties: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "[STANDALONE_APP] Target GPU: " << prop.name << " (Compute " << prop.major << "." << prop.minor << ")" << std::endl;

    const size_t NUM_ELEMENTS = 4096;
    const size_t BUFFER_SIZE = NUM_ELEMENTS * sizeof(float);

    float* d_buf = nullptr;
    err = cudaMalloc(reinterpret_cast<void**>(&d_buf), BUFFER_SIZE);
    if (err != cudaSuccess || !d_buf) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Verify 256-byte alignment
    uintptr_t addr = reinterpret_cast<uintptr_t>(d_buf);
    assert((addr % 256) == 0);

    std::vector<float> h_in(NUM_ELEMENTS);
    std::vector<float> h_out(NUM_ELEMENTS, 0.0f);
    for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
        h_in[i] = static_cast<float>(i) * 3.14159f;
    }

    err = cudaMemcpy(d_buf, h_in.data(), BUFFER_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "H2D memcpy failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    err = cudaMemcpy(h_out.data(), d_buf, BUFFER_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "D2H memcpy failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
        if (h_out[i] != h_in[i]) {
            std::cerr << "Data mismatch at index " << i << std::endl;
            return 1;
        }
    }

    err = cudaFree(d_buf);
    if (err != cudaSuccess) {
        std::cerr << "cudaFree failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "[STANDALONE_APP] SUCCESS: All standalone operations validated." << std::endl;
    return 0;
}
