// SPDX-License-Identifier: Apache-2.0
#include "../src/runtime/cuda_runtime.h"

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstring>
#include <cassert>
#include <cstdint>

// Utility for float32 <-> float16 conversion (IEEE 754-2008)
uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xff) - 127;
    uint32_t mant = x & 0x7fffff;

    if (exp > 15) {
        return static_cast<uint16_t>(sign | 0x7c00); // Inf
    } else if (exp < -14) {
        return static_cast<uint16_t>(sign); // Denorm or 0
    } else {
        exp += 15;
        mant >>= 13;
        return static_cast<uint16_t>(sign | (exp << 10) | mant);
    }
}

float half_to_float(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000) << 16;
    int32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;

    if (exp == 0x1f) {
        exp = 255;
    } else if (exp == 0) {
        exp = 0;
    } else {
        exp = exp - 15 + 127;
    }
    mant <<= 13;
    uint32_t res = sign | (static_cast<uint32_t>(exp) << 23) | mant;
    float f;
    std::memcpy(&f, &res, 4);
    return f;
}

int main(int argc, char** argv) {
    std::string test_case = "";
    size_t elements = 1048576;
    size_t dim = 2048;
    size_t m = 1024, n = 1024, k = 1024;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--case=", 0) == 0) {
            test_case = arg.substr(7);
        } else if (arg.rfind("--elements=", 0) == 0) {
            elements = std::stoull(arg.substr(11));
        } else if (arg.rfind("--dim=", 0) == 0) {
            dim = std::stoull(arg.substr(6));
        } else if (arg.rfind("--m=", 0) == 0) {
            m = std::stoull(arg.substr(4));
        } else if (arg.rfind("--n=", 0) == 0) {
            n = std::stoull(arg.substr(4));
        } else if (arg.rfind("--k=", 0) == 0) {
            k = std::stoull(arg.substr(4));
        }
    }

    if (test_case == "vector_add") {
        size_t buffer_size = elements * sizeof(float);
        float* d_a = nullptr;
        float* d_b = nullptr;
        float* d_c = nullptr;

        cudaMalloc(reinterpret_cast<void**>(&d_a), buffer_size);
        cudaMalloc(reinterpret_cast<void**>(&d_b), buffer_size);
        cudaMalloc(reinterpret_cast<void**>(&d_c), buffer_size);

        std::vector<float> h_a(elements);
        std::vector<float> h_b(elements);
        std::vector<float> h_gold(elements);

        for (size_t i = 0; i < elements; ++i) {
            h_a[i] = static_cast<float>(i % 1000) * 0.5f;
            h_b[i] = static_cast<float>((i + 7) % 500) * 0.25f;
            h_gold[i] = h_a[i] + h_b[i];
        }

        cudaMemcpy(d_a, h_a.data(), buffer_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), buffer_size, cudaMemcpyHostToDevice);

        struct PushParams {
            uint64_t a;
            uint64_t b;
            uint64_t c;
            uint32_t n;
        } push;
        push.a = reinterpret_cast<uint64_t>(d_a);
        push.b = reinterpret_cast<uint64_t>(d_b);
        push.c = reinterpret_cast<uint64_t>(d_c);
        push.n = static_cast<uint32_t>(elements);

        dim3 grid(static_cast<unsigned int>((elements + 255) / 256), 1, 1);
        dim3 block(256, 1, 1);

        cudaError_t launch_err = cudaLaunchSpirv("build/shaders/vector_add.spv", grid, block, &push, sizeof(push), nullptr);
        if (launch_err != cudaSuccess) {
            std::cerr << "cudaLaunchSpirv failed: " << cudaGetErrorString(launch_err) << std::endl;
        }
        cudaDeviceSynchronize();

        std::vector<float> h_c(elements, 0.0f);
        cudaMemcpy(h_c.data(), d_c, buffer_size, cudaMemcpyDeviceToHost);

        float max_diff = 0.0f;
        for (size_t i = 0; i < elements; ++i) {
            float diff = std::abs(h_c[i] - h_gold[i]);
            if (diff > max_diff) max_diff = diff;
        }

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        if (max_diff == 0.0f) {
            std::cout << "TEST_PASSED: EPSILON=0.000000 CHECKSUM_MATCH" << std::endl;
            return 0;
        } else {
            std::cerr << "Vector add mismatch: max_diff = " << max_diff << std::endl;
            return 1;
        }
    } else if (test_case == "matrix_transpose") {
        size_t total_elements = dim * dim;
        size_t buffer_size = total_elements * sizeof(float);

        float* d_in = nullptr;
        float* d_out = nullptr;

        cudaMalloc(reinterpret_cast<void**>(&d_in), buffer_size);
        cudaMalloc(reinterpret_cast<void**>(&d_out), buffer_size);

        std::vector<float> h_in(total_elements);
        std::vector<float> h_gold(total_elements);

        for (size_t r = 0; r < dim; ++r) {
            for (size_t c = 0; c < dim; ++c) {
                float val = static_cast<float>((r * 37 + c * 19) % 10000);
                h_in[r * dim + c] = val;
                h_gold[c * dim + r] = val;
            }
        }

        cudaMemcpy(d_in, h_in.data(), buffer_size, cudaMemcpyHostToDevice);

        struct PushConstants {
            uint64_t in_buf;
            uint64_t out_buf;
            uint32_t dim;
        } push;
        push.in_buf = reinterpret_cast<uint64_t>(d_in);
        push.out_buf = reinterpret_cast<uint64_t>(d_out);
        push.dim = static_cast<uint32_t>(dim);

        dim3 grid(static_cast<unsigned int>((dim + 31) / 32), static_cast<unsigned int>((dim + 31) / 32), 1);
        dim3 block(32, 8, 1);

        cudaLaunchSpirv("build/shaders/matrix_transpose.spv", grid, block, &push, sizeof(push), nullptr);
        cudaDeviceSynchronize();

        std::vector<float> h_out(total_elements, 0.0f);
        cudaMemcpy(h_out.data(), d_out, buffer_size, cudaMemcpyDeviceToHost);

        uint64_t bit_diff = 0;
        for (size_t i = 0; i < total_elements; ++i) {
            uint32_t actual_bits, expected_bits;
            std::memcpy(&actual_bits, &h_out[i], 4);
            std::memcpy(&expected_bits, &h_gold[i], 4);
            if (actual_bits != expected_bits) {
                bit_diff++;
            }
        }

        cudaFree(d_in);
        cudaFree(d_out);

        if (bit_diff == 0) {
            std::cout << "TEST_PASSED: TRANSPOSE_EXACT BIT_DIFF=0" << std::endl;
            return 0;
        } else {
            std::cerr << "Transpose mismatch, bit differences: " << bit_diff << std::endl;
            return 1;
        }
    } else if (test_case == "gemm_fp16") {
        size_t size_a = m * k;
        size_t size_b = k * n;
        size_t size_c = m * n;

        void* d_a = nullptr;
        void* d_b = nullptr;
        void* d_c = nullptr;

        cudaMalloc(&d_a, size_a * sizeof(uint16_t));
        cudaMalloc(&d_b, size_b * sizeof(uint16_t));
        cudaMalloc(&d_c, size_c * sizeof(uint16_t));

        std::vector<uint16_t> h_a_fp16(size_a);
        std::vector<uint16_t> h_b_fp16(size_b);
        std::vector<float> h_a_f32(size_a);
        std::vector<float> h_b_f32(size_b);

        for (size_t i = 0; i < size_a; ++i) {
            float val = static_cast<float>((i % 17) - 8) * 0.125f;
            h_a_f32[i] = val;
            h_a_fp16[i] = float_to_half(val);
        }
        for (size_t i = 0; i < size_b; ++i) {
            float val = static_cast<float>((i % 13) - 6) * 0.125f;
            h_b_f32[i] = val;
            h_b_fp16[i] = float_to_half(val);
        }

        cudaMemcpy(d_a, h_a_fp16.data(), size_a * sizeof(uint16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b_fp16.data(), size_b * sizeof(uint16_t), cudaMemcpyHostToDevice);

        struct PushConstants {
            uint64_t a;
            uint64_t b;
            uint64_t c;
            uint32_t m;
            uint32_t n;
            uint32_t k;
        } push;
        push.a = reinterpret_cast<uint64_t>(d_a);
        push.b = reinterpret_cast<uint64_t>(d_b);
        push.c = reinterpret_cast<uint64_t>(d_c);
        push.m = static_cast<uint32_t>(m);
        push.n = static_cast<uint32_t>(n);
        push.k = static_cast<uint32_t>(k);

        dim3 grid(static_cast<unsigned int>((n + 15) / 16), static_cast<unsigned int>((m + 15) / 16), 1);
        dim3 block(16, 16, 1);

        cudaLaunchSpirv("build/shaders/gemm_fp16.spv", grid, block, &push, sizeof(push), nullptr);
        cudaDeviceSynchronize();

        std::vector<uint16_t> h_c_fp16(size_c, 0);
        cudaMemcpy(h_c_fp16.data(), d_c, size_c * sizeof(uint16_t), cudaMemcpyDeviceToHost);

        // Verify sampled elements against CPU baseline
        float max_rel_diff = 0.0f;
        const size_t check_samples = 512;
        for (size_t s = 0; s < check_samples; ++s) {
            size_t row = (s * 31) % m;
            size_t col = (s * 47) % n;
            float cpu_acc = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                cpu_acc += h_a_f32[row * k + p] * h_b_f32[p * n + col];
            }
            float gpu_val = half_to_float(h_c_fp16[row * n + col]);
            float abs_err = std::abs(gpu_val - cpu_acc);
            float rel_err = abs_err / (std::abs(cpu_acc) + 1e-4f);
            if (rel_err > max_rel_diff) max_rel_diff = rel_err;
        }

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        if (max_rel_diff < 1e-3f) {
            std::cout << "TEST_PASSED: MAX_REL_DIFF<1e-3" << std::endl;
            return 0;
        } else {
            std::cerr << "GEMM relative difference too large: " << max_rel_diff << std::endl;
            return 1;
        }
    } else {
        std::cerr << "Unknown case: " << test_case << std::endl;
        return 1;
    }
}
