// SPDX-License-Identifier: Apache-2.0
#include "../src/runtime/cuda.h"
#include "../src/runtime/cuda_runtime.h"

#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <iomanip>
#include <cmath>
#include <cassert>

using Clock = std::chrono::high_resolution_clock;

struct BenchResult {
    std::string name;
    double avg_ms;
    double min_ms;
    double throughput; // GB/s or GFLOP/s or ops/s
    std::string unit;
};

void run_initialization_benchmark(std::vector<BenchResult>& results) {
    std::cout << "[BENCHMARK] 1. Measuring Initialization Overhead..." << std::endl;

    auto t0 = Clock::now();
    CUresult res = cuInit(0);
    auto t1 = Clock::now();
    assert(res == CUDA_SUCCESS);
    double init_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    CUdevice dev = 0;
    cuDeviceGet(&dev, 0);

    t0 = Clock::now();
    CUcontext ctx = nullptr;
    res = cuCtxCreate_v2(&ctx, 0, dev);
    t1 = Clock::now();
    assert(res == CUDA_SUCCESS);
    double ctx_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    results.push_back({"cuInit", init_ms, init_ms, 1000.0 / (init_ms > 0 ? init_ms : 0.001), "calls/s"});
    results.push_back({"cuCtxCreate", ctx_ms, ctx_ms, 1000.0 / (ctx_ms > 0 ? ctx_ms : 0.001), "contexts/s"});

    std::cout << "  - cuInit latency: " << std::fixed << std::setprecision(3) << init_ms << " ms" << std::endl;
    std::cout << "  - cuCtxCreate latency: " << ctx_ms << " ms" << std::endl;

    cuCtxDestroy_v2(ctx);
}

void run_memory_benchmark(std::vector<BenchResult>& results) {
    std::cout << "[BENCHMARK] 2. Measuring Memory Allocation & Transfer Overhead..." << std::endl;

    const size_t ALLOC_ITERS = 200;
    const size_t ALLOC_SIZE = 1024 * 1024; // 1 MB

    // Allocation throughput
    std::vector<void*> ptrs(ALLOC_ITERS, nullptr);
    auto t0 = Clock::now();
    for (size_t i = 0; i < ALLOC_ITERS; ++i) {
        cudaMalloc(&ptrs[i], ALLOC_SIZE);
    }
    auto t1 = Clock::now();
    double alloc_total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double alloc_ops_sec = (ALLOC_ITERS / (alloc_total_ms / 1000.0));

    t0 = Clock::now();
    for (size_t i = 0; i < ALLOC_ITERS; ++i) {
        cudaFree(ptrs[i]);
    }
    t1 = Clock::now();
    double free_total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double free_ops_sec = (ALLOC_ITERS / (free_total_ms / 1000.0));

    results.push_back({"cudaMalloc (1MB)", alloc_total_ms / ALLOC_ITERS, alloc_total_ms / ALLOC_ITERS, alloc_ops_sec, "allocs/s"});
    results.push_back({"cudaFree", free_total_ms / ALLOC_ITERS, free_total_ms / ALLOC_ITERS, free_ops_sec, "frees/s"});

    std::cout << "  - cudaMalloc throughput: " << std::fixed << std::setprecision(0) << alloc_ops_sec << " allocs/s ("
              << std::setprecision(3) << alloc_total_ms / ALLOC_ITERS << " ms/alloc)" << std::endl;
    std::cout << "  - cudaFree throughput: " << free_ops_sec << " frees/s ("
              << free_total_ms / ALLOC_ITERS << " ms/free)" << std::endl;

    // Memcpy Bandwidth (16 MB buffer, 20 iterations)
    const size_t TRANSFER_SIZE = 16 * 1024 * 1024; // 16 MB
    const size_t WARMUP = 2;
    const size_t BENCH_ITERS = 20;

    void* d_buf = nullptr;
    cudaMalloc(&d_buf, TRANSFER_SIZE);
    std::vector<uint8_t> h_buf(TRANSFER_SIZE, 0xAB);

    // Warmup
    for (size_t i = 0; i < WARMUP; ++i) {
        cudaMemcpy(d_buf, h_buf.data(), TRANSFER_SIZE, cudaMemcpyHostToDevice);
    }

    // H2D
    std::vector<double> h2d_times;
    for (size_t i = 0; i < BENCH_ITERS; ++i) {
        t0 = Clock::now();
        cudaMemcpy(d_buf, h_buf.data(), TRANSFER_SIZE, cudaMemcpyHostToDevice);
        t1 = Clock::now();
        h2d_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    double h2d_avg_ms = std::accumulate(h2d_times.begin(), h2d_times.end(), 0.0) / BENCH_ITERS;
    double h2d_bw = (TRANSFER_SIZE / (h2d_avg_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);

    // D2H
    std::vector<double> d2h_times;
    for (size_t i = 0; i < BENCH_ITERS; ++i) {
        t0 = Clock::now();
        cudaMemcpy(h_buf.data(), d_buf, TRANSFER_SIZE, cudaMemcpyDeviceToHost);
        t1 = Clock::now();
        d2h_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    double d2h_avg_ms = std::accumulate(d2h_times.begin(), d2h_times.end(), 0.0) / BENCH_ITERS;
    double d2h_bw = (TRANSFER_SIZE / (d2h_avg_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);

    // D2D
    void* d_buf2 = nullptr;
    cudaMalloc(&d_buf2, TRANSFER_SIZE);
    std::vector<double> d2d_times;
    for (size_t i = 0; i < BENCH_ITERS; ++i) {
        t0 = Clock::now();
        cudaMemcpy(d_buf2, d_buf, TRANSFER_SIZE, cudaMemcpyDeviceToDevice);
        t1 = Clock::now();
        d2d_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    double d2d_avg_ms = std::accumulate(d2d_times.begin(), d2d_times.end(), 0.0) / BENCH_ITERS;
    double d2d_bw = (TRANSFER_SIZE / (d2d_avg_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);

    cudaFree(d_buf);
    cudaFree(d_buf2);

    results.push_back({"H2D Bandwidth (16MB)", h2d_avg_ms, h2d_avg_ms, h2d_bw, "GB/s"});
    results.push_back({"D2H Bandwidth (16MB)", d2h_avg_ms, d2h_avg_ms, d2h_bw, "GB/s"});
    results.push_back({"D2D Bandwidth (16MB)", d2d_avg_ms, d2d_avg_ms, d2d_bw, "GB/s"});

    std::cout << "  - H2D Bandwidth: " << std::fixed << std::setprecision(2) << h2d_bw << " GB/s (" << h2d_avg_ms << " ms)" << std::endl;
    std::cout << "  - D2H Bandwidth: " << d2h_bw << " GB/s (" << d2h_avg_ms << " ms)" << std::endl;
    std::cout << "  - D2D Bandwidth: " << d2d_bw << " GB/s (" << d2d_avg_ms << " ms)" << std::endl;
}

void run_compute_benchmark(std::vector<BenchResult>& results) {
    std::cout << "[BENCHMARK] 3. Measuring Compute Kernel Performance..." << std::endl;

    // 1. Vector Add (1M elements)
    const size_t VEC_ELEMS = 1048576;
    const size_t VEC_BYTES = VEC_ELEMS * sizeof(float);
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&d_a), VEC_BYTES);
    cudaMalloc(reinterpret_cast<void**>(&d_b), VEC_BYTES);
    cudaMalloc(reinterpret_cast<void**>(&d_c), VEC_BYTES);

    struct PushParams {
        uint64_t a, b, c;
        uint32_t n;
    } vecPush;
    vecPush.a = reinterpret_cast<uint64_t>(d_a);
    vecPush.b = reinterpret_cast<uint64_t>(d_b);
    vecPush.c = reinterpret_cast<uint64_t>(d_c);
    vecPush.n = static_cast<uint32_t>(VEC_ELEMS);

    dim3 grid((static_cast<unsigned int>(VEC_ELEMS) + 255) / 256, 1, 1);
    dim3 block(256, 1, 1);

    // Warmup
    for (int i = 0; i < 3; ++i) {
        cudaLaunchSpirv("build/shaders/vector_add.spv", grid, block, &vecPush, sizeof(vecPush), nullptr);
        cudaDeviceSynchronize();
    }

    const size_t KERNEL_ITERS = 30;
    std::vector<double> vec_times;
    for (size_t i = 0; i < KERNEL_ITERS; ++i) {
        auto t0 = Clock::now();
        cudaLaunchSpirv("build/shaders/vector_add.spv", grid, block, &vecPush, sizeof(vecPush), nullptr);
        cudaDeviceSynchronize();
        auto t1 = Clock::now();
        vec_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    double vec_avg_ms = std::accumulate(vec_times.begin(), vec_times.end(), 0.0) / KERNEL_ITERS;
    // 1 FLOP per element (addition)
    double vec_gflops = (VEC_ELEMS / (vec_avg_ms / 1000.0)) / 1e9;
    // 3 memory accesses (2 reads, 1 write) = 3 * 4 bytes = 12 bytes/element
    double vec_eff_bw = (VEC_ELEMS * 12.0 / (vec_avg_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    results.push_back({"vector_add (1M floats)", vec_avg_ms, vec_avg_ms, vec_gflops, "GFLOP/s"});
    results.push_back({"vector_add Memory BW", vec_avg_ms, vec_avg_ms, vec_eff_bw, "GB/s"});

    std::cout << "  - vector_add: " << std::fixed << std::setprecision(3) << vec_avg_ms << " ms | "
              << std::setprecision(2) << vec_gflops << " GFLOP/s | " << vec_eff_bw << " GB/s" << std::endl;

    // 2. Matrix Transpose (2048x2048)
    const size_t DIM = 2048;
    const size_t MAT_ELEMS = DIM * DIM;
    const size_t MAT_BYTES = MAT_ELEMS * sizeof(float);
    float* d_in = nullptr;
    float* d_out = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&d_in), MAT_BYTES);
    cudaMalloc(reinterpret_cast<void**>(&d_out), MAT_BYTES);

    struct TransposePush {
        uint64_t in_buf, out_buf;
        uint32_t dim;
    } matPush;
    matPush.in_buf = reinterpret_cast<uint64_t>(d_in);
    matPush.out_buf = reinterpret_cast<uint64_t>(d_out);
    matPush.dim = static_cast<uint32_t>(DIM);

    dim3 matGrid((static_cast<unsigned int>(DIM) + 31) / 32, (static_cast<unsigned int>(DIM) + 31) / 32, 1);
    dim3 matBlock(32, 8, 1);

    for (int i = 0; i < 3; ++i) {
        cudaLaunchSpirv("build/shaders/matrix_transpose.spv", matGrid, matBlock, &matPush, sizeof(matPush), nullptr);
        cudaDeviceSynchronize();
    }

    std::vector<double> trans_times;
    for (size_t i = 0; i < KERNEL_ITERS; ++i) {
        auto t0 = Clock::now();
        cudaLaunchSpirv("build/shaders/matrix_transpose.spv", matGrid, matBlock, &matPush, sizeof(matPush), nullptr);
        cudaDeviceSynchronize();
        auto t1 = Clock::now();
        trans_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    double trans_avg_ms = std::accumulate(trans_times.begin(), trans_times.end(), 0.0) / KERNEL_ITERS;
    // 2 accesses per element (1 read, 1 write) = 8 bytes/element
    double trans_bw = (MAT_ELEMS * 8.0 / (trans_avg_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);

    cudaFree(d_in);
    cudaFree(d_out);

    results.push_back({"matrix_transpose (2048x2048)", trans_avg_ms, trans_avg_ms, trans_bw, "GB/s"});
    std::cout << "  - matrix_transpose: " << std::fixed << std::setprecision(3) << trans_avg_ms << " ms | "
              << std::setprecision(2) << trans_bw << " GB/s" << std::endl;

    // 3. GEMM FP16 (512x512x512)
    const size_t M = 512, N = 512, K = 512;
    void *d_ga = nullptr, *d_gb = nullptr, *d_gc = nullptr;
    cudaMalloc(&d_ga, M * K * sizeof(uint16_t));
    cudaMalloc(&d_gb, K * N * sizeof(uint16_t));
    cudaMalloc(&d_gc, M * N * sizeof(uint16_t));

    struct GemmPush {
        uint64_t a, b, c;
        uint32_t m, n, k;
    } gemmPush;
    gemmPush.a = reinterpret_cast<uint64_t>(d_ga);
    gemmPush.b = reinterpret_cast<uint64_t>(d_gb);
    gemmPush.c = reinterpret_cast<uint64_t>(d_gc);
    gemmPush.m = static_cast<uint32_t>(M);
    gemmPush.n = static_cast<uint32_t>(N);
    gemmPush.k = static_cast<uint32_t>(K);

    dim3 gemmGrid((static_cast<unsigned int>(N) + 15) / 16, (static_cast<unsigned int>(M) + 15) / 16, 1);
    dim3 gemmBlock(16, 16, 1);

    for (int i = 0; i < 3; ++i) {
        cudaLaunchSpirv("build/shaders/gemm_fp16.spv", gemmGrid, gemmBlock, &gemmPush, sizeof(gemmPush), nullptr);
        cudaDeviceSynchronize();
    }

    std::vector<double> gemm_times;
    for (size_t i = 0; i < KERNEL_ITERS; ++i) {
        auto t0 = Clock::now();
        cudaLaunchSpirv("build/shaders/gemm_fp16.spv", gemmGrid, gemmBlock, &gemmPush, sizeof(gemmPush), nullptr);
        cudaDeviceSynchronize();
        auto t1 = Clock::now();
        gemm_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    double gemm_avg_ms = std::accumulate(gemm_times.begin(), gemm_times.end(), 0.0) / KERNEL_ITERS;
    // 2 * M * N * K FLOPs
    double gemm_gflops = (2.0 * M * N * K / (gemm_avg_ms / 1000.0)) / 1e9;

    cudaFree(d_ga);
    cudaFree(d_gb);
    cudaFree(d_gc);

    results.push_back({"gemm_fp16 (512x512x512)", gemm_avg_ms, gemm_avg_ms, gemm_gflops, "GFLOP/s"});
    std::cout << "  - gemm_fp16: " << std::fixed << std::setprecision(3) << gemm_avg_ms << " ms | "
              << std::setprecision(2) << gemm_gflops << " GFLOP/s" << std::endl;
}

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << " CVUT Hardware Benchmark Suite                             " << std::endl;
    std::cout << "============================================================" << std::endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MiB" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    std::vector<BenchResult> results;
    run_initialization_benchmark(results);
    run_memory_benchmark(results);
    run_compute_benchmark(results);

    std::cout << "============================================================" << std::endl;
    std::cout << " Benchmark Results Summary                                  " << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << std::left << std::setw(30) << "Benchmark"
              << std::right << std::setw(14) << "Avg Latency"
              << std::setw(18) << "Throughput" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    for (const auto& r : results) {
        std::cout << std::left << std::setw(30) << r.name
                  << std::right << std::fixed << std::setprecision(3) << std::setw(10) << r.avg_ms << " ms"
                  << std::setprecision(2) << std::setw(11) << r.throughput << " " << r.unit << std::endl;
    }

    std::cout << "============================================================" << std::endl;
    return 0;
}
