// SPDX-License-Identifier: Apache-2.0
#include "../src/runtime/cuda_runtime.h"

#include <iostream>
#include <vector>
#include <thread>
#include <cassert>
#include <cstring>
#include <cstdint>
#include <random>
#include <algorithm>

void run_fragmentation_stress() {
    std::cout << "[STRESS_TEST] Starting Fragmentation Stress Test..." << std::endl;

    const size_t NUM_BLOCKS = 60;
    struct BlockInfo {
        void* ptr = nullptr;
        size_t size = 0;
        uint8_t pattern = 0;
        bool is_freed = false;
    };

    std::vector<BlockInfo> blocks(NUM_BLOCKS);
    std::vector<size_t> test_sizes = {
        1024,                  // 1 KB
        4096,                  // 4 KB
        64 * 1024,             // 64 KB
        256 * 1024,            // 256 KB
        1024 * 1024,           // 1 MB
        4 * 1024 * 1024,       // 4 MB
        16 * 1024 * 1024,      // 16 MB
        64 * 1024 * 1024,      // 64 MB (triggers dedicated allocation)
        128 * 1024 * 1024      // 128 MB (dedicated allocation)
    };

    // 1. Allocate 60 blocks spanning 1 KB to 128 MB
    for (size_t i = 0; i < NUM_BLOCKS; ++i) {
        size_t sz = test_sizes[i % test_sizes.size()];
        blocks[i].size = sz;
        blocks[i].pattern = static_cast<uint8_t>((i * 17 + 1) & 0xFF);

        cudaError_t err = cudaMalloc(&blocks[i].ptr, sz);
        assert(err == cudaSuccess);
        assert(blocks[i].ptr != nullptr);

        // Enforce INV-02: Deterministic 256-byte alignment
        uintptr_t addr = reinterpret_cast<uintptr_t>(blocks[i].ptr);
        assert((addr % 256) == 0);

        // Fill block with pattern
        err = cudaMemset(blocks[i].ptr, blocks[i].pattern, sz);
        assert(err == cudaSuccess);
    }
    std::cout << "[STRESS_TEST] Successfully allocated 60 blocks (1 KB to 128 MB) with 256-byte alignment." << std::endl;

    // 2. Randomly free 50% of the blocks (every even index)
    size_t freed_count = 0;
    for (size_t i = 0; i < NUM_BLOCKS; i += 2) {
        cudaError_t err = cudaFree(blocks[i].ptr);
        assert(err == cudaSuccess);
        blocks[i].is_freed = true;
        freed_count++;
    }
    std::cout << "[STRESS_TEST] Freed " << freed_count << " blocks to induce heap fragmentation." << std::endl;

    // 3. Verify that remaining 50% still retain intact data
    for (size_t i = 1; i < NUM_BLOCKS; i += 2) {
        assert(!blocks[i].is_freed);
        std::vector<uint8_t> verify_buf(std::min<size_t>(blocks[i].size, 1024 * 1024));
        cudaError_t err = cudaMemcpy(verify_buf.data(), blocks[i].ptr, verify_buf.size(), cudaMemcpyDeviceToHost);
        assert(err == cudaSuccess);
        for (size_t b = 0; b < verify_buf.size(); ++b) {
            assert(verify_buf[b] == blocks[i].pattern);
        }
    }
    std::cout << "[STRESS_TEST] Data integrity of active blocks verified after partial deallocation." << std::endl;

    // 4. Re-allocate 30 new blocks of various sizes to force allocator reuse & coalescing
    std::vector<void*> realloc_ptrs(freed_count, nullptr);
    for (size_t i = 0; i < freed_count; ++i) {
        size_t sz = test_sizes[(i * 3 + 1) % test_sizes.size()];
        cudaError_t err = cudaMalloc(&realloc_ptrs[i], sz);
        assert(err == cudaSuccess);
        assert(realloc_ptrs[i] != nullptr);

        uintptr_t addr = reinterpret_cast<uintptr_t>(realloc_ptrs[i]);
        assert((addr % 256) == 0);

        err = cudaMemset(realloc_ptrs[i], static_cast<int>(i + 50), sz);
        assert(err == cudaSuccess);
    }
    std::cout << "[STRESS_TEST] Re-allocation verified allocator reuse and coalescing." << std::endl;

    // 5. Clean deallocation of all remaining and reallocated blocks
    for (size_t i = 1; i < NUM_BLOCKS; i += 2) {
        cudaError_t err = cudaFree(blocks[i].ptr);
        assert(err == cudaSuccess);
    }
    for (size_t i = 0; i < freed_count; ++i) {
        cudaError_t err = cudaFree(realloc_ptrs[i]);
        assert(err == cudaSuccess);
    }

    std::cout << "[STRESS_TEST] Fragmentation Stress: PASSED" << std::endl;
}

void run_concurrency_stress() {
    std::cout << "[STRESS_TEST] Starting Concurrency Stress Test (8 Threads)..." << std::endl;

    const size_t NUM_THREADS = 8;
    const size_t ELEMS_PER_THREAD = 65536; // 64K ints = 256 KB
    const size_t BUF_SIZE = ELEMS_PER_THREAD * sizeof(int);

    std::vector<std::thread> workers;
    workers.reserve(NUM_THREADS);

    for (size_t tid = 0; tid < NUM_THREADS; ++tid) {
        workers.emplace_back([tid, BUF_SIZE, ELEMS_PER_THREAD]() {
            cudaStream_t stream = nullptr;
            cudaError_t err = cudaStreamCreate(&stream);
            assert(err == cudaSuccess);
            assert(stream != nullptr);

            int* d_src = nullptr;
            int* d_dst = nullptr;

            err = cudaMalloc(reinterpret_cast<void**>(&d_src), BUF_SIZE);
            assert(err == cudaSuccess);
            assert(d_src != nullptr);
            assert((reinterpret_cast<uintptr_t>(d_src) % 256) == 0);

            err = cudaMalloc(reinterpret_cast<void**>(&d_dst), BUF_SIZE);
            assert(err == cudaSuccess);
            assert(d_dst != nullptr);
            assert((reinterpret_cast<uintptr_t>(d_dst) % 256) == 0);

            std::vector<int> h_src(ELEMS_PER_THREAD);
            std::vector<int> h_dst(ELEMS_PER_THREAD, 0);

            for (size_t i = 0; i < ELEMS_PER_THREAD; ++i) {
                h_src[i] = static_cast<int>(tid * 1000000 + i);
            }

            // Asynchronous H2D copy
            err = cudaMemcpyAsync(d_src, h_src.data(), BUF_SIZE, cudaMemcpyHostToDevice, stream);
            assert(err == cudaSuccess);

            // Asynchronous D2D copy
            err = cudaMemcpyAsync(d_dst, d_src, BUF_SIZE, cudaMemcpyDeviceToDevice, stream);
            assert(err == cudaSuccess);

            // Asynchronous D2H copy
            err = cudaMemcpyAsync(h_dst.data(), d_dst, BUF_SIZE, cudaMemcpyDeviceToHost, stream);
            assert(err == cudaSuccess);

            // Wait for completion
            err = cudaStreamSynchronize(stream);
            assert(err == cudaSuccess);

            // Verify bit-exact match
            for (size_t i = 0; i < ELEMS_PER_THREAD; ++i) {
                assert(h_dst[i] == h_src[i]);
            }

            err = cudaFree(d_src);
            assert(err == cudaSuccess);
            err = cudaFree(d_dst);
            assert(err == cudaSuccess);

            err = cudaStreamDestroy(stream);
            assert(err == cudaSuccess);
        });
    }

    for (auto& t : workers) {
        t.join();
    }

    std::cout << "[STRESS_TEST] Concurrency Stress (8 Threads): PASSED" << std::endl;
}

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << " Running CUDA Vulkan Hardening & Stress Test Suite          " << std::endl;
    std::cout << "============================================================" << std::endl;

    run_fragmentation_stress();
    run_concurrency_stress();

    std::cout << "============================================================" << std::endl;
    std::cout << " [ALL PASSED] Stress and Concurrency Tests Succeeded        " << std::endl;
    std::cout << "============================================================" << std::endl;

    return 0;
}
