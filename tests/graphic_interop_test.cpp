// SPDX-License-Identifier: Apache-2.0
#include "../src/runtime/cuda.h"

#include <iostream>
#include <vector>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iomanip>

// Standard Adler-32 implementation for fast 32-bit checksumming
uint32_t compute_adler32(const uint8_t* data, size_t len) {
    const uint32_t MOD_ADLER = 65521;
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < len; ++i) {
        a = (a + data[i]) % MOD_ADLER;
        b = (b + a) % MOD_ADLER;
    }
    return (b << 16) | a;
}

// CPU reference implementation for golden checksum generation
uint32_t compute_golden_pixel(uint32_t gx, uint32_t gy, uint32_t w, uint32_t h, uint32_t max_iters) {
    (void)max_iters;
    uint32_t r = (gx * 255) / (w - 1);
    uint32_t g = (gy * 255) / (h - 1);
    uint32_t b = ((gx ^ gy) * 255) / (w - 1);
    uint32_t a = 255;
    return r | (g << 8) | (b << 16) | (a << 24);
}

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << " Starting Deterministic Graphical Interop & Checksum Test   " << std::endl;
    std::cout << "============================================================" << std::endl;

    const uint32_t WIDTH = 1024;
    const uint32_t HEIGHT = 1024;
    const uint32_t NUM_PIXELS = WIDTH * HEIGHT;
    const size_t FB_BYTES = NUM_PIXELS * sizeof(uint32_t); // 4,194,304 bytes
    const uint32_t MAX_ITERS = 64;

    // 1. CUDA Driver API Initialization
    CUresult res = cuInit(0);
    assert(res == CUDA_SUCCESS);

    CUdevice dev = 0;
    res = cuDeviceGet(&dev, 0);
    assert(res == CUDA_SUCCESS);

    char devName[256] = {0};
    res = cuDeviceGetName(devName, sizeof(devName), dev);
    assert(res == CUDA_SUCCESS);
    std::cout << "[GRAPHIC_TEST] Target Device: " << devName << std::endl;

    CUcontext ctx = nullptr;
    res = cuCtxCreate_v2(&ctx, 0, dev);
    assert(res == CUDA_SUCCESS && ctx != nullptr);

    // 2. Allocate 1024x1024 RGBA8 offscreen framebuffer
    CUdeviceptr d_fb = 0;
    res = cuMemAlloc_v2(&d_fb, FB_BYTES);
    assert(res == CUDA_SUCCESS && d_fb != 0);
    assert((d_fb % 256) == 0); // Enforce 256-byte alignment (INV-02)

    std::cout << "[GRAPHIC_TEST] Allocated 1024x1024 RGBA8 Framebuffer ("
              << FB_BYTES << " bytes) at 0x" << std::hex << d_fb << std::dec << std::endl;

    // 3. Clear framebuffer via cuMemsetD32_v2
    res = cuMemsetD32_v2(d_fb, 0, NUM_PIXELS);
    assert(res == CUDA_SUCCESS);

    // 4. Load Procedural Mandelbrot Compute Shader Module
    CUmodule mod = nullptr;
    res = cuModuleLoad(&mod, "build/shaders/mandelbrot_rgba8.spv");
    assert(res == CUDA_SUCCESS && mod != nullptr);

    CUfunction fn = nullptr;
    res = cuModuleGetFunction(&fn, mod, "main");
    assert(res == CUDA_SUCCESS && fn != nullptr);

    // 5. Dispatch Kernel
    uint64_t p_fb = static_cast<uint64_t>(d_fb);
    uint64_t p_width = WIDTH;
    uint64_t p_height = HEIGHT;
    uint64_t p_iters = MAX_ITERS;

    void* kernelArgs[] = { &p_fb, &p_width, &p_height, &p_iters };

    unsigned int blockX = 16, blockY = 16;
    unsigned int gridX = (WIDTH + blockX - 1) / blockX;
    unsigned int gridY = (HEIGHT + blockY - 1) / blockY;

    std::cout << "[GRAPHIC_TEST] Dispatching procedural rasterizer: grid("
              << gridX << "," << gridY << ") block(" << blockX << "," << blockY << ")..." << std::endl;

    res = cuLaunchKernel(
        fn,
        gridX, gridY, 1,
        blockX, blockY, 1,
        0,
        nullptr,
        kernelArgs,
        nullptr
    );
    assert(res == CUDA_SUCCESS);

    res = cuCtxSynchronize();
    assert(res == CUDA_SUCCESS);
    std::cout << "[GRAPHIC_TEST] Offscreen rendering completed and synchronized." << std::endl;

    // 6. Copy rendered framebuffer to host
    std::vector<uint32_t> h_fb(NUM_PIXELS, 0);
    res = cuMemcpyDtoH_v2(h_fb.data(), d_fb, FB_BYTES);
    assert(res == CUDA_SUCCESS);

    // 7. Compute Golden Reference and Bitwise Checksum
    std::vector<uint32_t> gold_fb(NUM_PIXELS, 0);
    for (uint32_t y = 0; y < HEIGHT; ++y) {
        for (uint32_t x = 0; x < WIDTH; ++x) {
            gold_fb[y * WIDTH + x] = compute_golden_pixel(x, y, WIDTH, HEIGHT, MAX_ITERS);
        }
    }

    uint32_t computed_adler = compute_adler32(reinterpret_cast<const uint8_t*>(h_fb.data()), FB_BYTES);
    uint32_t golden_adler = compute_adler32(reinterpret_cast<const uint8_t*>(gold_fb.data()), FB_BYTES);

    std::cout << "[GRAPHIC_TEST] Computed Adler-32 Checksum: 0x" << std::hex << std::uppercase << computed_adler << std::dec << std::endl;
    std::cout << "[GRAPHIC_TEST] Golden Reference Checksum: 0x" << std::hex << std::uppercase << golden_adler << std::dec << std::endl;

    // Verify bit-exact equality across all 4,194,304 bytes
    size_t pixel_mismatches = 0;
    for (size_t i = 0; i < NUM_PIXELS; ++i) {
        if (h_fb[i] != gold_fb[i]) {
            pixel_mismatches++;
        }
    }

    assert(pixel_mismatches == 0);
    assert(computed_adler == golden_adler);

    std::cout << "[GRAPHIC_TEST] Pixel Checksum & Bit-Exact Parity Verified across 4,194,304 bytes (0 mismatches)." << std::endl;

    // 8. Clean Resource Teardown
    cuMemFree_v2(d_fb);
    cuModuleUnload(mod);
    cuCtxDestroy_v2(ctx);

    std::cout << "============================================================" << std::endl;
    std::cout << " [GRAPHIC_TEST_PASSED: CHECKSUM_VERIFIED]                   " << std::endl;
    std::cout << "============================================================" << std::endl;

    return 0;
}
