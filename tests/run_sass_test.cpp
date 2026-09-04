// SPDX-License-Identifier: Apache-2.0
#include "../src/runtime/cuda_runtime.h"

#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cassert>
#include <numeric>

int main(int argc, char** argv) {
    std::string cubin_path = "";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--cubin=", 0) == 0) {
            cubin_path = arg.substr(8);
        }
    }

    if (cubin_path.empty()) {
        std::cerr << "Usage: run_sass_test --cubin=<path>" << std::endl;
        return 1;
    }

    std::string lifted_spv = "build/shaders/reduction_lifted.spv";
    std::string lifter_cmd;
#ifdef _WIN32
    lifter_cmd = ".\\build\\bin\\sass_lifter.exe --input " + cubin_path + " --output " + lifted_spv;
    std::string val_cmd = "spirv-val " + lifted_spv + " >nul 2>&1";
#else
    lifter_cmd = "./build/bin/sass_lifter --input " + cubin_path + " --output " + lifted_spv;
    std::string val_cmd = "spirv-val " + lifted_spv + " >/dev/null 2>&1";
#endif

    int ret = std::system(lifter_cmd.c_str());
    if (ret != 0) {
        // Fallback to cargo release target path
#ifdef _WIN32
        lifter_cmd = ".\\src\\lifter\\target\\release\\sass_lifter.exe --input " + cubin_path + " --output " + lifted_spv;
#else
        lifter_cmd = "./src/lifter/target/release/sass_lifter --input " + cubin_path + " --output " + lifted_spv;
#endif
        ret = std::system(lifter_cmd.c_str());
    }
    if (ret != 0) {
        lifter_cmd = "sass_lifter --input " + cubin_path + " --output " + lifted_spv;
        ret = std::system(lifter_cmd.c_str());
    }
    assert(ret == 0);

    // Validate lifted SPIR-V
    ret = std::system(val_cmd.c_str());
    assert(ret == 0);

    // Execute reduction on GPU using Vulkan 1.3 runtime
    const size_t NUM_ELEMENTS = 256;
    float* d_input = nullptr;
    float* d_output = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&d_input), NUM_ELEMENTS * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_output), sizeof(float));

    std::vector<float> h_input(NUM_ELEMENTS, 1.0f);
    cudaMemcpy(d_input, h_input.data(), NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);

    struct PushConstants {
        uint64_t input_data;
        uint64_t output_data;
        uint32_t n;
    } push;
    push.input_data = reinterpret_cast<uint64_t>(d_input);
    push.output_data = reinterpret_cast<uint64_t>(d_output);
    push.n = static_cast<uint32_t>(NUM_ELEMENTS);

    cudaLaunchSpirv(lifted_spv.c_str(), dim3(1, 1, 1), dim3(256, 1, 1), &push, sizeof(push), nullptr);
    cudaDeviceSynchronize();

    float h_result = 0.0f;
    cudaMemcpy(&h_result, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    if (h_result == static_cast<float>(NUM_ELEMENTS)) {
        std::cout << "TEST_PASSED: SASS_EXECUTION_VALIDATED" << std::endl;
        return 0;
    } else {
        std::cerr << "SASS execution mismatch: got " << h_result << ", expected " << NUM_ELEMENTS << std::endl;
        return 1;
    }
}
