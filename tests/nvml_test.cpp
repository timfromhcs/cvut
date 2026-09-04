// SPDX-License-Identifier: Apache-2.0
#include "../src/runtime/nvml.h"

#include <iostream>
#include <vector>
#include <cassert>
#include <cstring>

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << " Starting NVML Telemetry & Introspection Test Suite          " << std::endl;
    std::cout << "============================================================" << std::endl;

    // 1. Initialize NVML
    nvmlReturn_t ret = nvmlInit_v2();
    assert(ret == NVML_SUCCESS);
    std::cout << "[NVML_TEST] 1. nvmlInit_v2 succeeded." << std::endl;

    // 2. Query system driver versions
    char driverVer[64] = {0};
    ret = nvmlSystemGetDriverVersion(driverVer, sizeof(driverVer));
    assert(ret == NVML_SUCCESS);
    std::cout << "[NVML_TEST] 2. System Driver Version: " << driverVer << std::endl;

    int cudaDriverVer = 0;
    ret = nvmlSystemGetCudaDriverVersion(&cudaDriverVer);
    assert(ret == NVML_SUCCESS);
    assert(cudaDriverVer == 12040);
    std::cout << "[NVML_TEST] 3. CUDA Driver Version: " << cudaDriverVer / 1000 << "." << (cudaDriverVer % 100) / 10 << std::endl;

    // 3. Query Device Count
    unsigned int deviceCount = 0;
    ret = nvmlDeviceGetCount_v2(&deviceCount);
    assert(ret == NVML_SUCCESS);
    assert(deviceCount > 0);
    std::cout << "[NVML_TEST] 4. Device Count: " << deviceCount << std::endl;

    // 4. Query Device Properties
    for (unsigned int i = 0; i < deviceCount; ++i) {
        nvmlDevice_t dev = nullptr;
        ret = nvmlDeviceGetHandleByIndex_v2(i, &dev);
        assert(ret == NVML_SUCCESS && dev != nullptr);

        char devName[256] = {0};
        ret = nvmlDeviceGetName(dev, devName, sizeof(devName));
        assert(ret == NVML_SUCCESS);
        std::cout << "[NVML_TEST] 5. Device [" << i << "] Name: " << devName << std::endl;

        nvmlMemory_t mem = {0};
        ret = nvmlDeviceGetMemoryInfo(dev, &mem);
        assert(ret == NVML_SUCCESS);
        assert(mem.total > 0);
        std::cout << "[NVML_TEST]    VRAM: " << mem.used / (1024 * 1024) << " MiB used / "
                  << mem.total / (1024 * 1024) << " MiB total (" << mem.free / (1024 * 1024) << " MiB free)" << std::endl;

        nvmlUtilization_t util = {0};
        ret = nvmlDeviceGetUtilizationRates(dev, &util);
        assert(ret == NVML_SUCCESS);
        std::cout << "[NVML_TEST]    Utilization: GPU " << util.gpu << "%, Mem " << util.memory << "%" << std::endl;

        unsigned int temp = 0;
        ret = nvmlDeviceGetTemperature(dev, 0, &temp);
        assert(ret == NVML_SUCCESS);
        std::cout << "[NVML_TEST]    Temperature: " << temp << " C" << std::endl;
    }

    // 5. Verify error strings
    const char* strSuccess = nvmlErrorString(NVML_SUCCESS);
    assert(std::strcmp(strSuccess, "Success") == 0);

    const char* strUninit = nvmlErrorString(NVML_ERROR_UNINITIALIZED);
    assert(strUninit != nullptr);

    // 6. Shutdown
    ret = nvmlShutdown();
    assert(ret == NVML_SUCCESS);
    std::cout << "[NVML_TEST] 6. nvmlShutdown succeeded." << std::endl;

    std::cout << "============================================================" << std::endl;
    std::cout << " [NVML_TEST_PASSED: ALL CHECKS VERIFIED]                    " << std::endl;
    std::cout << "============================================================" << std::endl;
    return 0;
}
