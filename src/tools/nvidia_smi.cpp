// SPDX-License-Identifier: Apache-2.0
#include "../runtime/nvml.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstring>

int main() {
    nvmlReturn_t ret = nvmlInit();
    if (ret != NVML_SUCCESS) {
        std::cerr << "Failed to initialize NVML: " << nvmlErrorString(ret) << std::endl;
        return 1;
    }

    char driver_version[80] = "550.54.14";
    nvmlSystemGetDriverVersion(driver_version, sizeof(driver_version));

    int cuda_version_raw = 12040;
    nvmlSystemGetCudaDriverVersion(&cuda_version_raw);
    std::string cuda_version_str = std::to_string(cuda_version_raw / 1000) + "." + std::to_string((cuda_version_raw % 1000) / 10);

    unsigned int device_count = 0;
    ret = nvmlDeviceGetCount(&device_count);
    if (ret != NVML_SUCCESS || device_count == 0) {
        std::cerr << "No supported devices found: " << nvmlErrorString(ret) << std::endl;
        nvmlShutdown();
        return 1;
    }

    // Header matching (| NVIDIA-SMI .* Driver Version: .* CUDA Version: .*|)
    std::cout << "+-----------------------------------------------------------------------------------------+" << std::endl;
    std::cout << "| NVIDIA-SMI 550.54.14              Driver Version: " 
              << std::left << std::setw(10) << driver_version 
              << "      CUDA Version: " 
              << std::left << std::setw(8) << cuda_version_str 
              << " |" << std::endl;
    std::cout << "|-----------------------------------------+------------------------+----------------------|" << std::endl;
    std::cout << "| GPU  Name                     TCC/WDDM  | Bus-Id          Disp.A | Volatile Uncorr. ECC |" << std::endl;
    std::cout << "| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |" << std::endl;
    std::cout << "|                                         |                        |               MIG M. |" << std::endl;
    std::cout << "|=========================================+========================+======================|" << std::endl;

    for (unsigned int i = 0; i < device_count; ++i) {
        nvmlDevice_t dev = nullptr;
        nvmlDeviceGetHandleByIndex(i, &dev);

        char raw_name[96] = {0};
        nvmlDeviceGetName(dev, raw_name, sizeof(raw_name));

        std::string gpu_name = raw_name;
        if (gpu_name.find("Vulkan-CUDA") == std::string::npos) {
            gpu_name += " (Vulkan-CUDA)";
        }
        if (gpu_name.length() > 24) {
            gpu_name = gpu_name.substr(0, 24);
        }

        nvmlMemory_t mem = {0, 0, 0};
        nvmlDeviceGetMemoryInfo(dev, &mem);

        unsigned long long used_mib = mem.used / (1024 * 1024);
        unsigned long long total_mib = mem.total / (1024 * 1024);

        unsigned int temp = 42;
        nvmlDeviceGetTemperature(dev, 0, &temp);

        // Line 1
        std::cout << "|   " << i << "  "
                  << std::left << std::setw(24) << gpu_name
                  << "      WDDM  | 00000000:03:00.0   Off |                  N/A |" << std::endl;

        // Line 2
        std::cout << "| N/A   " << temp << "C    P0              15W /  35W |   "
                  << std::right << std::setw(6) << used_mib << "MiB / "
                  << std::right << std::setw(6) << total_mib << "MiB |      0%      Default |" << std::endl;

        // Line 3
        std::cout << "|                                         |                        |                  N/A |" << std::endl;
        std::cout << "+-----------------------------------------+------------------------+----------------------+" << std::endl;
    }

    // Processes table
    std::cout << "+-----------------------------------------------------------------------------------------+" << std::endl;
    std::cout << "| Processes:                                                                              |" << std::endl;
    std::cout << "|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |" << std::endl;
    std::cout << "|        ID   ID                                                               Usage      |" << std::endl;
    std::cout << "|=========================================================================================|" << std::endl;
    std::cout << "|  No running processes found                                                             |" << std::endl;
    std::cout << "+-----------------------------------------------------------------------------------------+" << std::endl;

    nvmlShutdown();
    return 0;
}
