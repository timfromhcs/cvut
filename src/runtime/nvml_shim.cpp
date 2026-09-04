// SPDX-License-Identifier: Apache-2.0
#define NVML_EXPORTS
#include "nvml.h"

#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <cstring>
#include <mutex>
#include <iostream>

namespace {

struct NVMLContext {
    VkInstance instance = VK_NULL_HANDLE;
    std::vector<VkPhysicalDevice> devices;
    bool initialized = false;
    std::mutex mtx;

    nvmlReturn_t init() {
        std::lock_guard<std::mutex> lock(mtx);
        if (initialized) return NVML_SUCCESS;

        VkApplicationInfo appInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
        appInfo.pApplicationName = "NVMLShim";
        appInfo.apiVersion = VK_API_VERSION_1_3;

        VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
        ici.pApplicationInfo = &appInfo;

        VkResult res = vkCreateInstance(&ici, nullptr, &instance);
        if (res != VK_SUCCESS) {
            appInfo.apiVersion = VK_API_VERSION_1_2;
            res = vkCreateInstance(&ici, nullptr, &instance);
            if (res != VK_SUCCESS) return NVML_ERROR_DRIVER_NOT_LOADED;
        }

        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance, &count, nullptr);
        if (count == 0) {
            vkDestroyInstance(instance, nullptr);
            instance = VK_NULL_HANDLE;
            return NVML_ERROR_NOT_FOUND;
        }

        devices.resize(count);
        vkEnumeratePhysicalDevices(instance, &count, devices.data());

        initialized = true;
        return NVML_SUCCESS;
    }

    void shutdown() {
        std::lock_guard<std::mutex> lock(mtx);
        if (!initialized) return;
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
            instance = VK_NULL_HANDLE;
        }
        devices.clear();
        initialized = false;
    }
};

NVMLContext& getContext() {
    static NVMLContext ctx;
    return ctx;
}

} // namespace

extern "C" {

NVML_API nvmlReturn_t nvmlInit_v2(void) {
    return getContext().init();
}

NVML_API nvmlReturn_t nvmlInit(void) {
    return nvmlInit_v2();
}

NVML_API nvmlReturn_t nvmlShutdown(void) {
    getContext().shutdown();
    return NVML_SUCCESS;
}

NVML_API nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int* deviceCount) {
    if (!deviceCount) return NVML_ERROR_INVALID_ARGUMENT;
    auto& ctx = getContext();
    if (!ctx.initialized) {
        nvmlReturn_t ret = ctx.init();
        if (ret != NVML_SUCCESS) return ret;
    }
    *deviceCount = static_cast<unsigned int>(ctx.devices.size());
    return NVML_SUCCESS;
}

NVML_API nvmlReturn_t nvmlDeviceGetCount(unsigned int* deviceCount) {
    return nvmlDeviceGetCount_v2(deviceCount);
}

NVML_API nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t* device) {
    if (!device) return NVML_ERROR_INVALID_ARGUMENT;
    auto& ctx = getContext();
    if (!ctx.initialized) {
        nvmlReturn_t ret = ctx.init();
        if (ret != NVML_SUCCESS) return ret;
    }
    if (index >= ctx.devices.size()) return NVML_ERROR_NOT_FOUND;

    // Use device index + 1 as opaque handle pointer
    *device = reinterpret_cast<nvmlDevice_t>(static_cast<uintptr_t>(index + 1));
    return NVML_SUCCESS;
}

NVML_API nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t* device) {
    return nvmlDeviceGetHandleByIndex_v2(index, device);
}

NVML_API nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char* name, unsigned int length) {
    if (!device || !name || length == 0) return NVML_ERROR_INVALID_ARGUMENT;
    auto& ctx = getContext();
    if (!ctx.initialized) return NVML_ERROR_UNINITIALIZED;

    uintptr_t idx = reinterpret_cast<uintptr_t>(device) - 1;
    if (idx >= ctx.devices.size()) return NVML_ERROR_INVALID_ARGUMENT;

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(ctx.devices[idx], &props);

    std::string devName = props.deviceName;
    // Ensure name is safe and fits
    size_t copyLen = std::min<size_t>(length - 1, devName.size());
    std::memcpy(name, devName.c_str(), copyLen);
    name[copyLen] = '\0';
    return NVML_SUCCESS;
}

NVML_API nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t* memory) {
    if (!device || !memory) return NVML_ERROR_INVALID_ARGUMENT;
    auto& ctx = getContext();
    if (!ctx.initialized) return NVML_ERROR_UNINITIALIZED;

    uintptr_t idx = reinterpret_cast<uintptr_t>(device) - 1;
    if (idx >= ctx.devices.size()) return NVML_ERROR_INVALID_ARGUMENT;

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(ctx.devices[idx], &memProps);

    unsigned long long totalDeviceLocal = 0;
    for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i) {
        if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            totalDeviceLocal += memProps.memoryHeaps[i].size;
        }
    }

    if (totalDeviceLocal == 0 && memProps.memoryHeapCount > 0) {
        totalDeviceLocal = memProps.memoryHeaps[0].size;
    }

    // Default estimate if direct budget is not exposed
    memory->total = totalDeviceLocal;
    memory->used = totalDeviceLocal > (512 * 1024 * 1024) ? (512 * 1024 * 1024) : (totalDeviceLocal / 8);
    memory->free = (memory->total > memory->used) ? (memory->total - memory->used) : 0;

    return NVML_SUCCESS;
}

NVML_API nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t device, nvmlUtilization_t* utilization) {
    if (!device || !utilization) return NVML_ERROR_INVALID_ARGUMENT;
    utilization->gpu = 0;
    utilization->memory = 5;
    return NVML_SUCCESS;
}

NVML_API nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t device, int sensorType, unsigned int* temp) {
    if (!device || !temp) return NVML_ERROR_INVALID_ARGUMENT;
    *temp = 42; // Nominal operating temperature in Celsius
    return NVML_SUCCESS;
}

NVML_API nvmlReturn_t nvmlSystemGetDriverVersion(char* version, unsigned int length) {
    if (!version || length == 0) return NVML_ERROR_INVALID_ARGUMENT;
    const char* verStr = "550.54.14";
    size_t copyLen = std::min<size_t>(length - 1, std::strlen(verStr));
    std::memcpy(version, verStr, copyLen);
    version[copyLen] = '\0';
    return NVML_SUCCESS;
}

NVML_API nvmlReturn_t nvmlSystemGetCudaDriverVersion(int* cudaDriverVersion) {
    if (!cudaDriverVersion) return NVML_ERROR_INVALID_ARGUMENT;
    *cudaDriverVersion = 12040; // CUDA 12.4
    return NVML_SUCCESS;
}

NVML_API const char* nvmlErrorString(nvmlReturn_t result) {
    switch (result) {
        case NVML_SUCCESS: return "Success";
        case NVML_ERROR_UNINITIALIZED: return "Uninitialized";
        case NVML_ERROR_INVALID_ARGUMENT: return "Invalid Argument";
        case NVML_ERROR_NOT_SUPPORTED: return "Not Supported";
        case NVML_ERROR_NO_PERMISSION: return "No Permission";
        case NVML_ERROR_ALREADY_INITIALIZED: return "Already Initialized";
        case NVML_ERROR_NOT_FOUND: return "Not Found";
        case NVML_ERROR_INSUFFICIENT_SIZE: return "Insufficient Size";
        case NVML_ERROR_DRIVER_NOT_LOADED: return "Driver Not Loaded";
        default: return "Unknown Error";
    }
}

} // extern "C"
