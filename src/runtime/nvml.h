// SPDX-License-Identifier: Apache-2.0
#ifndef NVML_H
#define NVML_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
  #if defined(NVML_EXPORTS)
    #define NVML_API __declspec(dllexport)
  #else
    #define NVML_API __declspec(dllimport)
  #endif
#else
  #define NVML_API __attribute__((visibility("default")))
#endif

typedef enum nvmlReturn_enum {
    NVML_SUCCESS = 0,
    NVML_ERROR_UNINITIALIZED = 1,
    NVML_ERROR_INVALID_ARGUMENT = 2,
    NVML_ERROR_NOT_SUPPORTED = 3,
    NVML_ERROR_NO_PERMISSION = 4,
    NVML_ERROR_ALREADY_INITIALIZED = 5,
    NVML_ERROR_NOT_FOUND = 6,
    NVML_ERROR_INSUFFICIENT_SIZE = 7,
    NVML_ERROR_INSUFFICIENT_POWER = 8,
    NVML_ERROR_DRIVER_NOT_LOADED = 9,
    NVML_ERROR_TIMEOUT = 10,
    NVML_ERROR_IRQ_ISSUE = 11,
    NVML_ERROR_LIBRARY_NOT_FOUND = 12,
    NVML_ERROR_FUNCTION_NOT_FOUND = 13,
    NVML_ERROR_CORRUPTED_INFOROM = 14,
    NVML_ERROR_GPU_IS_LOST = 15,
    NVML_ERROR_RESET_REQUIRED = 16,
    NVML_ERROR_OPERATING_SYSTEM = 17,
    NVML_ERROR_UNKNOWN = 999
} nvmlReturn_t;

typedef struct nvmlDevice_st* nvmlDevice_t;

typedef struct nvmlMemory_st {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
} nvmlMemory_t;

typedef struct nvmlUtilization_st {
    unsigned int gpu;
    unsigned int memory;
} nvmlUtilization_t;

NVML_API nvmlReturn_t nvmlInit_v2(void);
NVML_API nvmlReturn_t nvmlInit(void);
NVML_API nvmlReturn_t nvmlShutdown(void);
NVML_API nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int* deviceCount);
NVML_API nvmlReturn_t nvmlDeviceGetCount(unsigned int* deviceCount);
NVML_API nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t* device);
NVML_API nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t* device);
NVML_API nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char* name, unsigned int length);
NVML_API nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t* memory);
NVML_API nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t device, nvmlUtilization_t* utilization);
NVML_API nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t device, int sensorType, unsigned int* temp);
NVML_API nvmlReturn_t nvmlSystemGetDriverVersion(char* version, unsigned int length);
NVML_API nvmlReturn_t nvmlSystemGetCudaDriverVersion(int* cudaDriverVersion);
NVML_API const char* nvmlErrorString(nvmlReturn_t result);

#ifdef __cplusplus
}
#endif

#endif // NVML_H
