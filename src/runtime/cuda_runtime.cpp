// SPDX-License-Identifier: Apache-2.0
#define CUDART_EXPORTS
#include "cuda_runtime.h"

#include <vulkan/vulkan.h>
#include <vector>
#include <mutex>
#include <map>
#include <string>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <chrono>

namespace {

struct MemoryChunk {
    size_t offset;
    size_t size;
    bool is_free;
};

struct Slab {
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceAddress baseAddress = 0;
    size_t totalSize = 0;
    std::vector<MemoryChunk> chunks;
};

struct Allocation {
    size_t slab_index = 0;
    size_t offset = 0;
    size_t size = 0;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory dedicated_memory = VK_NULL_HANDLE;
    bool is_dedicated = false;
};

struct StagingBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mappedPtr = nullptr;
    size_t size = 0;
};

} // namespace

struct CUstream_st {
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkSemaphore timelineSemaphore = VK_NULL_HANDLE;
    uint64_t timelineValue = 0;
    std::mutex streamMutex;
};

struct CUevent_st {
    VkSemaphore timelineSemaphore = VK_NULL_HANDLE;
    uint64_t targetValue = 0;
    std::chrono::high_resolution_clock::time_point recordedTime;
    bool isRecorded = false;
};

class VulkanRuntime {
public:
    static VulkanRuntime& get() {
        static VulkanRuntime instance;
        return instance;
    }

    cudaError_t initialize() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_initialized) return cudaSuccess;

        VkApplicationInfo appInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
        appInfo.pApplicationName = "CUDAVulkanRuntime";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "UniversalTranslator";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_3;

        VkInstanceCreateInfo instanceInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
        instanceInfo.pApplicationInfo = &appInfo;

        VkResult res = vkCreateInstance(&instanceInfo, nullptr, &m_instance);
        if (res != VK_SUCCESS) {
            appInfo.apiVersion = VK_API_VERSION_1_2;
            res = vkCreateInstance(&instanceInfo, nullptr, &m_instance);
            if (res != VK_SUCCESS) {
                return cudaErrorInitializationError;
            }
        }

        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr);
        if (deviceCount == 0) return cudaErrorNoDevice;

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices.data());

        m_physicalDevice = devices[0];
        for (auto& pd : devices) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(pd, &props);
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                m_physicalDevice = pd;
                break;
            }
        }

        vkGetPhysicalDeviceProperties(m_physicalDevice, &m_deviceProperties);
        vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &m_memProperties);

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueFamilyCount, queueFamilies.data());

        m_computeQueueFamily = UINT32_MAX;
        for (uint32_t i = 0; i < queueFamilyCount; ++i) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                m_computeQueueFamily = i;
                break;
            }
        }
        if (m_computeQueueFamily == UINT32_MAX) return cudaErrorInitializationError;

        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        queueInfo.queueFamilyIndex = m_computeQueueFamily;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &queuePriority;

        VkPhysicalDeviceBufferDeviceAddressFeatures bdaFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES };
        bdaFeatures.bufferDeviceAddress = VK_TRUE;

        VkPhysicalDeviceTimelineSemaphoreFeatures timelineFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES };
        timelineFeatures.pNext = &bdaFeatures;
        timelineFeatures.timelineSemaphore = VK_TRUE;

        VkPhysicalDeviceVulkan11Features v11Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
        v11Features.storageBuffer16BitAccess = VK_TRUE;

        VkPhysicalDeviceVulkan12Features v12Features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
        v12Features.pNext = &v11Features;
        v12Features.bufferDeviceAddress = VK_TRUE;
        v12Features.timelineSemaphore = VK_TRUE;
        v12Features.shaderFloat16 = VK_TRUE;

        VkPhysicalDeviceFeatures deviceFeatures = {};
        deviceFeatures.shaderInt64 = VK_TRUE;
        deviceFeatures.shaderFloat64 = VK_TRUE;

        VkDeviceCreateInfo devCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        devCreateInfo.pNext = &v12Features;
        devCreateInfo.pEnabledFeatures = &deviceFeatures;
        devCreateInfo.queueCreateInfoCount = 1;
        devCreateInfo.pQueueCreateInfos = &queueInfo;

        uint32_t extCount = 0;
        vkEnumerateDeviceExtensionProperties(m_physicalDevice, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> availableExts(extCount);
        vkEnumerateDeviceExtensionProperties(m_physicalDevice, nullptr, &extCount, availableExts.data());

        auto hasExtension = [&](const char* name) {
            for (const auto& ext : availableExts) {
                if (std::strcmp(ext.extensionName, name) == 0) return true;
            }
            return false;
        };

        std::vector<const char*> enabledExts;
        if (hasExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)) {
            enabledExts.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
        }
        if (hasExtension(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME)) {
            enabledExts.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
        }
        if (hasExtension("VK_KHR_portability_subset")) {
            enabledExts.push_back("VK_KHR_portability_subset");
        }

        devCreateInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExts.size());
        devCreateInfo.ppEnabledExtensionNames = enabledExts.empty() ? nullptr : enabledExts.data();

        res = vkCreateDevice(m_physicalDevice, &devCreateInfo, nullptr, &m_device);
        if (res != VK_SUCCESS) {
            devCreateInfo.pNext = &timelineFeatures;
            res = vkCreateDevice(m_physicalDevice, &devCreateInfo, nullptr, &m_device);
            if (res != VK_SUCCESS) {
                return cudaErrorInitializationError;
            }
        }

        vkGetDeviceQueue(m_device, m_computeQueueFamily, 0, &m_queue);

        VkCommandPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        poolInfo.queueFamilyIndex = m_computeQueueFamily;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_cmdPool);

        initStagingBuffer(32 * 1024 * 1024);

        m_defaultStream.queue = m_queue;
        m_defaultStream.cmdPool = m_cmdPool;
        VkSemaphoreTypeCreateInfo stci = { VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
        stci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        stci.initialValue = 0;
        VkSemaphoreCreateInfo sci = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &stci };
        vkCreateSemaphore(m_device, &sci, nullptr, &m_defaultStream.timelineSemaphore);
        m_defaultStream.timelineValue = 0;

        m_initialized = true;
        return cudaSuccess;
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        for (uint32_t i = 0; i < m_memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (m_memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        for (uint32_t i = 0; i < m_memProperties.memoryTypeCount; i++) {
            if (typeFilter & (1 << i)) return i;
        }
        return 0;
    }

    cudaError_t allocateSlab(size_t minSize, size_t& slabIndex) {
        const size_t DEFAULT_SLAB_SIZE = 64 * 1024 * 1024; // 64 MB
        size_t allocSize = std::max(DEFAULT_SLAB_SIZE, minSize);

        VkBufferCreateInfo bufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufferInfo.size = allocSize;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        Slab slab;
        slab.totalSize = allocSize;

        VkResult res = vkCreateBuffer(m_device, &bufferInfo, nullptr, &slab.buffer);
        if (res != VK_SUCCESS) return cudaErrorMemoryAllocation;

        VkMemoryRequirements memReq;
        vkGetBufferMemoryRequirements(m_device, slab.buffer, &memReq);

        VkMemoryAllocateFlagsInfo flagsInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO };
        flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

        VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &flagsInfo };
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        res = vkAllocateMemory(m_device, &allocInfo, nullptr, &slab.memory);
        if (res != VK_SUCCESS) {
            vkDestroyBuffer(m_device, slab.buffer, nullptr);
            return cudaErrorMemoryAllocation;
        }

        vkBindBufferMemory(m_device, slab.buffer, slab.memory, 0);

        VkBufferDeviceAddressInfo bdaInfo = { VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
        bdaInfo.buffer = slab.buffer;
        slab.baseAddress = vkGetBufferDeviceAddress(m_device, &bdaInfo);

        if (slab.baseAddress == 0) {
            vkFreeMemory(m_device, slab.memory, nullptr);
            vkDestroyBuffer(m_device, slab.buffer, nullptr);
            return cudaErrorMemoryAllocation;
        }

        slab.chunks.push_back({0, allocSize, true});
        m_slabs.push_back(slab);
        slabIndex = m_slabs.size() - 1;
        return cudaSuccess;
    }

    cudaError_t allocateDedicated(size_t alignedSize, void** devPtr) {
        VkBufferCreateInfo bufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufferInfo.size = alignedSize;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer buffer = VK_NULL_HANDLE;
        VkResult res = vkCreateBuffer(m_device, &bufferInfo, nullptr, &buffer);
        if (res != VK_SUCCESS) return cudaErrorMemoryAllocation;

        VkMemoryRequirements memReq;
        vkGetBufferMemoryRequirements(m_device, buffer, &memReq);

        VkMemoryDedicatedAllocateInfo dedicatedInfo = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO };
        dedicatedInfo.buffer = buffer;

        VkMemoryAllocateFlagsInfo flagsInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO };
        flagsInfo.pNext = &dedicatedInfo;
        flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

        VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        allocInfo.pNext = &flagsInfo;
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        VkDeviceMemory memory = VK_NULL_HANDLE;
        res = vkAllocateMemory(m_device, &allocInfo, nullptr, &memory);
        if (res != VK_SUCCESS) {
            flagsInfo.pNext = nullptr;
            res = vkAllocateMemory(m_device, &allocInfo, nullptr, &memory);
            if (res != VK_SUCCESS) {
                vkDestroyBuffer(m_device, buffer, nullptr);
                return cudaErrorMemoryAllocation;
            }
        }

        res = vkBindBufferMemory(m_device, buffer, memory, 0);
        if (res != VK_SUCCESS) {
            vkFreeMemory(m_device, memory, nullptr);
            vkDestroyBuffer(m_device, buffer, nullptr);
            return cudaErrorMemoryAllocation;
        }

        VkBufferDeviceAddressInfo bdaInfo = { VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
        bdaInfo.buffer = buffer;
        VkDeviceAddress gpuAddress = vkGetBufferDeviceAddress(m_device, &bdaInfo);
        if (gpuAddress == 0) {
            vkFreeMemory(m_device, memory, nullptr);
            vkDestroyBuffer(m_device, buffer, nullptr);
            return cudaErrorMemoryAllocation;
        }

        Allocation alloc;
        alloc.slab_index = static_cast<size_t>(-1);
        alloc.offset = 0;
        alloc.size = alignedSize;
        alloc.buffer = buffer;
        alloc.dedicated_memory = memory;
        alloc.is_dedicated = true;

        m_allocations[gpuAddress] = alloc;
        *devPtr = reinterpret_cast<void*>(gpuAddress);
        return cudaSuccess;
    }

    cudaError_t malloc(void** devPtr, size_t size) {
        if (!devPtr) return cudaErrorInvalidValue;
        if (size == 0) {
            *devPtr = nullptr;
            return cudaSuccess;
        }

        cudaError_t err = initialize();
        if (err != cudaSuccess) return err;

        std::lock_guard<std::mutex> lock(m_mutex);

        // 256-byte alignment
        size_t alignedSize = (size + 255) & ~((size_t)255);

        // Fallback to dedicated allocation for blocks >= 64 MB
        if (alignedSize >= 64 * 1024 * 1024) {
            return allocateDedicated(alignedSize, devPtr);
        }

        // Search existing slabs
        for (size_t i = 0; i < m_slabs.size(); ++i) {
            auto& slab = m_slabs[i];
            for (size_t c = 0; c < slab.chunks.size(); ++c) {
                if (slab.chunks[c].is_free && slab.chunks[c].size >= alignedSize) {
                    size_t rem = slab.chunks[c].size - alignedSize;
                    slab.chunks[c].size = alignedSize;
                    slab.chunks[c].is_free = false;
                    size_t allocOffset = slab.chunks[c].offset;

                    if (rem > 0) {
                        MemoryChunk remChunk = { allocOffset + alignedSize, rem, true };
                        slab.chunks.insert(slab.chunks.begin() + c + 1, remChunk);
                    }

                    uint64_t gpuAddress = slab.baseAddress + allocOffset;
                    Allocation alloc;
                    alloc.slab_index = i;
                    alloc.offset = allocOffset;
                    alloc.size = alignedSize;
                    alloc.buffer = slab.buffer;
                    alloc.is_dedicated = false;
                    m_allocations[gpuAddress] = alloc;
                    *devPtr = reinterpret_cast<void*>(gpuAddress);
                    return cudaSuccess;
                }
            }
        }

        // Allocate a new slab
        size_t newSlabIdx = 0;
        err = allocateSlab(alignedSize, newSlabIdx);
        if (err != cudaSuccess) return err;

        auto& slab = m_slabs[newSlabIdx];
        size_t rem = slab.chunks[0].size - alignedSize;
        slab.chunks[0].size = alignedSize;
        slab.chunks[0].is_free = false;
        size_t allocOffset = slab.chunks[0].offset;

        if (rem > 0) {
            MemoryChunk remChunk = { allocOffset + alignedSize, rem, true };
            slab.chunks.push_back(remChunk);
        }

        uint64_t gpuAddress = slab.baseAddress + allocOffset;
        Allocation alloc;
        alloc.slab_index = newSlabIdx;
        alloc.offset = allocOffset;
        alloc.size = alignedSize;
        alloc.buffer = slab.buffer;
        alloc.is_dedicated = false;
        m_allocations[gpuAddress] = alloc;
        *devPtr = reinterpret_cast<void*>(gpuAddress);
        return cudaSuccess;
    }

    cudaError_t free(void* devPtr) {
        if (!devPtr) return cudaSuccess;

        cudaError_t err = initialize();
        if (err != cudaSuccess) return err;

        std::lock_guard<std::mutex> lock(m_mutex);
        uint64_t addr = reinterpret_cast<uint64_t>(devPtr);
        auto it = m_allocations.find(addr);
        if (it == m_allocations.end()) return cudaErrorInvalidValue;

        Allocation alloc = it->second;
        m_allocations.erase(it);

        if (alloc.is_dedicated) {
            vkFreeMemory(m_device, alloc.dedicated_memory, nullptr);
            vkDestroyBuffer(m_device, alloc.buffer, nullptr);
            return cudaSuccess;
        }

        auto& slab = m_slabs[alloc.slab_index];
        for (size_t c = 0; c < slab.chunks.size(); ++c) {
            if (slab.chunks[c].offset == alloc.offset) {
                slab.chunks[c].is_free = true;

                // Coalesce forward
                if (c + 1 < slab.chunks.size() && slab.chunks[c + 1].is_free) {
                    slab.chunks[c].size += slab.chunks[c + 1].size;
                    slab.chunks.erase(slab.chunks.begin() + c + 1);
                }

                // Coalesce backward
                if (c > 0 && slab.chunks[c - 1].is_free) {
                    slab.chunks[c - 1].size += slab.chunks[c].size;
                    slab.chunks.erase(slab.chunks.begin() + c);
                }
                return cudaSuccess;
            }
        }
        return cudaErrorInvalidValue;
    }

    bool findAllocation(uint64_t addr, Allocation& outAlloc, size_t& offsetInAlloc) {
        if (m_allocations.empty()) return false;
        auto it = m_allocations.upper_bound(addr);
        if (it != m_allocations.begin()) {
            --it;
            if (addr >= it->first && addr < it->first + it->second.size) {
                outAlloc = it->second;
                offsetInAlloc = addr - it->first;
                return true;
            }
        }
        return false;
    }

    cudaError_t memcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = nullptr) {
        if (count == 0) return cudaSuccess;
        if (!dst || !src) return cudaErrorInvalidValue;

        cudaError_t err = initialize();
        if (err != cudaSuccess) return err;

        if (kind == cudaMemcpyHostToHost) {
            std::memcpy(dst, src, count);
            return cudaSuccess;
        }

        std::lock_guard<std::mutex> lock(m_mutex);

        CUstream_st* targetStream = stream ? stream : &m_defaultStream;

        if (kind == cudaMemcpyHostToDevice) {
            uint64_t dstAddr = reinterpret_cast<uint64_t>(dst);
            Allocation dstAlloc;
            size_t off = 0;
            if (!findAllocation(dstAddr, dstAlloc, off)) return cudaErrorInvalidValue;

            ensureStagingBufferSize(count);
            std::memcpy(m_staging.mappedPtr, src, count);

            executeCopyBuffer(targetStream, m_staging.buffer, 0, dstAlloc.buffer, dstAlloc.offset + off, count);
            return cudaSuccess;
        } else if (kind == cudaMemcpyDeviceToHost) {
            uint64_t srcAddr = reinterpret_cast<uint64_t>(src);
            Allocation srcAlloc;
            size_t off = 0;
            if (!findAllocation(srcAddr, srcAlloc, off)) return cudaErrorInvalidValue;

            ensureStagingBufferSize(count);
            executeCopyBuffer(targetStream, srcAlloc.buffer, srcAlloc.offset + off, m_staging.buffer, 0, count);
            std::memcpy(dst, m_staging.mappedPtr, count);
            return cudaSuccess;
        } else if (kind == cudaMemcpyDeviceToDevice) {
            uint64_t srcAddr = reinterpret_cast<uint64_t>(src);
            uint64_t dstAddr = reinterpret_cast<uint64_t>(dst);
            Allocation srcAlloc, dstAlloc;
            size_t srcOff = 0, dstOff = 0;
            if (!findAllocation(srcAddr, srcAlloc, srcOff) || !findAllocation(dstAddr, dstAlloc, dstOff)) {
                return cudaErrorInvalidValue;
            }
            executeCopyBuffer(targetStream, srcAlloc.buffer, srcAlloc.offset + srcOff, dstAlloc.buffer, dstAlloc.offset + dstOff, count);
            return cudaSuccess;
        }

        return cudaErrorInvalidValue;
    }

    cudaError_t streamCreate(cudaStream_t* pStream) {
        if (!pStream) return cudaErrorInvalidValue;
        cudaError_t err = initialize();
        if (err != cudaSuccess) return err;

        CUstream_st* stream = new CUstream_st();
        stream->queue = m_queue;

        VkCommandPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        poolInfo.queueFamilyIndex = m_computeQueueFamily;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        vkCreateCommandPool(m_device, &poolInfo, nullptr, &stream->cmdPool);

        VkSemaphoreTypeCreateInfo stci = { VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
        stci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        stci.initialValue = 0;

        VkSemaphoreCreateInfo sci = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &stci };
        vkCreateSemaphore(m_device, &sci, nullptr, &stream->timelineSemaphore);
        stream->timelineValue = 0;

        *pStream = stream;
        return cudaSuccess;
    }

    cudaError_t streamDestroy(cudaStream_t stream) {
        if (!stream) return cudaSuccess;
        if (stream == &m_defaultStream) return cudaSuccess;

        streamSynchronize(stream);
        vkDestroySemaphore(m_device, stream->timelineSemaphore, nullptr);
        vkDestroyCommandPool(m_device, stream->cmdPool, nullptr);
        delete stream;
        return cudaSuccess;
    }

    cudaError_t streamSynchronize(cudaStream_t stream) {
        cudaError_t err = initialize();
        if (err != cudaSuccess) return err;

        CUstream_st* s = stream ? stream : &m_defaultStream;
        std::lock_guard<std::mutex> streamLock(s->streamMutex);

        if (s->timelineValue == 0) return cudaSuccess;

        uint64_t waitVal = s->timelineValue;
        VkSemaphoreWaitInfo waitInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
        waitInfo.semaphoreCount = 1;
        waitInfo.pSemaphores = &s->timelineSemaphore;
        waitInfo.pValues = &waitVal;

        VkResult res = vkWaitSemaphores(m_device, &waitInfo, UINT64_MAX);
        return (res == VK_SUCCESS) ? cudaSuccess : cudaErrorLaunchFailure;
    }

    cudaError_t deviceSynchronize() {
        cudaError_t err = initialize();
        if (err != cudaSuccess) return err;

        vkDeviceWaitIdle(m_device);
        return cudaSuccess;
    }

    VkDevice getDevice() const { return m_device; }
    VkPhysicalDevice getPhysicalDevice() const { return m_physicalDevice; }
    const VkPhysicalDeviceProperties& getDeviceProperties() const { return m_deviceProperties; }

    cudaError_t getVulkanContext(void** pInstance, void** pPhysicalDevice, void** pDevice, void** pQueue, uint32_t* pQueueFamily) {
        cudaError_t err = initialize();
        if (err != cudaSuccess) return err;
        if (pInstance) *pInstance = m_instance;
        if (pPhysicalDevice) *pPhysicalDevice = m_physicalDevice;
        if (pDevice) *pDevice = m_device;
        if (pQueue) *pQueue = m_queue;
        if (pQueueFamily) *pQueueFamily = m_computeQueueFamily;
        return cudaSuccess;
    }

    size_t getTotalDeviceLocalMemory() const {
        size_t total = 0;
        for (uint32_t i = 0; i < m_memProperties.memoryHeapCount; ++i) {
            if (m_memProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                total += m_memProperties.memoryHeaps[i].size;
            }
        }
        return total;
    }

    cudaError_t launchSpirv(const char* spvPath, dim3 gridDim, dim3 blockDim, const void* pushConstants, size_t pushConstantsSize, cudaStream_t stream) {
        cudaError_t err = initialize();
        if (err != cudaSuccess) return err;

        FILE* f = fopen(spvPath, "rb");
        if (!f) return cudaErrorFileNotFound;
        fseek(f, 0, SEEK_END);
        size_t spvSize = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::vector<uint32_t> spvData(spvSize / sizeof(uint32_t));
        fread(spvData.data(), 1, spvSize, f);
        fclose(f);

        VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        smci.codeSize = spvSize;
        smci.pCode = spvData.data();
        VkShaderModule shaderModule = VK_NULL_HANDLE;
        VkResult res = vkCreateShaderModule(m_device, &smci, nullptr, &shaderModule);
        if (res != VK_SUCCESS) return cudaErrorInvalidDeviceFunction;

        uint32_t alignedPcSize = static_cast<uint32_t>((pushConstantsSize + 3) & ~size_t(3));
        VkPushConstantRange pcRange = {};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset = 0;
        pcRange.size = alignedPcSize;

        VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        if (alignedPcSize > 0) {
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges = &pcRange;
        }

        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        res = vkCreatePipelineLayout(m_device, &plci, nullptr, &pipelineLayout);
        if (res != VK_SUCCESS) {
            vkDestroyShaderModule(m_device, shaderModule, nullptr);
            return cudaErrorInitializationError;
        }

        VkComputePipelineCreateInfo cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = shaderModule;
        cpci.stage.pName = "main";
        cpci.layout = pipelineLayout;

        VkPipeline pipeline = VK_NULL_HANDLE;
        res = vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline);
        if (res != VK_SUCCESS) {
            fprintf(stderr, "vkCreateComputePipelines failed: res=%d\n", (int)res);
            vkDestroyPipelineLayout(m_device, pipelineLayout, nullptr);
            vkDestroyShaderModule(m_device, shaderModule, nullptr);
            return cudaErrorLaunchFailure;
        }

        CUstream_st* targetStream = stream ? stream : &m_defaultStream;
        std::lock_guard<std::mutex> streamLock(targetStream->streamMutex);

        VkCommandBufferAllocateInfo cbai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        cbai.commandPool = targetStream->cmdPool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;

        VkCommandBuffer cb;
        vkAllocateCommandBuffers(m_device, &cbai, &cb);

        VkCommandBufferBeginInfo begInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        begInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cb, &begInfo);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        if (pushConstants && alignedPcSize > 0) {
            if (alignedPcSize == pushConstantsSize) {
                vkCmdPushConstants(cb, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, alignedPcSize, pushConstants);
            } else {
                std::vector<uint8_t> padded(alignedPcSize, 0);
                std::memcpy(padded.data(), pushConstants, pushConstantsSize);
                vkCmdPushConstants(cb, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, alignedPcSize, padded.data());
            }
        }
        vkCmdDispatch(cb, gridDim.x, gridDim.y, gridDim.z);

        vkEndCommandBuffer(cb);

        targetStream->timelineValue++;
        uint64_t sigVal = targetStream->timelineValue;

        VkTimelineSemaphoreSubmitInfo timelineInfo = { VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
        timelineInfo.signalSemaphoreValueCount = 1;
        timelineInfo.pSignalSemaphoreValues = &sigVal;

        VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO, &timelineInfo };
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cb;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &targetStream->timelineSemaphore;

        {
            std::lock_guard<std::mutex> queueLock(m_queueMutex);
            vkQueueSubmit(targetStream->queue, 1, &submitInfo, VK_NULL_HANDLE);
        }

        VkSemaphoreWaitInfo waitInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
        waitInfo.semaphoreCount = 1;
        waitInfo.pSemaphores = &targetStream->timelineSemaphore;
        waitInfo.pValues = &sigVal;
        vkWaitSemaphores(m_device, &waitInfo, UINT64_MAX);

        vkFreeCommandBuffers(m_device, targetStream->cmdPool, 1, &cb);
        vkDestroyPipeline(m_device, pipeline, nullptr);
        vkDestroyPipelineLayout(m_device, pipelineLayout, nullptr);
        vkDestroyShaderModule(m_device, shaderModule, nullptr);

        return cudaSuccess;
    }

    void cleanup() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_initialized) return;

        if (m_device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(m_device);

            for (auto& pair : m_allocations) {
                if (pair.second.is_dedicated) {
                    if (pair.second.dedicated_memory != VK_NULL_HANDLE) {
                        vkFreeMemory(m_device, pair.second.dedicated_memory, nullptr);
                    }
                    if (pair.second.buffer != VK_NULL_HANDLE) {
                        vkDestroyBuffer(m_device, pair.second.buffer, nullptr);
                    }
                }
            }
            m_allocations.clear();

            for (auto& slab : m_slabs) {
                if (slab.memory != VK_NULL_HANDLE) vkFreeMemory(m_device, slab.memory, nullptr);
                if (slab.buffer != VK_NULL_HANDLE) vkDestroyBuffer(m_device, slab.buffer, nullptr);
            }
            m_slabs.clear();

            if (m_staging.mappedPtr) {
                vkUnmapMemory(m_device, m_staging.memory);
                m_staging.mappedPtr = nullptr;
            }
            if (m_staging.buffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(m_device, m_staging.buffer, nullptr);
                m_staging.buffer = VK_NULL_HANDLE;
            }
            if (m_staging.memory != VK_NULL_HANDLE) {
                vkFreeMemory(m_device, m_staging.memory, nullptr);
                m_staging.memory = VK_NULL_HANDLE;
            }

            if (m_defaultStream.timelineSemaphore != VK_NULL_HANDLE) {
                vkDestroySemaphore(m_device, m_defaultStream.timelineSemaphore, nullptr);
                m_defaultStream.timelineSemaphore = VK_NULL_HANDLE;
            }
            if (m_cmdPool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(m_device, m_cmdPool, nullptr);
                m_cmdPool = VK_NULL_HANDLE;
            }

            vkDestroyDevice(m_device, nullptr);
            m_device = VK_NULL_HANDLE;
        }

        if (m_instance != VK_NULL_HANDLE) {
            vkDestroyInstance(m_instance, nullptr);
            m_instance = VK_NULL_HANDLE;
        }

        m_initialized = false;
    }

private:
    VulkanRuntime() = default;
    ~VulkanRuntime() = default;

    void initStagingBuffer(size_t size) {
        m_staging.size = size;
        VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(m_device, &bci, nullptr, &m_staging.buffer);

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(m_device, m_staging.buffer, &req);

        VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(m_device, &mai, nullptr, &m_staging.memory);
        vkBindBufferMemory(m_device, m_staging.buffer, m_staging.memory, 0);
        vkMapMemory(m_device, m_staging.memory, 0, req.size, 0, &m_staging.mappedPtr);
    }

    void ensureStagingBufferSize(size_t needed) {
        if (needed <= m_staging.size) return;
        vkDeviceWaitIdle(m_device);
        if (m_staging.mappedPtr) vkUnmapMemory(m_device, m_staging.memory);
        if (m_staging.buffer) vkDestroyBuffer(m_device, m_staging.buffer, nullptr);
        if (m_staging.memory) vkFreeMemory(m_device, m_staging.memory, nullptr);
        initStagingBuffer(needed * 2);
    }

    void executeCopyBuffer(CUstream_st* stream, VkBuffer src, size_t srcOffset, VkBuffer dst, size_t dstOffset, size_t size) {
        std::lock_guard<std::mutex> streamLock(stream->streamMutex);

        VkCommandBufferAllocateInfo cbai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        cbai.commandPool = stream->cmdPool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;

        VkCommandBuffer cb;
        vkAllocateCommandBuffers(m_device, &cbai, &cb);

        VkCommandBufferBeginInfo begInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        begInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cb, &begInfo);

        VkBufferCopy region = { srcOffset, dstOffset, size };
        vkCmdCopyBuffer(cb, src, dst, 1, &region);

        vkEndCommandBuffer(cb);

        stream->timelineValue++;
        uint64_t sigVal = stream->timelineValue;

        VkTimelineSemaphoreSubmitInfo timelineInfo = { VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
        timelineInfo.signalSemaphoreValueCount = 1;
        timelineInfo.pSignalSemaphoreValues = &sigVal;

        VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO, &timelineInfo };
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cb;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &stream->timelineSemaphore;

        {
            std::lock_guard<std::mutex> queueLock(m_queueMutex);
            vkQueueSubmit(stream->queue, 1, &submitInfo, VK_NULL_HANDLE);
        }

        // Wait on timeline semaphore
        VkSemaphoreWaitInfo waitInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
        waitInfo.semaphoreCount = 1;
        waitInfo.pSemaphores = &stream->timelineSemaphore;
        waitInfo.pValues = &sigVal;
        vkWaitSemaphores(m_device, &waitInfo, UINT64_MAX);

        vkFreeCommandBuffers(m_device, stream->cmdPool, 1, &cb);
    }

    bool m_initialized = false;
    std::mutex m_mutex;
    std::mutex m_queueMutex;
    VkInstance m_instance = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties m_deviceProperties{};
    VkPhysicalDeviceMemoryProperties m_memProperties{};
    uint32_t m_computeQueueFamily = 0;
    VkQueue m_queue = VK_NULL_HANDLE;
    VkCommandPool m_cmdPool = VK_NULL_HANDLE;

    CUstream_st m_defaultStream;
    StagingBuffer m_staging;
    std::vector<Slab> m_slabs;
    std::map<uint64_t, Allocation> m_allocations;
};

// C API Implementations
extern "C" {

CUDART_API cudaError_t cudaMalloc(void** devPtr, size_t size) {
    return VulkanRuntime::get().malloc(devPtr, size);
}

CUDART_API cudaError_t cudaFree(void* devPtr) {
    return VulkanRuntime::get().free(devPtr);
}

CUDART_API cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    return VulkanRuntime::get().memcpy(dst, src, count, kind, nullptr);
}

CUDART_API cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
    return VulkanRuntime::get().memcpy(dst, src, count, kind, stream);
}

CUDART_API cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
    if (!devPtr || count == 0) return cudaSuccess;
    std::vector<uint8_t> hostBuf(count, static_cast<uint8_t>(value));
    return VulkanRuntime::get().memcpy(devPtr, hostBuf.data(), count, cudaMemcpyHostToDevice, nullptr);
}

CUDART_API cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) {
    if (!devPtr || count == 0) return cudaSuccess;
    std::vector<uint8_t> hostBuf(count, static_cast<uint8_t>(value));
    return VulkanRuntime::get().memcpy(devPtr, hostBuf.data(), count, cudaMemcpyHostToDevice, stream);
}

CUDART_API cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags) {
    if (!pHost || size == 0) return cudaErrorInvalidValue;
    *pHost = std::malloc(size);
    return (*pHost) ? cudaSuccess : cudaErrorMemoryAllocation;
}

CUDART_API cudaError_t cudaFreeHost(void* ptr) {
    if (ptr) std::free(ptr);
    return cudaSuccess;
}

CUDART_API cudaError_t cudaStreamCreate(cudaStream_t* pStream) {
    return VulkanRuntime::get().streamCreate(pStream);
}

CUDART_API cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags) {
    return VulkanRuntime::get().streamCreate(pStream);
}

CUDART_API cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    return VulkanRuntime::get().streamDestroy(stream);
}

CUDART_API cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    return VulkanRuntime::get().streamSynchronize(stream);
}

CUDART_API cudaError_t cudaDeviceSynchronize(void) {
    return VulkanRuntime::get().deviceSynchronize();
}

CUDART_API cudaError_t cudaDeviceReset(void) {
    VulkanRuntime::get().cleanup();
    return cudaSuccess;
}

CUDART_API cudaError_t cudaGetDevice(int* device) {
    if (!device) return cudaErrorInvalidValue;
    *device = 0;
    return cudaSuccess;
}

CUDART_API cudaError_t cudaSetDevice(int device) {
    if (device != 0) return cudaErrorInvalidDevice;
    return cudaSuccess;
}

CUDART_API cudaError_t cudaGetDeviceCount(int* count) {
    if (!count) return cudaErrorInvalidValue;
    *count = 1;
    return cudaSuccess;
}

CUDART_API cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) {
    if (!prop || device != 0) return cudaErrorInvalidDevice;
    cudaError_t err = VulkanRuntime::get().initialize();
    if (err != cudaSuccess) return err;

    std::memset(prop, 0, sizeof(cudaDeviceProp));
    const auto& vkProps = VulkanRuntime::get().getDeviceProperties();
    std::strncpy(prop->name, vkProps.deviceName, sizeof(prop->name) - 1);
    size_t totalLocal = VulkanRuntime::get().getTotalDeviceLocalMemory();
    prop->totalGlobalMem = (totalLocal > 0) ? totalLocal : (8ULL * 1024 * 1024 * 1024);
    prop->sharedMemPerBlock = 64 * 1024;
    prop->regsPerBlock = 65536;
    prop->warpSize = 32;
    prop->memPitch = 2147483647;
    prop->maxThreadsPerBlock = 1024;
    prop->maxThreadsDim[0] = 1024;
    prop->maxThreadsDim[1] = 1024;
    prop->maxThreadsDim[2] = 64;
    prop->maxGridSize[0] = 2147483647;
    prop->maxGridSize[1] = 65535;
    prop->maxGridSize[2] = 65535;
    prop->major = 8;
    prop->minor = 0; // SM 8.0 Ampere feature level parity
    prop->multiProcessorCount = 32;
    prop->unifiedAddressing = 1;
    return cudaSuccess;
}

CUDART_API cudaError_t cudaEventCreate(cudaEvent_t* event) {
    if (!event) return cudaErrorInvalidValue;
    CUevent_st* ev = new CUevent_st();
    *event = ev;
    return cudaSuccess;
}

CUDART_API cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
    return cudaEventCreate(event);
}

CUDART_API cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    if (!event) return cudaErrorInvalidValue;
    event->recordedTime = std::chrono::high_resolution_clock::now();
    event->isRecorded = true;
    return cudaSuccess;
}

CUDART_API cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    if (!event) return cudaErrorInvalidValue;
    return cudaSuccess;
}

CUDART_API cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
    if (!ms || !start || !end) return cudaErrorInvalidValue;
    if (!start->isRecorded || !end->isRecorded) return cudaErrorInvalidResourceHandle;
    std::chrono::duration<float, std::milli> diff = end->recordedTime - start->recordedTime;
    *ms = diff.count();
    return cudaSuccess;
}

CUDART_API cudaError_t cudaEventDestroy(cudaEvent_t event) {
    if (event) delete event;
    return cudaSuccess;
}

CUDART_API const char* cudaGetErrorString(cudaError_t error) {
    switch (error) {
        case cudaSuccess: return "cudaSuccess";
        case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
        case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
        case cudaErrorInitializationError: return "cudaErrorInitializationError";
        case cudaErrorLaunchFailure: return "cudaErrorLaunchFailure";
        case cudaErrorNoDevice: return "cudaErrorNoDevice";
        case cudaErrorInvalidDevice: return "cudaErrorInvalidDevice";
        case cudaErrorNotSupported: return "cudaErrorNotSupported";
        default: return "cudaErrorUnknown";
    }
}

CUDART_API const char* cudaGetErrorName(cudaError_t error) {
    return cudaGetErrorString(error);
}

static thread_local cudaError_t g_lastError = cudaSuccess;

CUDART_API cudaError_t cudaGetLastError(void) {
    cudaError_t err = g_lastError;
    g_lastError = cudaSuccess;
    return err;
}

CUDART_API cudaError_t cudaPeekAtLastError(void) {
    return g_lastError;
}

CUDART_API cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream) {
    if (!func) return cudaErrorInvalidDeviceFunction;
    return cudaSuccess;
}

CUDART_API cudaError_t cudaLaunchSpirv(const char* spvPath, dim3 gridDim, dim3 blockDim, const void* pushConstants, size_t pushConstantsSize, cudaStream_t stream) {
    return VulkanRuntime::get().launchSpirv(spvPath, gridDim, blockDim, pushConstants, pushConstantsSize, stream);
}

CUDART_API cudaError_t cudaGetVulkanContext(void** pInstance, void** pPhysicalDevice, void** pDevice, void** pQueue, uint32_t* pQueueFamily) {
    return VulkanRuntime::get().getVulkanContext(pInstance, pPhysicalDevice, pDevice, pQueue, pQueueFamily);
}

} // extern "C"
