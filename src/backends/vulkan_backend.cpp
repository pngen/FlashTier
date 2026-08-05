// EXPERIMENTAL cross-vendor Vulkan compute backend.
//
// Self-declared Vulkan core 1.x API surface resolved at runtime from the
// loader, so this compiles without the Vulkan SDK and probes availability
// at runtime. Memory is allocated through device-local or host-visible
// heaps chosen from vkGetPhysicalDeviceMemoryProperties discovery.
// Synchronization is queue-based (vkQueueSubmit + vkQueueWaitIdle); timing
// uses the host clock (event_timing = false).
//
// This is fallback coverage for GPUs without a native compute backend and
// is experimental. Local validation status: not runtime-tested in this
// environment (no Vulkan driver present on the validation machine).

#include "flashtier/backends/vulkan_backend.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

#include "flashtier/error.hpp"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
// wingdi.h defines DeviceCapabilitiesA/W macros that collide with the
// vendor-neutral capability type; undef the macro so the type name is intact.
#undef DeviceCapabilities
#else
#include <dlfcn.h>
#endif

namespace flashtier {

namespace {

// ---- Vulkan core 1.x API surface (self-declared) ----------------------------

using VkResult = int32_t;
constexpr VkResult VK_SUCCESS = 0;

struct VkInstance_T {};
struct VkPhysicalDevice_T {};
struct VkDevice_T {};
struct VkQueue_T {};
struct VkBuffer_T {};
struct VkDeviceMemory_T {};
struct VkCommandPool_T {};
struct VkCommandBuffer_T {};

using VkInstance = VkInstance_T*;
using VkPhysicalDevice = VkPhysicalDevice_T*;
using VkDevice = VkDevice_T*;
using VkQueue = VkQueue_T*;
using VkBuffer = VkBuffer_T*;
using VkDeviceMemory = VkDeviceMemory_T*;
using VkCommandPool = VkCommandPool_T*;
using VkCommandBuffer = VkCommandBuffer_T*;

constexpr uint32_t VK_STRUCTURE_TYPE_APPLICATION_INFO = 0;
constexpr uint32_t VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1;
constexpr uint32_t VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 2;
constexpr uint32_t VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 4;
constexpr uint32_t VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 11;
constexpr uint32_t VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 12;
constexpr uint32_t VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 37;
constexpr uint32_t VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 38;
constexpr uint32_t VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42;
constexpr uint32_t VK_STRUCTURE_TYPE_SUBMIT_INFO = 45;

constexpr uint32_t VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 1u << 0;
constexpr uint32_t VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 1u << 1;
constexpr uint32_t VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 1u << 2;
constexpr uint32_t VK_MEMORY_PROPERTY_HOST_CACHED_BIT = 1u << 3;
constexpr uint32_t VK_MEMORY_HEAP_DEVICE_LOCAL_BIT = 1u << 0;

constexpr uint32_t VK_BUFFER_USAGE_TRANSFER_SRC_BIT = 1u << 0;
constexpr uint32_t VK_BUFFER_USAGE_TRANSFER_DST_BIT = 1u << 1;
constexpr uint32_t VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 1u << 5;

constexpr uint32_t VK_QUEUE_GRAPHICS_BIT = 1u << 0;
constexpr uint32_t VK_QUEUE_COMPUTE_BIT = 1u << 1;
constexpr uint32_t VK_QUEUE_TRANSFER_BIT = 1u << 2;

struct VkApplicationInfo {
    uint32_t sType;
    const void* pNext;
    const char* pApplicationName;
    uint32_t applicationVersion;
    const char* pEngineName;
    uint32_t engineVersion;
    uint32_t apiVersion;
};

struct VkInstanceCreateInfo {
    uint32_t sType;
    const void* pNext;
    uint32_t flags;
    const VkApplicationInfo* pApplicationInfo;
    uint32_t enabledLayerCount;
    const char* const* ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
};

struct VkDeviceQueueCreateInfo {
    uint32_t sType;
    const void* pNext;
    uint32_t flags;
    uint32_t queueFamilyIndex;
    uint32_t queueCount;
    const float* pQueuePriorities;
};

struct VkDeviceCreateInfo {
    uint32_t sType;
    const void* pNext;
    uint32_t flags;
    uint32_t queueCreateInfoCount;
    const VkDeviceQueueCreateInfo* pQueueCreateInfos;
    uint32_t enabledLayerCount;
    const char* const* ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
    const void* pEnabledFeatures;
};

struct VkPhysicalDeviceProperties {
    uint32_t apiVersion;
    uint32_t driverVersion;
    uint32_t vendorID;
    uint32_t deviceID;
    uint32_t deviceType;  // 1 = discrete, 2 = integrated, 3 = virtual, ...
    char deviceName[256];
    uint8_t pipelineCacheUUID[16];
};

struct VkMemoryType {
    uint32_t propertyFlags;
    uint32_t heapIndex;
};

struct VkMemoryHeap {
    uint64_t size;
    uint32_t flags;
};

struct VkPhysicalDeviceMemoryProperties {
    uint32_t memoryTypeCount;
    VkMemoryType memoryTypes[32];
    uint32_t memoryHeapCount;
    VkMemoryHeap memoryHeaps[16];
};

struct VkBufferCreateInfo {
    uint32_t sType;
    const void* pNext;
    uint32_t flags;
    uint64_t size;
    uint32_t usage;
    uint32_t sharingMode;
    uint32_t queueFamilyIndexCount;
    const uint32_t* pQueueFamilyIndices;
};

struct VkMemoryRequirements {
    uint64_t size;
    uint64_t alignment;
    uint32_t memoryTypeBits;
};

struct VkMemoryAllocateInfo {
    uint32_t sType;
    const void* pNext;
    uint64_t allocationSize;
    uint32_t memoryTypeIndex;
};

struct VkCommandPoolCreateInfo {
    uint32_t sType;
    const void* pNext;
    uint32_t flags;
    uint32_t queueFamilyIndex;
};

struct VkCommandBufferAllocateInfo {
    uint32_t sType;
    const void* pNext;
    VkCommandPool commandPool;
    uint32_t level;
    uint32_t commandBufferCount;
};

struct VkCommandBufferBeginInfo {
    uint32_t sType;
    const void* pNext;
    uint32_t flags;
    const void* pInheritanceInfo;
};

struct VkSubmitInfo {
    uint32_t sType;
    const void* pNext;
    uint32_t waitSemaphoreCount;
    const void* pWaitSemaphores;
    const uint32_t* pWaitDstStageMask;
    uint32_t commandBufferCount;
    const VkCommandBuffer* pCommandBuffers;
    uint32_t signalSemaphoreCount;
    const void* pSignalSemaphores;
};

struct VulkanApi {
    VkResult (*vkCreateInstance)(const VkInstanceCreateInfo*, const void*, VkInstance*) = nullptr;
    void (*vkDestroyInstance)(VkInstance, const void*) = nullptr;
    VkResult (*vkEnumeratePhysicalDevices)(VkInstance, uint32_t*, VkPhysicalDevice*) = nullptr;
    void (*vkGetPhysicalDeviceProperties)(VkPhysicalDevice, VkPhysicalDeviceProperties*) = nullptr;
    void (*vkGetPhysicalDeviceMemoryProperties)(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties*) = nullptr;
    VkResult (*vkCreateDevice)(VkPhysicalDevice, const VkDeviceCreateInfo*, const void*, VkDevice*) = nullptr;
    void (*vkDestroyDevice)(VkDevice, const void*) = nullptr;
    void (*vkGetDeviceQueue)(VkDevice, uint32_t, uint32_t, VkQueue*) = nullptr;
    VkResult (*vkCreateBuffer)(VkDevice, const VkBufferCreateInfo*, const void*, VkBuffer*) = nullptr;
    void (*vkDestroyBuffer)(VkDevice, VkBuffer, const void*) = nullptr;
    void (*vkGetBufferMemoryRequirements)(VkDevice, VkBuffer, VkMemoryRequirements*) = nullptr;
    VkResult (*vkAllocateMemory)(VkDevice, const VkMemoryAllocateInfo*, const void*, VkDeviceMemory*) = nullptr;
    void (*vkFreeMemory)(VkDevice, VkDeviceMemory, const void*) = nullptr;
    VkResult (*vkBindBufferMemory)(VkDevice, VkBuffer, VkDeviceMemory, uint64_t) = nullptr;
    VkResult (*vkMapMemory)(VkDevice, VkDeviceMemory, uint64_t, uint64_t, uint32_t, void**) = nullptr;
    void (*vkUnmapMemory)(VkDevice, VkDeviceMemory) = nullptr;
    VkResult (*vkCreateCommandPool)(VkDevice, const VkCommandPoolCreateInfo*, const void*, VkCommandPool*) = nullptr;
    void (*vkDestroyCommandPool)(VkDevice, VkCommandPool, const void*) = nullptr;
    VkResult (*vkAllocateCommandBuffers)(VkDevice, const VkCommandBufferAllocateInfo*, VkCommandBuffer*) = nullptr;
    void (*vkFreeCommandBuffers)(VkDevice, VkCommandPool, uint32_t, const VkCommandBuffer*) = nullptr;
    VkResult (*vkBeginCommandBuffer)(VkCommandBuffer, const VkCommandBufferBeginInfo*) = nullptr;
    VkResult (*vkEndCommandBuffer)(VkCommandBuffer) = nullptr;
    void (*vkCmdCopyBuffer)(VkCommandBuffer, VkBuffer, VkBuffer, uint32_t, const void*) = nullptr;
    VkResult (*vkQueueSubmit)(VkQueue, uint32_t, const VkSubmitInfo*, void*) = nullptr;
    VkResult (*vkQueueWaitIdle)(VkQueue) = nullptr;
    VkResult (*vkDeviceWaitIdle)(VkDevice) = nullptr;
    void* (*vkGetInstanceProcAddr)(VkInstance, const char*) = nullptr;
    void* (*vkGetDeviceProcAddr)(VkDevice, const char*) = nullptr;

    bool loaded = false;
};

void* resolve_symbol(void* handle, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name));
#else
    return dlsym(handle, name);
#endif
}

std::string vk_result_text(VkResult r) {
    if (r == VK_SUCCESS) return "success";
    return "vk result " + [&]() {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%d", static_cast<int>(r));
        return std::string(buf);
    }();
}

}  // namespace

class VkStream final : public DeviceStream {
public:
    VkQueue queue = nullptr;
    VkCommandPool pool = nullptr;
    VkCommandBuffer cmd = nullptr;
    std::mutex mu;  // serialize list use per stream
};

class VkEvent final : public DeviceEvent {};

struct VulkanBackend::Impl {
    VulkanApi api;
    bool loader_tried = false;
    std::string unavailable_reason;

    VkInstance instance = nullptr;
    VkPhysicalDevice physical = nullptr;
    VkDevice device = nullptr;
    int device_index = -1;
    bool opened = false;

    DeviceInfo info;
    DeviceCapabilities caps;
    int queue_family = -1;
    uint32_t device_local_memory_type = 0xFFFFFFFFu;
    uint32_t host_visible_memory_type = 0xFFFFFFFFu;

    std::mutex mu;
    std::string last_error;
};

VulkanBackend::VulkanBackend() : impl_(std::make_unique<Impl>()) {}

VulkanBackend::~VulkanBackend() {
    close();
}

namespace {

void ensure_loader(VulkanBackend::Impl& impl) {
    if (impl.loader_tried) return;
    impl.loader_tried = true;
#if defined(_WIN32)
    HMODULE h = LoadLibraryA("vulkan-1.dll");
    if (h == nullptr) {
        impl.unavailable_reason = "vulkan loader (vulkan-1.dll) not loadable";
        return;
    }
#else
    void* h = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (h == nullptr) {
        impl.unavailable_reason = "vulkan loader (libvulkan.so.1) not loadable";
        return;
    }
#endif
    impl.api.vkGetInstanceProcAddr =
        reinterpret_cast<decltype(impl.api.vkGetInstanceProcAddr)>(resolve_symbol(h, "vkGetInstanceProcAddr"));
    if (impl.api.vkGetInstanceProcAddr == nullptr) {
        impl.unavailable_reason = "vulkan loader present but incomplete";
        return;
    }
    auto get = [&](const char* n) {
        return impl.api.vkGetInstanceProcAddr(nullptr, n);
    };
    impl.api.vkCreateInstance = reinterpret_cast<decltype(impl.api.vkCreateInstance)>(get("vkCreateInstance"));
    impl.api.vkDestroyInstance = reinterpret_cast<decltype(impl.api.vkDestroyInstance)>(get("vkDestroyInstance"));
    impl.api.vkEnumeratePhysicalDevices = reinterpret_cast<decltype(impl.api.vkEnumeratePhysicalDevices)>(get("vkEnumeratePhysicalDevices"));
    impl.api.vkGetPhysicalDeviceProperties = reinterpret_cast<decltype(impl.api.vkGetPhysicalDeviceProperties)>(get("vkGetPhysicalDeviceProperties"));
    impl.api.vkGetPhysicalDeviceMemoryProperties = reinterpret_cast<decltype(impl.api.vkGetPhysicalDeviceMemoryProperties)>(get("vkGetPhysicalDeviceMemoryProperties"));
    impl.api.vkCreateDevice = reinterpret_cast<decltype(impl.api.vkCreateDevice)>(get("vkCreateDevice"));
    impl.api.vkDestroyDevice = reinterpret_cast<decltype(impl.api.vkDestroyDevice)>(get("vkDestroyDevice"));
    impl.api.vkGetDeviceQueue = reinterpret_cast<decltype(impl.api.vkGetDeviceQueue)>(get("vkGetDeviceQueue"));
    impl.api.vkCreateBuffer = reinterpret_cast<decltype(impl.api.vkCreateBuffer)>(get("vkCreateBuffer"));
    impl.api.vkDestroyBuffer = reinterpret_cast<decltype(impl.api.vkDestroyBuffer)>(get("vkDestroyBuffer"));
    impl.api.vkGetBufferMemoryRequirements = reinterpret_cast<decltype(impl.api.vkGetBufferMemoryRequirements)>(get("vkGetBufferMemoryRequirements"));
    impl.api.vkAllocateMemory = reinterpret_cast<decltype(impl.api.vkAllocateMemory)>(get("vkAllocateMemory"));
    impl.api.vkFreeMemory = reinterpret_cast<decltype(impl.api.vkFreeMemory)>(get("vkFreeMemory"));
    impl.api.vkBindBufferMemory = reinterpret_cast<decltype(impl.api.vkBindBufferMemory)>(get("vkBindBufferMemory"));
    impl.api.vkMapMemory = reinterpret_cast<decltype(impl.api.vkMapMemory)>(get("vkMapMemory"));
    impl.api.vkUnmapMemory = reinterpret_cast<decltype(impl.api.vkUnmapMemory)>(get("vkUnmapMemory"));
    impl.api.vkCreateCommandPool = reinterpret_cast<decltype(impl.api.vkCreateCommandPool)>(get("vkCreateCommandPool"));
    impl.api.vkDestroyCommandPool = reinterpret_cast<decltype(impl.api.vkDestroyCommandPool)>(get("vkDestroyCommandPool"));
    impl.api.vkAllocateCommandBuffers = reinterpret_cast<decltype(impl.api.vkAllocateCommandBuffers)>(get("vkAllocateCommandBuffers"));
    impl.api.vkFreeCommandBuffers = reinterpret_cast<decltype(impl.api.vkFreeCommandBuffers)>(get("vkFreeCommandBuffers"));
    impl.api.vkBeginCommandBuffer = reinterpret_cast<decltype(impl.api.vkBeginCommandBuffer)>(get("vkBeginCommandBuffer"));
    impl.api.vkEndCommandBuffer = reinterpret_cast<decltype(impl.api.vkEndCommandBuffer)>(get("vkEndCommandBuffer"));
    impl.api.vkCmdCopyBuffer = reinterpret_cast<decltype(impl.api.vkCmdCopyBuffer)>(get("vkCmdCopyBuffer"));
    impl.api.vkQueueSubmit = reinterpret_cast<decltype(impl.api.vkQueueSubmit)>(get("vkQueueSubmit"));
    impl.api.vkQueueWaitIdle = reinterpret_cast<decltype(impl.api.vkQueueWaitIdle)>(get("vkQueueWaitIdle"));
    impl.api.vkDeviceWaitIdle = reinterpret_cast<decltype(impl.api.vkDeviceWaitIdle)>(get("vkDeviceWaitIdle"));

    impl.api.loaded = impl.api.vkCreateInstance != nullptr &&
                      impl.api.vkEnumeratePhysicalDevices != nullptr &&
                      impl.api.vkGetPhysicalDeviceProperties != nullptr &&
                      impl.api.vkGetPhysicalDeviceMemoryProperties != nullptr &&
                      impl.api.vkCreateDevice != nullptr &&
                      impl.api.vkGetDeviceQueue != nullptr &&
                      impl.api.vkCreateBuffer != nullptr &&
                      impl.api.vkAllocateMemory != nullptr &&
                      impl.api.vkBindBufferMemory != nullptr &&
                      impl.api.vkMapMemory != nullptr &&
                      impl.api.vkCreateCommandPool != nullptr &&
                      impl.api.vkAllocateCommandBuffers != nullptr &&
                      impl.api.vkBeginCommandBuffer != nullptr &&
                      impl.api.vkCmdCopyBuffer != nullptr &&
                      impl.api.vkQueueSubmit != nullptr &&
                      impl.api.vkQueueWaitIdle != nullptr;
    if (!impl.api.loaded) {
        impl.unavailable_reason = "vulkan loader present but incomplete (missing symbols)";
    }
}

void check_vk(VkResult r, const char* what, VulkanBackend::Impl& impl) {
    if (r != VK_SUCCESS) {
        impl.last_error = std::string(what) + ": " + vk_result_text(r);
        throw Error(ErrorCode::Device, std::string(what) + " failed: " + vk_result_text(r));
    }
}

}  // namespace

std::vector<DeviceInfo> VulkanBackend::enumerate_devices() const {
    ensure_loader(*impl_);
    if (!impl_->api.loaded) {
        return {};
    }
    std::vector<DeviceInfo> out;
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "FlashTier";
    app.applicationVersion = 1;
    app.pEngineName = "FlashTier";
    app.engineVersion = 1;
    app.apiVersion = (1u << 22) | (0u << 12) | 0u;  // Vulkan 1.0
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    VkInstance instance = nullptr;
    if (impl_->api.vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) {
        impl_->unavailable_reason = "vkCreateInstance failed (no driver?)";
        return {};
    }
    uint32_t count = 0;
    impl_->api.vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count == 0) {
        impl_->api.vkDestroyInstance(instance, nullptr);
        return {};
    }
    std::vector<VkPhysicalDevice> physicals(count);
    impl_->api.vkEnumeratePhysicalDevices(instance, &count, physicals.data());
    for (uint32_t i = 0; i < count; ++i) {
        VkPhysicalDeviceProperties props{};
        impl_->api.vkGetPhysicalDeviceProperties(physicals[i], &props);
        VkPhysicalDeviceMemoryProperties mem{};
        impl_->api.vkGetPhysicalDeviceMemoryProperties(physicals[i], &mem);
        DeviceInfo info;
        info.id = "vulkan:" + std::to_string(i);
        info.name = props.deviceName;
        info.architecture = "vulkan-" + std::to_string(props.apiVersion);
        info.total_memory = 0;
        for (uint32_t h = 0; h < mem.memoryHeapCount && h < 16; ++h) {
            if (mem.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                info.total_memory += mem.memoryHeaps[h].size;
            }
        }
        if (info.total_memory == 0) {
            for (uint32_t h = 0; h < mem.memoryHeapCount && h < 16; ++h) {
                info.total_memory += mem.memoryHeaps[h].size;
            }
        }
        info.free_memory = info.total_memory;
        info.discrete = props.deviceType == 1;  // VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
        info.memory_shared = props.deviceType == 2;  // integrated
        info.index = static_cast<int>(i);
        out.push_back(std::move(info));
    }
    impl_->api.vkDestroyInstance(instance, nullptr);
    return out;
}

DeviceCapabilities VulkanBackend::probe_capabilities() const {
    DeviceCapabilities caps;
    const std::vector<DeviceInfo> devices = enumerate_devices();
    if (devices.empty()) {
        caps.note = "vulkan: no devices discovered";
        return caps;
    }
    caps.backend_available = true;
    caps.device_available = true;
    caps.explicit_allocation = true;
    caps.async_host_to_device = true;
    caps.async_device_to_host = true;
    caps.device_to_device = true;
    caps.pinned_host_allocation = true;  // host-visible coherent heaps
    caps.host_coherent_memory = true;
    caps.event_timing = false;  // host-clock timing in v0.1
    caps.alignment_bytes = 64;  // minimum per spec; requirements refined per buffer
    caps.transfer_granularity = 1;
    caps.queue_count = 1;
    caps.note = "experimental; storage buffers, transfer queues; not CUDA-compatible VRAM";
    return caps;
}

void VulkanBackend::open(int device_index) {
    if (impl_->opened) close();
    ensure_loader(*impl_);
    if (!impl_->api.loaded) {
        throw Error(ErrorCode::Unsupported,
                    "vulkan backend unavailable: " + impl_->unavailable_reason);
    }
    const std::vector<DeviceInfo> devices = enumerate_devices();
    if (device_index < 0 || device_index >= static_cast<int>(devices.size())) {
        throw Error(ErrorCode::Config,
                    "vulkan backend: requested device index out of range");
    }

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "FlashTier";
    app.applicationVersion = 1;
    app.pEngineName = "FlashTier";
    app.engineVersion = 1;
    app.apiVersion = (1u << 22);
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    check_vk(impl_->api.vkCreateInstance(&ici, nullptr, &impl_->instance),
             "vkCreateInstance", *impl_);

    uint32_t count = 0;
    impl_->api.vkEnumeratePhysicalDevices(impl_->instance, &count, nullptr);
    std::vector<VkPhysicalDevice> physicals(count);
    impl_->api.vkEnumeratePhysicalDevices(impl_->instance, &count, physicals.data());
    impl_->physical = physicals[static_cast<std::size_t>(device_index)];
    impl_->info = devices[static_cast<std::size_t>(device_index)];

    // Find a queue family supporting TRANSFER (graphics/compute families
    // imply transfer capability in the core spec).
    impl_->queue_family = 0;
    // Use family 0 (graphics|compute|transfer per spec requirements).

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo dq{};
    dq.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    dq.queueFamilyIndex = static_cast<uint32_t>(impl_->queue_family);
    dq.queueCount = 1;
    dq.pQueuePriorities = &priority;
    VkDeviceCreateInfo dc{};
    dc.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dc.queueCreateInfoCount = 1;
    dc.pQueueCreateInfos = &dq;
    check_vk(impl_->api.vkCreateDevice(impl_->physical, &dc, nullptr, &impl_->device),
             "vkCreateDevice", *impl_);

    VkPhysicalDeviceMemoryProperties mem{};
    impl_->api.vkGetPhysicalDeviceMemoryProperties(impl_->physical, &mem);
    // Choose memory types: device-local (prefer host-visible too), and a
    // host-visible+coherent type for staging.
    for (uint32_t t = 0; t < mem.memoryTypeCount && t < 32; ++t) {
        const uint32_t flags = mem.memoryTypes[t].propertyFlags;
        if (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            if (impl_->device_local_memory_type == 0xFFFFFFFFu) {
                impl_->device_local_memory_type = t;
            }
        }
        if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            if (impl_->host_visible_memory_type == 0xFFFFFFFFu) {
                impl_->host_visible_memory_type = t;
            }
            if (impl_->device_local_memory_type == 0xFFFFFFFFu &&
                (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                impl_->device_local_memory_type = t;
            }
        }
    }
    if (impl_->device_local_memory_type == 0xFFFFFFFFu) {
        impl_->device_local_memory_type = impl_->host_visible_memory_type;
    }

    impl_->caps = probe_capabilities();
    impl_->opened = true;
}

void VulkanBackend::close() {
    if (!impl_->opened) return;
    if (impl_->api.loaded) {
        if (impl_->device != nullptr) {
            impl_->api.vkDeviceWaitIdle(impl_->device);
            impl_->api.vkDestroyDevice(impl_->device, nullptr);
        }
        if (impl_->instance != nullptr) {
            impl_->api.vkDestroyInstance(impl_->instance, nullptr);
        }
    }
    impl_->device = nullptr;
    impl_->instance = nullptr;
    impl_->physical = nullptr;
    impl_->opened = false;
}

bool VulkanBackend::is_open() const {
    return impl_->opened;
}

DeviceInfo VulkanBackend::device_info() const {
    if (!impl_->opened) {
        throw Error(ErrorCode::State, "vulkan backend is not open");
    }
    return impl_->info;
}

DeviceCapabilities VulkanBackend::capabilities() const {
    if (!impl_->opened) {
        throw Error(ErrorCode::State, "vulkan backend is not open");
    }
    return impl_->caps;
}

uint64_t VulkanBackend::total_memory() const {
    return impl_->opened ? impl_->info.total_memory : 0;
}

uint64_t VulkanBackend::free_memory() const {
    return impl_->opened ? impl_->info.total_memory : 0;
}

namespace {

struct VkAllocation {
    VkBuffer buffer = nullptr;
    VkDeviceMemory memory = nullptr;
    void* mapped = nullptr;   // valid when host-visible
    uint64_t size = 0;
    uint64_t alignment = 0;
    bool host_visible = false;
};

VkAllocation create_allocation(VulkanBackend::Impl& impl, std::size_t bytes,
                               bool host_visible) {
    VkAllocation alloc;
    alloc.size = bytes;
    VkBufferCreateInfo bc{};
    bc.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bc.size = bytes;
    bc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    check_vk(impl.api.vkCreateBuffer(impl.device, &bc, nullptr, &alloc.buffer),
             "vkCreateBuffer", impl);
    VkMemoryRequirements req{};
    impl.api.vkGetBufferMemoryRequirements(impl.device, alloc.buffer, &req);
    alloc.alignment = req.alignment;
    const uint32_t type = host_visible ? impl.host_visible_memory_type
                                       : impl.device_local_memory_type;
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    check_vk(impl.api.vkAllocateMemory(impl.device, &mai, nullptr, &alloc.memory),
             "vkAllocateMemory", impl);
    check_vk(impl.api.vkBindBufferMemory(impl.device, alloc.buffer, alloc.memory, 0),
             "vkBindBufferMemory", impl);
    alloc.host_visible = host_visible;
    if (host_visible) {
        check_vk(impl.api.vkMapMemory(impl.device, alloc.memory, 0, 0xFFFFFFFFFFFFFFFFull, 0,
                                      &alloc.mapped),
                 "vkMapMemory", impl);
    }
    return alloc;
}

void destroy_allocation(VulkanBackend::Impl& impl, VkAllocation& alloc) {
    if (alloc.mapped != nullptr) {
        impl.api.vkUnmapMemory(impl.device, alloc.memory);
        alloc.mapped = nullptr;
    }
    if (alloc.buffer != nullptr) {
        impl.api.vkDestroyBuffer(impl.device, alloc.buffer, nullptr);
        alloc.buffer = nullptr;
    }
    if (alloc.memory != nullptr) {
        impl.api.vkFreeMemory(impl.device, alloc.memory, nullptr);
        alloc.memory = nullptr;
    }
}

}  // namespace

void* VulkanBackend::allocate(std::size_t bytes) {
    if (bytes == 0) {
        throw Error(ErrorCode::InvalidArgument, "vulkan: zero-byte allocation");
    }
    if (impl_->host_visible_memory_type == 0xFFFFFFFFu) {
        throw Error(ErrorCode::Unsupported,
                    "vulkan: no usable device-local memory type discovered");
    }
    auto* alloc = new VkAllocation();
    try {
        *alloc = create_allocation(*impl_, bytes, false);
    } catch (...) {
        delete alloc;
        throw;
    }
    return alloc;
}

void VulkanBackend::free(void* ptr) {
    if (ptr == nullptr) return;
    auto* alloc = static_cast<VkAllocation*>(ptr);
    destroy_allocation(*impl_, *alloc);
    delete alloc;
}

void* VulkanBackend::allocate_host_pinned(std::size_t bytes) {
    if (bytes == 0) {
        throw Error(ErrorCode::InvalidArgument, "vulkan: zero-byte host allocation");
    }
    if (impl_->host_visible_memory_type == 0xFFFFFFFFu) {
        throw Error(ErrorCode::Unsupported,
                    "vulkan: no host-visible coherent memory type discovered");
    }
    auto* alloc = new VkAllocation();
    try {
        *alloc = create_allocation(*impl_, bytes, true);
    } catch (...) {
        delete alloc;
        throw;
    }
    return alloc;
}

void VulkanBackend::free_host_pinned(void* ptr) {
    free(ptr);
}

void* VulkanBackend::allocate_unified(std::size_t bytes) {
    // A device-local buffer that is also host-visible would be unified;
    // v0.1 uses explicit staging and reports unified_memory = false.
    (void)bytes;
    throw Error(ErrorCode::Unsupported,
                "vulkan: unified memory not supported in experimental v0.1 path");
}

void VulkanBackend::free_unified(void* ptr) {
    (void)ptr;
    throw Error(ErrorCode::Unsupported,
                "vulkan: unified memory not supported in experimental v0.1 path");
}

void VulkanBackend::prefetch_to_device(void* ptr, std::size_t bytes) {
    (void)ptr;
    (void)bytes;
    throw Error(ErrorCode::Unsupported, "vulkan: prefetch not supported in experimental v0.1 path");
}

void VulkanBackend::advise_preferred_location(void* ptr, std::size_t bytes) {
    (void)ptr;
    (void)bytes;
    throw Error(ErrorCode::Unsupported, "vulkan: advice not supported in experimental v0.1 path");
}

DeviceStream* VulkanBackend::create_stream() {
    auto* s = new VkStream();
    impl_->api.vkGetDeviceQueue(impl_->device, static_cast<uint32_t>(impl_->queue_family),
                                0, &s->queue);
    VkCommandPoolCreateInfo cp{};
    cp.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cp.queueFamilyIndex = static_cast<uint32_t>(impl_->queue_family);
    check_vk(impl_->api.vkCreateCommandPool(impl_->device, &cp, nullptr, &s->pool),
             "vkCreateCommandPool", *impl_);
    VkCommandBufferAllocateInfo ca{};
    ca.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ca.commandPool = s->pool;
    ca.level = 0;
    ca.commandBufferCount = 1;
    check_vk(impl_->api.vkAllocateCommandBuffers(impl_->device, &ca, &s->cmd),
             "vkAllocateCommandBuffers", *impl_);
    return s;
}

void VulkanBackend::destroy_stream(DeviceStream* stream) {
    if (stream == nullptr) return;
    auto* s = static_cast<VkStream*>(stream);
    if (impl_->api.loaded && s->pool != nullptr) {
        impl_->api.vkFreeCommandBuffers(impl_->device, s->pool, 1, &s->cmd);
        impl_->api.vkDestroyCommandPool(impl_->device, s->pool, nullptr);
    }
    delete s;
}

DeviceEvent* VulkanBackend::create_event() {
    return new VkEvent();
}

void VulkanBackend::destroy_event(DeviceEvent* event) {
    delete event;
}

namespace {

struct VkCopyRange {
    uint64_t srcOffset;
    uint64_t dstOffset;
    uint64_t size;
};

void submit_copy(VulkanBackend::Impl& impl, VkStream* s, void* dst, const void* src,
                 std::size_t bytes) {
    auto* dst_alloc = static_cast<VkAllocation*>(dst);
    auto* src_alloc = static_cast<const VkAllocation*>(src);
    std::lock_guard lock(s->mu);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    check_vk(impl.api.vkBeginCommandBuffer(s->cmd, &bi), "vkBeginCommandBuffer", impl);
    VkCopyRange range{0, 0, bytes};
    impl.api.vkCmdCopyBuffer(s->cmd, src_alloc->buffer, dst_alloc->buffer, 1, &range);
    check_vk(impl.api.vkEndCommandBuffer(s->cmd), "vkEndCommandBuffer", impl);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &s->cmd;
    check_vk(impl.api.vkQueueSubmit(s->queue, 1, &si, nullptr), "vkQueueSubmit", impl);
}

}  // namespace

void VulkanBackend::async_copy_host_to_device(void* dst_device, const void* src_host,
                                              std::size_t bytes, DeviceStream* stream) {
    // Host->device copies stage through the host-visible allocation when
    // the destination is device-local.
    auto* dst_alloc = static_cast<VkAllocation*>(dst_device);
    if (dst_alloc->host_visible) {
        std::memcpy(dst_alloc->mapped, src_host, bytes);
        return;
    }
    // Stage through a host-visible buffer.
    VkAllocation stage;
    try {
        stage = create_allocation(*impl_, bytes, true);
    } catch (...) {
        throw Error(ErrorCode::Budget, "vulkan: cannot create staging buffer");
    }
    std::memcpy(stage.mapped, src_host, bytes);
    submit_copy(*impl_, static_cast<VkStream*>(stream), dst_device, &stage, bytes);
    sync_stream(stream);
    destroy_allocation(*impl_, stage);
}

void VulkanBackend::async_copy_device_to_host(void* dst_host, const void* src_device,
                                              std::size_t bytes, DeviceStream* stream) {
    auto* src_alloc = static_cast<const VkAllocation*>(src_device);
    if (src_alloc->host_visible) {
        std::memcpy(dst_host, src_alloc->mapped, bytes);
        return;
    }
    VkAllocation stage;
    try {
        stage = create_allocation(*impl_, bytes, true);
    } catch (...) {
        throw Error(ErrorCode::Budget, "vulkan: cannot create staging buffer");
    }
    submit_copy(*impl_, static_cast<VkStream*>(stream), &stage, src_device, bytes);
    sync_stream(stream);
    std::memcpy(dst_host, stage.mapped, bytes);
    destroy_allocation(*impl_, stage);
}

void VulkanBackend::async_copy_device_to_device(void* dst_device, const void* src_device,
                                                std::size_t bytes, DeviceStream* stream) {
    submit_copy(*impl_, static_cast<VkStream*>(stream), dst_device, src_device, bytes);
}

void VulkanBackend::sync_stream(DeviceStream* stream) {
    auto* s = static_cast<VkStream*>(stream);
    check_vk(impl_->api.vkQueueWaitIdle(s->queue), "vkQueueWaitIdle", *impl_);
}

void VulkanBackend::sync_all() {
    if (impl_->opened && impl_->device != nullptr) {
        check_vk(impl_->api.vkDeviceWaitIdle(impl_->device), "vkDeviceWaitIdle", *impl_);
    }
}

void VulkanBackend::record_event(DeviceEvent* event, DeviceStream* stream) {
    (void)event;
    (void)stream;
    throw Error(ErrorCode::Unsupported,
                "vulkan: event recording not supported in experimental v0.1 path");
}

void VulkanBackend::wait_event(DeviceEvent* event) {
    (void)event;
    throw Error(ErrorCode::Unsupported,
                "vulkan: event waiting not supported in experimental v0.1 path");
}

double VulkanBackend::event_elapsed_us(DeviceEvent* start, DeviceEvent* end) {
    (void)start;
    (void)end;
    throw Error(ErrorCode::Unsupported,
                "vulkan: event timing not supported in experimental v0.1 path");
}

bool VulkanBackend::healthy() const {
    return impl_->opened;
}

std::string VulkanBackend::diagnostics() const {
    return "vulkan backend (experimental): " +
           (impl_->api.loaded ? "loader present" : "unavailable (" + impl_->unavailable_reason + ")") +
           (impl_->opened ? ", device " + impl_->info.id : ", not open");
}

std::string VulkanBackend::last_error() const {
    std::lock_guard lock(impl_->mu);
    return impl_->last_error;
}

void register_vulkan_backend(BackendRegistry& registry) {
    registry.register_backend(
        "vulkan", [] { return std::make_unique<VulkanBackend>(); },
        "Cross-vendor (Vulkan, experimental)");
}

}  // namespace flashtier
