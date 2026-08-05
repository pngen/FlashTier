// Intel Level Zero backend.
//
// The Level Zero API surface below is self-declared from the Level Zero 1.x
// specification and resolved at runtime from the loader (ze_loader.dll on
// Windows, libze_loader.so on Linux), so this module compiles without the
// Intel SDK and remains disabled when no loader/driver exists. Only the
// documented prefix fields of the property structs are read.
//
// Local validation status: not runtime-tested in this environment (no
// Intel GPU / Level Zero driver installed). Conformance must pass on real
// Intel hardware before this backend is called validated.

#include "flashtier/backends/level_zero_backend.hpp"

#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

#include <cstdio>

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

// ---- Level Zero core API surface (self-declared; Level Zero 1.x spec) ----

using ze_result_t = uint32_t;
constexpr ze_result_t ZE_RESULT_SUCCESS = 0;

struct ze_context_handle_t_ {};
struct ze_driver_handle_t_ {};
struct ze_device_handle_t_ {};
struct ze_command_queue_handle_t_ {};
struct ze_command_list_handle_t_ {};
struct ze_fence_handle_t_ {};
struct ze_event_handle_t_ {};
using ze_context_handle_t = ze_context_handle_t_*;
using ze_driver_handle_t = ze_driver_handle_t_*;
using ze_device_handle_t = ze_device_handle_t_*;
using ze_command_queue_handle_t = ze_command_queue_handle_t_*;
using ze_command_list_handle_t = ze_command_list_handle_t_*;
using ze_fence_handle_t = ze_fence_handle_t_*;
using ze_event_handle_t = ze_event_handle_t_*;

struct ze_base_desc_t {
    uint32_t stype;
    const void* pNext;
};
struct ze_context_desc_t {
    uint32_t stype;
    const void* pNext;
    uint32_t flags;
};
struct ze_command_queue_desc_t {
    uint32_t stype;
    const void* pNext;
    uint32_t ordinal;
    uint32_t flags;
};
struct ze_command_list_desc_t {
    uint32_t stype;
    const void* pNext;
    uint32_t flags;
};
struct ze_fence_desc_t {
    uint32_t stype;
    const void* pNext;
    uint32_t flags;
};
struct ze_device_mem_alloc_desc_t {
    uint32_t stype;
    const void* pNext;
    uint32_t flags;
    uint32_t ordinal;
};
struct ze_host_mem_alloc_desc_t {
    uint32_t stype;
    const void* pNext;
    uint32_t flags;
};

constexpr uint32_t ZE_STRUCTURE_TYPE_CONTEXT_DESC = 1;
constexpr uint32_t ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC = 2;
constexpr uint32_t ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC = 3;
constexpr uint32_t ZE_STRUCTURE_TYPE_FENCE_DESC = 4;
constexpr uint32_t ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES = 10;
constexpr uint32_t ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC = 20;
constexpr uint32_t ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC = 21;

constexpr uint32_t ZE_DEVICE_PROPERTY_FLAG_INTEGRATED = 1u << 0;

// Fields are the documented prefix of the spec struct (stable across
// Level Zero 1.x); only these fields are read.
struct ze_device_properties_t {
    uint32_t stype;
    const void* pNext;
    uint32_t type;
    uint32_t id;
    uint32_t flags;
    uint32_t numMemoryProperties;
    uint32_t memoryClass;
    uint32_t maxMemAllocSize;
    uint32_t memAllocAlignment;
    uint32_t memAllocGranularity;
    uint32_t subdeviceId;
    uint32_t coreClockRate;
    uint32_t maxHardwareContexts;
    uint32_t maxCommandQueuePriority;
    uint32_t numThreadsPerEU;
    uint32_t numPhysicalEUs;
    uint32_t numEUsPerSubslice;
    uint32_t numSubslicesPerSlice;
    uint32_t numSlices;
    uint32_t physicalEUSimdWidth;
    uint32_t numAsyncComputeEngines;
    uint32_t numAsyncCopyEngines;
    uint32_t numAsyncWarriorEngines;
    uint32_t numAsyncCommandQueues;
    uint32_t maxCommandQueues;
    uint32_t baseProfile;
    uint32_t reserved[16];
};

struct ze_device_memory_properties_t {
    uint32_t stype;
    const void* pNext;
    uint32_t flags;
    char memoryType[256];
    uint32_t location;
    uint32_t busWidth;
    uint32_t numChannels;
    uint32_t bandwidth;
    uint32_t maxClockRate;
    uint32_t physicalSize;
    uint32_t reserved[4];
};

// Function pointer table resolved from the loader.
struct ZeApi {
    ze_result_t (*zeInit)(uint32_t flags) = nullptr;
    ze_result_t (*zeDriverGet)(uint32_t* pCount, ze_driver_handle_t* phDrivers) = nullptr;
    ze_result_t (*zeDriverGetApiVersion)(ze_driver_handle_t, uint32_t* version) = nullptr;
    ze_result_t (*zeDeviceGet)(ze_driver_handle_t, uint32_t* pCount, ze_device_handle_t* phDevices) = nullptr;
    ze_result_t (*zeDeviceGetProperties)(ze_device_handle_t, ze_device_properties_t* pProperties) = nullptr;
    ze_result_t (*zeDeviceGetMemoryProperties)(ze_device_handle_t, uint32_t* pCount, ze_device_memory_properties_t* pMemProperties) = nullptr;
    ze_result_t (*zeContextCreate)(ze_driver_handle_t, const ze_context_desc_t*, ze_context_handle_t*) = nullptr;
    ze_result_t (*zeContextDestroy)(ze_context_handle_t) = nullptr;
    ze_result_t (*zeCommandQueueCreate)(ze_context_handle_t, ze_device_handle_t, const ze_command_queue_desc_t*, ze_command_queue_handle_t*) = nullptr;
    ze_result_t (*zeCommandQueueDestroy)(ze_command_queue_handle_t) = nullptr;
    ze_result_t (*zeCommandQueueExecuteCommandLists)(ze_command_queue_handle_t, uint32_t, ze_command_list_handle_t*, ze_fence_handle_t) = nullptr;
    ze_result_t (*zeCommandQueueSynchronize)(ze_command_queue_handle_t, uint64_t timeout) = nullptr;
    ze_result_t (*zeCommandListCreate)(ze_context_handle_t, ze_device_handle_t, const ze_command_list_desc_t*, ze_command_list_handle_t*) = nullptr;
    ze_result_t (*zeCommandListDestroy)(ze_command_list_handle_t) = nullptr;
    ze_result_t (*zeCommandListAppendMemoryCopy)(ze_command_list_handle_t, void* dstptr, const void* srcptr, size_t size, ze_event_handle_t hSignalEvent, uint32_t numWaitEvents, ze_event_handle_t* phWaitEvents) = nullptr;
    ze_result_t (*zeCommandListClose)(ze_command_list_handle_t) = nullptr;
    ze_result_t (*zeCommandListReset)(ze_command_list_handle_t) = nullptr;
    ze_result_t (*zeFenceCreate)(ze_command_queue_handle_t, const ze_fence_desc_t*, ze_fence_handle_t*) = nullptr;
    ze_result_t (*zeFenceDestroy)(ze_fence_handle_t) = nullptr;
    ze_result_t (*zeFenceHostSynchronize)(ze_fence_handle_t, uint64_t timeout) = nullptr;
    ze_result_t (*zeMemAllocDevice)(ze_context_handle_t, const ze_device_mem_alloc_desc_t*, size_t size, size_t alignment, ze_device_handle_t, void** pptr) = nullptr;
    ze_result_t (*zeMemAllocHost)(ze_context_handle_t, const ze_host_mem_alloc_desc_t*, size_t size, size_t alignment, void** pptr) = nullptr;
    ze_result_t (*zeMemAllocShared)(ze_context_handle_t, const ze_device_mem_alloc_desc_t*, const ze_host_mem_alloc_desc_t*, size_t size, size_t alignment, ze_device_handle_t, void** pptr) = nullptr;
    ze_result_t (*zeMemFree)(ze_context_handle_t, void* ptr) = nullptr;

    bool loaded = false;
};

void* resolve_symbol(void* handle, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name));
#else
    return dlsym(handle, name);
#endif
}

std::string ze_result_text(ze_result_t r) {
    switch (r) {
        case ZE_RESULT_SUCCESS: return "success";
        case 0x78000001: return "not ready";
        case 0x78000002: return "error device lost";
        case 0x78000003: return "error out of host memory";
        case 0x78000004: return "error out of device memory";
        case 0x78000005: return "error module build failure";
        case 0x78000006: return "error module link failure";
        case 0x78000009: return "error device unavailable";
        case 0x7800000b: return "error invalid argument";
        case 0x78000010: return "error device not found";
        case 0x7800000c: return "error invalid null handle";
        default: return "ze_result 0x" + [&]() {
            char buf[16];
            std::snprintf(buf, sizeof(buf), "%08X", r);
            return std::string(buf);
        }();
    }
}

std::string load_failure_reason() {
#if defined(_WIN32)
    const DWORD err = GetLastError();
    return "level zero loader (ze_loader.dll) not loadable, error " +
           std::to_string(err);
#else
    return "level zero loader (libze_loader.so) not loadable: " +
           std::string(dlerror() != nullptr ? dlerror() : "unknown");
#endif
}

}  // namespace

class LzStream final : public DeviceStream {
public:
    ze_command_queue_handle_t queue = nullptr;
    ze_command_list_handle_t list = nullptr;
    ze_fence_handle_t fence = nullptr;
};

class LzEvent final : public DeviceEvent {
public:
    // Timing through events requires a query pool; v0.1 uses host-side
    // timing around queue synchronization instead and reports
    // event_timing = false. This handle exists for contract uniformity.
};

struct LevelZeroBackend::Impl {
    ZeApi api;
    bool loader_tried = false;
    std::string unavailable_reason;

    ze_driver_handle_t driver = nullptr;
    ze_device_handle_t device = nullptr;
    ze_context_handle_t context = nullptr;
    int device_index = -1;
    bool opened = false;

    DeviceInfo info;
    DeviceCapabilities caps;
    uint32_t api_version = 0;

    std::mutex mu;
    std::string last_error;
};

LevelZeroBackend::LevelZeroBackend() : impl_(std::make_unique<Impl>()) {}

LevelZeroBackend::~LevelZeroBackend() {
    close();
}

namespace {

void ensure_loader(LevelZeroBackend::Impl& impl) {
    if (impl.loader_tried) return;
    impl.loader_tried = true;
#if defined(_WIN32)
    HMODULE h = LoadLibraryA("ze_loader.dll");
    if (h == nullptr) {
        impl.unavailable_reason = load_failure_reason();
        return;
    }
#else
    void* h = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_LOCAL);
    if (h == nullptr) {
        h = dlopen("libze_loader.so", RTLD_NOW | RTLD_LOCAL);
    }
    if (h == nullptr) {
        impl.unavailable_reason = load_failure_reason();
        return;
    }
#endif
    impl.api.zeInit = reinterpret_cast<decltype(impl.api.zeInit)>(resolve_symbol(h, "zeInit"));
    impl.api.zeDriverGet = reinterpret_cast<decltype(impl.api.zeDriverGet)>(resolve_symbol(h, "zeDriverGet"));
    impl.api.zeDriverGetApiVersion = reinterpret_cast<decltype(impl.api.zeDriverGetApiVersion)>(resolve_symbol(h, "zeDriverGetApiVersion"));
    impl.api.zeDeviceGet = reinterpret_cast<decltype(impl.api.zeDeviceGet)>(resolve_symbol(h, "zeDeviceGet"));
    impl.api.zeDeviceGetProperties = reinterpret_cast<decltype(impl.api.zeDeviceGetProperties)>(resolve_symbol(h, "zeDeviceGetProperties"));
    impl.api.zeDeviceGetMemoryProperties = reinterpret_cast<decltype(impl.api.zeDeviceGetMemoryProperties)>(resolve_symbol(h, "zeDeviceGetMemoryProperties"));
    impl.api.zeContextCreate = reinterpret_cast<decltype(impl.api.zeContextCreate)>(resolve_symbol(h, "zeContextCreate"));
    impl.api.zeContextDestroy = reinterpret_cast<decltype(impl.api.zeContextDestroy)>(resolve_symbol(h, "zeContextDestroy"));
    impl.api.zeCommandQueueCreate = reinterpret_cast<decltype(impl.api.zeCommandQueueCreate)>(resolve_symbol(h, "zeCommandQueueCreate"));
    impl.api.zeCommandQueueDestroy = reinterpret_cast<decltype(impl.api.zeCommandQueueDestroy)>(resolve_symbol(h, "zeCommandQueueDestroy"));
    impl.api.zeCommandQueueExecuteCommandLists = reinterpret_cast<decltype(impl.api.zeCommandQueueExecuteCommandLists)>(resolve_symbol(h, "zeCommandQueueExecuteCommandLists"));
    impl.api.zeCommandQueueSynchronize = reinterpret_cast<decltype(impl.api.zeCommandQueueSynchronize)>(resolve_symbol(h, "zeCommandQueueSynchronize"));
    impl.api.zeCommandListCreate = reinterpret_cast<decltype(impl.api.zeCommandListCreate)>(resolve_symbol(h, "zeCommandListCreate"));
    impl.api.zeCommandListDestroy = reinterpret_cast<decltype(impl.api.zeCommandListDestroy)>(resolve_symbol(h, "zeCommandListDestroy"));
    impl.api.zeCommandListAppendMemoryCopy = reinterpret_cast<decltype(impl.api.zeCommandListAppendMemoryCopy)>(resolve_symbol(h, "zeCommandListAppendMemoryCopy"));
    impl.api.zeCommandListClose = reinterpret_cast<decltype(impl.api.zeCommandListClose)>(resolve_symbol(h, "zeCommandListClose"));
    impl.api.zeCommandListReset = reinterpret_cast<decltype(impl.api.zeCommandListReset)>(resolve_symbol(h, "zeCommandListReset"));
    impl.api.zeFenceCreate = reinterpret_cast<decltype(impl.api.zeFenceCreate)>(resolve_symbol(h, "zeFenceCreate"));
    impl.api.zeFenceDestroy = reinterpret_cast<decltype(impl.api.zeFenceDestroy)>(resolve_symbol(h, "zeFenceDestroy"));
    impl.api.zeFenceHostSynchronize = reinterpret_cast<decltype(impl.api.zeFenceHostSynchronize)>(resolve_symbol(h, "zeFenceHostSynchronize"));
    impl.api.zeMemAllocDevice = reinterpret_cast<decltype(impl.api.zeMemAllocDevice)>(resolve_symbol(h, "zeMemAllocDevice"));
    impl.api.zeMemAllocHost = reinterpret_cast<decltype(impl.api.zeMemAllocHost)>(resolve_symbol(h, "zeMemAllocHost"));
    impl.api.zeMemAllocShared = reinterpret_cast<decltype(impl.api.zeMemAllocShared)>(resolve_symbol(h, "zeMemAllocShared"));
    impl.api.zeMemFree = reinterpret_cast<decltype(impl.api.zeMemFree)>(resolve_symbol(h, "zeMemFree"));
    impl.api.loaded = impl.api.zeInit != nullptr && impl.api.zeDriverGet != nullptr &&
                      impl.api.zeDeviceGet != nullptr && impl.api.zeDeviceGetProperties != nullptr &&
                      impl.api.zeContextCreate != nullptr && impl.api.zeCommandQueueCreate != nullptr &&
                      impl.api.zeCommandListCreate != nullptr &&
                      impl.api.zeCommandListAppendMemoryCopy != nullptr &&
                      impl.api.zeCommandQueueExecuteCommandLists != nullptr &&
                      impl.api.zeCommandQueueSynchronize != nullptr &&
                      impl.api.zeMemAllocDevice != nullptr && impl.api.zeMemFree != nullptr;
    if (!impl.api.loaded) {
        impl.unavailable_reason = "level zero loader present but incomplete (missing symbols)";
    }
}

void check_ze(ze_result_t r, const char* what, LevelZeroBackend::Impl& impl) {
    if (r != ZE_RESULT_SUCCESS) {
        const std::string text = ze_result_text(r);
        impl.last_error = std::string(what) + ": " + text;
        throw Error(ErrorCode::Device, std::string(what) + " failed: " + text,
                    "ze_result 0x" + [&]() {
                        char buf[16];
                        std::snprintf(buf, sizeof(buf), "%08X", r);
                        return std::string(buf);
                    }());
    }
}

}  // namespace

std::vector<DeviceInfo> LevelZeroBackend::enumerate_devices() const {
    ensure_loader(*impl_);
    if (!impl_->api.loaded) {
        return {};
    }
    std::vector<DeviceInfo> out;
    ze_result_t r = impl_->api.zeInit(0);
    if (r != ZE_RESULT_SUCCESS) {
        impl_->last_error = "zeInit: " + ze_result_text(r);
        return {};
    }
    uint32_t driver_count = 0;
    r = impl_->api.zeDriverGet(&driver_count, nullptr);
    if (r != ZE_RESULT_SUCCESS || driver_count == 0) {
        return {};
    }
    std::vector<ze_driver_handle_t> drivers(driver_count);
    r = impl_->api.zeDriverGet(&driver_count, drivers.data());
    if (r != ZE_RESULT_SUCCESS) {
        return {};
    }
    for (ze_driver_handle_t driver : drivers) {
        uint32_t device_count = 0;
        r = impl_->api.zeDeviceGet(driver, &device_count, nullptr);
        if (r != ZE_RESULT_SUCCESS || device_count == 0) continue;
        std::vector<ze_device_handle_t> devices(device_count);
        r = impl_->api.zeDeviceGet(driver, &device_count, devices.data());
        if (r != ZE_RESULT_SUCCESS) continue;
        for (uint32_t i = 0; i < device_count; ++i) {
            ze_device_properties_t props{};
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            if (impl_->api.zeDeviceGetProperties(devices[i], &props) != ZE_RESULT_SUCCESS) {
                continue;
            }
            DeviceInfo info;
            info.id = "level_zero:" + std::to_string(out.size());
            info.name = "Intel Level Zero device (id " + std::to_string(props.id) + ")";
            info.architecture = "level_zero";
            info.total_memory = 0;
            uint32_t mem_count = 0;
            if (impl_->api.zeDeviceGetMemoryProperties(devices[i], &mem_count, nullptr) ==
                    ZE_RESULT_SUCCESS &&
                mem_count > 0) {
                std::vector<ze_device_memory_properties_t> mems(mem_count);
                if (impl_->api.zeDeviceGetMemoryProperties(devices[i], &mem_count, mems.data()) ==
                    ZE_RESULT_SUCCESS) {
                    for (const auto& m : mems) {
                        if (m.flags & 1u) {  // ZE_DEVICE_MEMORY_PROPERTY_FLAG_TBD_LOCAL
                            info.total_memory += m.physicalSize;
                        }
                    }
                }
            }
            if (info.total_memory == 0) {
                info.total_memory = static_cast<uint64_t>(props.maxMemAllocSize);
            }
            info.free_memory = info.total_memory;
            info.discrete = (props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) == 0;
            info.memory_shared = !info.discrete;
            info.index = static_cast<int>(out.size());
            out.push_back(std::move(info));
        }
    }
    return out;
}

DeviceCapabilities LevelZeroBackend::probe_capabilities() const {
    DeviceCapabilities caps;
    const std::vector<DeviceInfo> devices = enumerate_devices();
    if (devices.empty()) {
        caps.note = "level zero: no devices discovered";
        return caps;
    }
    caps.backend_available = true;
    caps.device_available = true;
    caps.explicit_allocation = true;
    caps.async_host_to_device = true;
    caps.async_device_to_host = true;
    caps.device_to_device = true;
    caps.pinned_host_allocation = true;
    caps.unified_memory = true;  // zeMemAllocShared
    caps.memory_prefetch = false;
    caps.memory_advice = false;
    caps.event_timing = false;  // host-side timing in v0.1
    caps.alignment_bytes = 64;
    caps.transfer_granularity = 1;
    caps.queue_count = 1;
    caps.note = "integrated GPUs report shared memory; memory_shared from discovery";
    return caps;
}

void LevelZeroBackend::open(int device_index) {
    if (impl_->opened) close();
    ensure_loader(*impl_);
    if (!impl_->api.loaded) {
        throw Error(ErrorCode::Unsupported,
                    "level zero backend unavailable: " + impl_->unavailable_reason);
    }
    const std::vector<DeviceInfo> devices = enumerate_devices();
    if (device_index < 0 || device_index >= static_cast<int>(devices.size())) {
        throw Error(ErrorCode::Config,
                    "level zero backend: requested device index out of range",
                    "device " + std::to_string(device_index) + " of " +
                        std::to_string(devices.size()));
    }
    // Re-resolve handles for the chosen device.
    impl_->device_index = device_index;
    uint32_t driver_count = 0;
    impl_->api.zeDriverGet(&driver_count, nullptr);
    if (driver_count == 0) {
        throw Error(ErrorCode::Unsupported, "level zero backend: no driver");
    }
    std::vector<ze_driver_handle_t> drivers(driver_count);
    impl_->api.zeDriverGet(&driver_count, drivers.data());
    uint32_t device_count = 0;
    impl_->api.zeDeviceGet(drivers[0], &device_count, nullptr);
    std::vector<ze_device_handle_t> devs(device_count);
    impl_->api.zeDeviceGet(drivers[0], &device_count, devs.data());

    impl_->driver = drivers[0];
    impl_->device = devs[static_cast<std::size_t>(device_index)];
    impl_->info = devices[static_cast<std::size_t>(device_index)];

    ze_context_desc_t ctx_desc{};
    ctx_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    check_ze(impl_->api.zeContextCreate(impl_->driver, &ctx_desc, &impl_->context),
             "zeContextCreate", *impl_);

    ze_device_properties_t props{};
    props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    check_ze(impl_->api.zeDeviceGetProperties(impl_->device, &props),
             "zeDeviceGetProperties", *impl_);

    impl_->caps = probe_capabilities();
    impl_->caps.max_allocation_size = static_cast<uint64_t>(props.maxMemAllocSize);
    impl_->caps.alignment_bytes = props.memAllocAlignment != 0 ? props.memAllocAlignment : 64;
    impl_->caps.transfer_granularity = props.memAllocGranularity != 0 ? props.memAllocGranularity : 1;
    impl_->opened = true;
}

void LevelZeroBackend::close() {
    if (!impl_->opened) return;
    if (impl_->api.loaded && impl_->context != nullptr) {
        impl_->api.zeContextDestroy(impl_->context);
    }
    impl_->context = nullptr;
    impl_->driver = nullptr;
    impl_->device = nullptr;
    impl_->opened = false;
}

bool LevelZeroBackend::is_open() const {
    return impl_->opened;
}

DeviceInfo LevelZeroBackend::device_info() const {
    if (!impl_->opened) {
        throw Error(ErrorCode::State, "level zero backend is not open");
    }
    return impl_->info;
}

DeviceCapabilities LevelZeroBackend::capabilities() const {
    if (!impl_->opened) {
        throw Error(ErrorCode::State, "level zero backend is not open");
    }
    return impl_->caps;
}

uint64_t LevelZeroBackend::total_memory() const {
    return impl_->opened ? impl_->info.total_memory : 0;
}

uint64_t LevelZeroBackend::free_memory() const {
    return impl_->opened ? impl_->info.total_memory : 0;
}

void* LevelZeroBackend::allocate(std::size_t bytes) {
    if (bytes == 0) {
        throw Error(ErrorCode::InvalidArgument, "level zero: zero-byte allocation");
    }
    if (bytes > impl_->caps.max_allocation_size) {
        throw Error(ErrorCode::Budget, "level zero: allocation exceeds device limit");
    }
    ze_device_mem_alloc_desc_t desc{};
    desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    void* ptr = nullptr;
    check_ze(impl_->api.zeMemAllocDevice(impl_->context, &desc, bytes, 0,
                                         impl_->device, &ptr),
             "zeMemAllocDevice", *impl_);
    return ptr;
}

void LevelZeroBackend::free(void* ptr) {
    if (ptr == nullptr) return;
    check_ze(impl_->api.zeMemFree(impl_->context, ptr), "zeMemFree", *impl_);
}

void* LevelZeroBackend::allocate_host_pinned(std::size_t bytes) {
    if (bytes == 0) {
        throw Error(ErrorCode::InvalidArgument, "level zero: zero-byte host allocation");
    }
    ze_host_mem_alloc_desc_t desc{};
    desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    void* ptr = nullptr;
    check_ze(impl_->api.zeMemAllocHost(impl_->context, &desc, bytes, 0, &ptr),
             "zeMemAllocHost", *impl_);
    return ptr;
}

void LevelZeroBackend::free_host_pinned(void* ptr) {
    free(ptr);
}

void* LevelZeroBackend::allocate_unified(std::size_t bytes) {
    if (bytes == 0) {
        throw Error(ErrorCode::InvalidArgument, "level zero: zero-byte shared allocation");
    }
    ze_device_mem_alloc_desc_t dev_desc{};
    dev_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    ze_host_mem_alloc_desc_t host_desc{};
    host_desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    void* ptr = nullptr;
    check_ze(impl_->api.zeMemAllocShared(impl_->context, &dev_desc, &host_desc, bytes,
                                         0, impl_->device, &ptr),
             "zeMemAllocShared", *impl_);
    return ptr;
}

void LevelZeroBackend::free_unified(void* ptr) {
    free(ptr);
}

void LevelZeroBackend::prefetch_to_device(void* ptr, std::size_t bytes) {
    (void)ptr;
    (void)bytes;
    throw Error(ErrorCode::Unsupported, "level zero: memory prefetch not supported in v0.1");
}

void LevelZeroBackend::advise_preferred_location(void* ptr, std::size_t bytes) {
    (void)ptr;
    (void)bytes;
    throw Error(ErrorCode::Unsupported, "level zero: memory advice not supported in v0.1");
}

DeviceStream* LevelZeroBackend::create_stream() {
    auto* s = new LzStream();
    ze_command_queue_desc_t qd{};
    qd.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    check_ze(impl_->api.zeCommandQueueCreate(impl_->context, impl_->device, &qd, &s->queue),
             "zeCommandQueueCreate", *impl_);
    ze_command_list_desc_t ld{};
    ld.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    check_ze(impl_->api.zeCommandListCreate(impl_->context, impl_->device, &ld, &s->list),
             "zeCommandListCreate", *impl_);
    ze_fence_desc_t fd{};
    fd.stype = ZE_STRUCTURE_TYPE_FENCE_DESC;
    check_ze(impl_->api.zeFenceCreate(s->queue, &fd, &s->fence),
             "zeFenceCreate", *impl_);
    return s;
}

void LevelZeroBackend::destroy_stream(DeviceStream* stream) {
    if (stream == nullptr) return;
    auto* s = static_cast<LzStream*>(stream);
    if (s->fence) impl_->api.zeFenceDestroy(s->fence);
    if (s->list) impl_->api.zeCommandListDestroy(s->list);
    if (s->queue) impl_->api.zeCommandQueueDestroy(s->queue);
    delete s;
}

DeviceEvent* LevelZeroBackend::create_event() {
    return new LzEvent();
}

void LevelZeroBackend::destroy_event(DeviceEvent* event) {
    delete event;
}

namespace {

void submit_copy(LevelZeroBackend::Impl& impl, LzStream* s, void* dst, const void* src,
                 std::size_t bytes) {
    check_ze(impl.api.zeCommandListReset(s->list), "zeCommandListReset", impl);
    check_ze(impl.api.zeCommandListAppendMemoryCopy(s->list, dst, src, bytes, nullptr, 0, nullptr),
             "zeCommandListAppendMemoryCopy", impl);
    check_ze(impl.api.zeCommandListClose(s->list), "zeCommandListClose", impl);
    ze_command_list_handle_t lists[] = {s->list};
    check_ze(impl.api.zeCommandQueueExecuteCommandLists(s->queue, 1, lists, s->fence),
             "zeCommandQueueExecuteCommandLists", impl);
}

}  // namespace

void LevelZeroBackend::async_copy_host_to_device(void* dst_device, const void* src_host,
                                                 std::size_t bytes, DeviceStream* stream) {
    submit_copy(*impl_, static_cast<LzStream*>(stream), dst_device, src_host, bytes);
}

void LevelZeroBackend::async_copy_device_to_host(void* dst_host, const void* src_device,
                                                 std::size_t bytes, DeviceStream* stream) {
    submit_copy(*impl_, static_cast<LzStream*>(stream), dst_host, src_device, bytes);
}

void LevelZeroBackend::async_copy_device_to_device(void* dst_device, const void* src_device,
                                                   std::size_t bytes, DeviceStream* stream) {
    submit_copy(*impl_, static_cast<LzStream*>(stream), dst_device, src_device, bytes);
}

void LevelZeroBackend::sync_stream(DeviceStream* stream) {
    auto* s = static_cast<LzStream*>(stream);
    check_ze(impl_->api.zeFenceHostSynchronize(s->fence, UINT64_MAX),
             "zeFenceHostSynchronize", *impl_);
    check_ze(impl_->api.zeCommandQueueSynchronize(s->queue, UINT64_MAX),
             "zeCommandQueueSynchronize", *impl_);
}

void LevelZeroBackend::sync_all() {
    if (impl_->api.loaded && impl_->context != nullptr) {
        // No global sync API in the used subset; open streams own their
        // fences, so sync_all is a no-op beyond per-stream sync.
    }
}

void LevelZeroBackend::record_event(DeviceEvent* event, DeviceStream* stream) {
    (void)event;
    (void)stream;
    throw Error(ErrorCode::Unsupported,
                "level zero: event recording not supported in v0.1 (host timing)");
}

void LevelZeroBackend::wait_event(DeviceEvent* event) {
    (void)event;
    throw Error(ErrorCode::Unsupported,
                "level zero: event waiting not supported in v0.1 (host timing)");
}

double LevelZeroBackend::event_elapsed_us(DeviceEvent* start, DeviceEvent* end) {
    (void)start;
    (void)end;
    throw Error(ErrorCode::Unsupported,
                "level zero: event timing not supported in v0.1");
}

bool LevelZeroBackend::healthy() const {
    return impl_->opened;
}

std::string LevelZeroBackend::diagnostics() const {
    return "level zero backend: " +
           (impl_->api.loaded ? "loader present, api version " +
                                    std::to_string(impl_->api_version)
                              : "unavailable (" + impl_->unavailable_reason + ")") +
           (impl_->opened ? ", device " + impl_->info.id : ", not open");
}

std::string LevelZeroBackend::last_error() const {
    std::lock_guard lock(impl_->mu);
    return impl_->last_error;
}

void register_level_zero_backend(BackendRegistry& registry) {
    registry.register_backend(
        "level_zero", [] { return std::make_unique<LevelZeroBackend>(); },
        "Intel (Level Zero)");
}

}  // namespace flashtier
