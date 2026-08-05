// AMD GPU backend through HIP/ROCm. Compiled only when
// FLASHTIER_ENABLE_HIP=ON and a HIP/ROCm toolchain is installed; the file
// is never built otherwise (the module is disabled cleanly).
//
// HIP mirrors the CUDA runtime closely; this module intentionally mirrors
// the CUDA backend structure so both stay conformance-identical.
//
// Local validation status: hardware execution depends on an AMD GPU plus
// ROCm runtime on the validation machine; this environment has neither, so
// this module is written but NOT runtime-validated here.

#include "flashtier/backends/hip_backend.hpp"

#include <cstdint>
#include <mutex>
#include <vector>

#include <hip/hip_runtime.h>

#include "flashtier/error.hpp"

namespace flashtier {

namespace {

void check_hip(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        throw Error(ErrorCode::Device,
                    std::string(what) + " failed: " + hipGetErrorString(e),
                    std::to_string(static_cast<int>(e)));
    }
}

std::string hip_device_arch(const hipDeviceProp_t& p) {
    return std::to_string(p.major) + "." + std::to_string(p.minor);
}

}  // namespace

class HipStream final : public DeviceStream {
public:
    hipStream_t handle = nullptr;
};

class HipEvent final : public DeviceEvent {
public:
    hipEvent_t handle = nullptr;
};

struct HipBackend::Impl {
    int device_index = -1;
    hipDeviceProp_t prop{};
    bool opened = false;
    std::mutex mu;
    std::string last_error;
};

HipBackend::HipBackend() : impl_(std::make_unique<Impl>()) {}

HipBackend::~HipBackend() {
    close();
}

std::vector<DeviceInfo> HipBackend::enumerate_devices() const {
    int count = 0;
    hipError_t e = hipGetDeviceCount(&count);
    if (e != hipSuccess || count <= 0) {
        return {};
    }
    std::vector<DeviceInfo> out;
    for (int i = 0; i < count; ++i) {
        hipDeviceProp_t p{};
        if (hipGetDeviceProperties(&p, i) != hipSuccess) {
            continue;
        }
        DeviceInfo info;
        info.id = "hip:" + std::to_string(i);
        info.name = p.name;
        info.architecture = hip_device_arch(p);
        info.total_memory = static_cast<uint64_t>(p.totalGlobalMem);
        info.free_memory = info.total_memory;
        info.discrete = true;  // HIP/ROCm targets discrete accelerators in practice;
                               // integrated APUs report shared memory via prop
        info.memory_shared = false;
        info.index = i;
        out.push_back(std::move(info));
    }
    return out;
}

DeviceCapabilities HipBackend::probe_capabilities() const {
    DeviceCapabilities caps;
    const std::vector<DeviceInfo> devices = enumerate_devices();
    if (devices.empty()) {
        return caps;
    }
    hipDeviceProp_t p{};
    if (hipGetDeviceProperties(&p, 0) != hipSuccess) {
        return caps;
    }
    caps.backend_available = true;
    caps.device_available = true;
    caps.explicit_allocation = true;
    caps.async_host_to_device = true;
    caps.async_device_to_host = true;
    caps.device_to_device = true;
    caps.pinned_host_allocation = true;
    caps.unified_memory = p.managedMemory != 0;
    caps.concurrent_managed_access = p.concurrentManagedAccess != 0;
    caps.memory_prefetch = p.managedMemory != 0;
    caps.memory_advice = p.managedMemory != 0;
    caps.event_timing = true;
    caps.max_allocation_size = static_cast<uint64_t>(p.totalGlobalMem);
    caps.alignment_bytes = 512;
    caps.transfer_granularity = 1;
    caps.queue_count = 64;
    caps.multi_device = devices.size() > 1;
    caps.note = "explicit allocations; unified memory only when reported by the driver";
    return caps;
}

void HipBackend::open(int device_index) {
    if (impl_->opened) close();
    const std::vector<DeviceInfo> devices = enumerate_devices();
    if (devices.empty()) {
        throw Error(ErrorCode::Unsupported, "hip backend: no HIP device available");
    }
    if (device_index < 0 || device_index >= static_cast<int>(devices.size())) {
        throw Error(ErrorCode::Config, "hip backend: requested device index out of range");
    }
    check_hip(hipSetDevice(device_index), "hipSetDevice");
    check_hip(hipGetDeviceProperties(&impl_->prop, device_index),
              "hipGetDeviceProperties");
    impl_->device_index = device_index;
    impl_->opened = true;
}

void HipBackend::close() {
    if (!impl_->opened) return;
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize");
    impl_->opened = false;
    impl_->device_index = -1;
}

bool HipBackend::is_open() const {
    return impl_->opened;
}

DeviceInfo HipBackend::device_info() const {
    if (!impl_->opened) {
        throw Error(ErrorCode::State, "hip backend is not open");
    }
    DeviceInfo info;
    info.id = "hip:" + std::to_string(impl_->device_index);
    info.name = impl_->prop.name;
    info.architecture = hip_device_arch(impl_->prop);
    info.total_memory = static_cast<uint64_t>(impl_->prop.totalGlobalMem);
    std::size_t free_ = 0;
    std::size_t total_ = 0;
    hipError_t e = hipMemGetInfo(&free_, &total_);
    info.free_memory = e == hipSuccess ? static_cast<uint64_t>(free_) : 0;
    info.discrete = true;
    info.index = impl_->device_index;
    return info;
}

DeviceCapabilities HipBackend::capabilities() const {
    if (!impl_->opened) {
        throw Error(ErrorCode::State, "hip backend is not open");
    }
    return probe_capabilities();
}

uint64_t HipBackend::total_memory() const {
    return impl_->opened ? static_cast<uint64_t>(impl_->prop.totalGlobalMem) : 0;
}

uint64_t HipBackend::free_memory() const {
    if (!impl_->opened) return 0;
    std::size_t free_ = 0;
    std::size_t total_ = 0;
    hipError_t e = hipMemGetInfo(&free_, &total_);
    return e == hipSuccess ? static_cast<uint64_t>(free_) : 0;
}

void* HipBackend::allocate(std::size_t bytes) {
    if (bytes == 0) {
        throw Error(ErrorCode::InvalidArgument, "hip backend: zero-byte allocation");
    }
    if (bytes > impl_->prop.totalGlobalMem) {
        throw Error(ErrorCode::Budget,
                    "hip backend: allocation exceeds device memory limit");
    }
    void* ptr = nullptr;
    check_hip(hipMalloc(&ptr, bytes), "hipMalloc");
    return ptr;
}

void HipBackend::free(void* ptr) {
    if (ptr == nullptr) return;
    check_hip(hipFree(ptr), "hipFree");
}

void* HipBackend::allocate_host_pinned(std::size_t bytes) {
    if (bytes == 0) {
        throw Error(ErrorCode::InvalidArgument,
                    "hip backend: zero-byte pinned allocation");
    }
    void* ptr = nullptr;
    check_hip(hipHostMalloc(&ptr, bytes, hipHostMallocDefault), "hipHostMalloc");
    return ptr;
}

void HipBackend::free_host_pinned(void* ptr) {
    if (ptr == nullptr) return;
    check_hip(hipHostFree(ptr), "hipHostFree");
}

void* HipBackend::allocate_unified(std::size_t bytes) {
    if (impl_->prop.managedMemory == 0) {
        throw Error(ErrorCode::Unsupported,
                    "hip backend: device reports no managed memory support");
    }
    void* ptr = nullptr;
    check_hip(hipMallocManaged(&ptr, bytes), "hipMallocManaged");
    return ptr;
}

void HipBackend::free_unified(void* ptr) {
    if (ptr == nullptr) return;
    check_hip(hipFree(ptr), "hipFree(managed)");
}

void HipBackend::prefetch_to_device(void* ptr, std::size_t bytes) {
    check_hip(hipMemPrefetchAsync(ptr, bytes, impl_->device_index, nullptr),
              "hipMemPrefetchAsync");
}

void HipBackend::advise_preferred_location(void* ptr, std::size_t bytes) {
    check_hip(hipMemAdvise(ptr, bytes, hipMemAdviseSetPreferredLocation,
                           impl_->device_index),
              "hipMemAdvise");
}

DeviceStream* HipBackend::create_stream() {
    auto* s = new HipStream();
    check_hip(hipStreamCreateWithFlags(&s->handle, hipStreamNonBlocking),
              "hipStreamCreate");
    return s;
}

void HipBackend::destroy_stream(DeviceStream* stream) {
    if (stream == nullptr) return;
    auto* s = static_cast<HipStream*>(stream);
    check_hip(hipStreamDestroy(s->handle), "hipStreamDestroy");
    delete s;
}

DeviceEvent* HipBackend::create_event() {
    auto* e = new HipEvent();
    check_hip(hipEventCreateWithFlags(&e->handle, hipEventDefault),
              "hipEventCreate");
    return e;
}

void HipBackend::destroy_event(DeviceEvent* event) {
    if (event == nullptr) return;
    auto* e = static_cast<HipEvent*>(event);
    check_hip(hipEventDestroy(e->handle), "hipEventDestroy");
    delete e;
}

void HipBackend::async_copy_host_to_device(void* dst_device, const void* src_host,
                                           std::size_t bytes, DeviceStream* stream) {
    hipStream_t s = stream != nullptr ? static_cast<HipStream*>(stream)->handle : nullptr;
    check_hip(hipMemcpyAsync(dst_device, src_host, bytes, hipMemcpyHostToDevice, s),
              "hipMemcpyAsync(H2D)");
}

void HipBackend::async_copy_device_to_host(void* dst_host, const void* src_device,
                                           std::size_t bytes, DeviceStream* stream) {
    hipStream_t s = stream != nullptr ? static_cast<HipStream*>(stream)->handle : nullptr;
    check_hip(hipMemcpyAsync(dst_host, src_device, bytes, hipMemcpyDeviceToHost, s),
              "hipMemcpyAsync(D2H)");
}

void HipBackend::async_copy_device_to_device(void* dst_device, const void* src_device,
                                             std::size_t bytes, DeviceStream* stream) {
    hipStream_t s = stream != nullptr ? static_cast<HipStream*>(stream)->handle : nullptr;
    check_hip(hipMemcpyAsync(dst_device, src_device, bytes, hipMemcpyDeviceToDevice, s),
              "hipMemcpyAsync(D2D)");
}

void HipBackend::sync_stream(DeviceStream* stream) {
    hipStream_t s = stream != nullptr ? static_cast<HipStream*>(stream)->handle : nullptr;
    check_hip(hipStreamSynchronize(s), "hipStreamSynchronize");
}

void HipBackend::sync_all() {
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize");
}

void HipBackend::record_event(DeviceEvent* event, DeviceStream* stream) {
    hipStream_t s = stream != nullptr ? static_cast<HipStream*>(stream)->handle : nullptr;
    check_hip(hipEventRecord(static_cast<HipEvent*>(event)->handle, s),
              "hipEventRecord");
}

void HipBackend::wait_event(DeviceEvent* event) {
    check_hip(hipEventSynchronize(static_cast<HipEvent*>(event)->handle),
              "hipEventSynchronize");
}

double HipBackend::event_elapsed_us(DeviceEvent* start, DeviceEvent* end) {
    float ms = 0.0f;
    check_hip(hipEventElapsedTime(&ms, static_cast<HipEvent*>(start)->handle,
                                  static_cast<HipEvent*>(end)->handle),
              "hipEventElapsedTime");
    return static_cast<double>(ms) * 1000.0;
}

bool HipBackend::healthy() const {
    return impl_->opened;
}

std::string HipBackend::diagnostics() const {
    int rt = 0;
    hipRuntimeGetVersion(&rt);
    return "hip backend: runtime " + std::to_string(rt) +
           (impl_->opened ? ", device " + std::string(impl_->prop.name) +
                                " arch " + hip_device_arch(impl_->prop)
                          : ", not open");
}

std::string HipBackend::last_error() const {
    std::lock_guard lock(impl_->mu);
    return impl_->last_error;
}

void register_hip_backend(BackendRegistry& registry) {
    registry.register_backend(
        "hip", [] { return std::make_unique<HipBackend>(); },
        "AMD (HIP/ROCm)");
}

}  // namespace flashtier
