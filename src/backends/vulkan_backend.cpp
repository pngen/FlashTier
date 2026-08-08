// Cross-vendor Vulkan backend fail-closed placeholder.
//
// The previous implementation called the Vulkan loader through a partial
// locally declared ABI and exposed opaque allocation objects as writable host
// pointers. Both violate the DeviceBackend contract and the former can let a
// driver overwrite caller storage. Keep this backend registered for
// configuration compatibility, but never call a Vulkan loader until an
// SDK-backed implementation passes conformance on real hardware.

#include "flashtier/backends/vulkan_backend.hpp"

#include <mutex>

#include "flashtier/error.hpp"

namespace flashtier {

namespace {

constexpr const char* kUnavailableReason =
    "disabled: Vulkan ABI/runtime implementation is not validated";

[[noreturn]] void throw_unavailable() {
    throw Error(ErrorCode::Unsupported,
                std::string("vulkan backend ") + kUnavailableReason);
}

}  // namespace

struct VulkanBackend::Impl {
    mutable std::mutex mu;
    std::string last_error;
};

VulkanBackend::VulkanBackend() : impl_(std::make_unique<Impl>()) {}
VulkanBackend::~VulkanBackend() = default;

std::vector<DeviceInfo> VulkanBackend::enumerate_devices() const { return {}; }

DeviceCapabilities VulkanBackend::probe_capabilities() const {
    DeviceCapabilities caps;
    caps.note = std::string("vulkan: ") + kUnavailableReason;
    return caps;
}

void VulkanBackend::open(int device_index) {
    (void)device_index;
    {
        std::lock_guard lock(impl_->mu);
        impl_->last_error = kUnavailableReason;
    }
    throw_unavailable();
}

void VulkanBackend::close() {}
bool VulkanBackend::is_open() const { return false; }
DeviceInfo VulkanBackend::device_info() const { throw_unavailable(); }
DeviceCapabilities VulkanBackend::capabilities() const { throw_unavailable(); }
uint64_t VulkanBackend::total_memory() const { return 0; }
uint64_t VulkanBackend::free_memory() const { return 0; }

void* VulkanBackend::allocate(std::size_t bytes) {
    (void)bytes;
    throw_unavailable();
}

void VulkanBackend::free(void* ptr) {
    if (ptr != nullptr) throw_unavailable();
}

void* VulkanBackend::allocate_host_pinned(std::size_t bytes) {
    (void)bytes;
    throw_unavailable();
}

void VulkanBackend::free_host_pinned(void* ptr) {
    if (ptr != nullptr) throw_unavailable();
}

void* VulkanBackend::allocate_unified(std::size_t bytes) {
    (void)bytes;
    throw_unavailable();
}

void VulkanBackend::free_unified(void* ptr) {
    if (ptr != nullptr) throw_unavailable();
}

void VulkanBackend::prefetch_to_device(void* ptr, std::size_t bytes) {
    (void)ptr;
    (void)bytes;
    throw_unavailable();
}

void VulkanBackend::advise_preferred_location(void* ptr, std::size_t bytes) {
    (void)ptr;
    (void)bytes;
    throw_unavailable();
}

DeviceStream* VulkanBackend::create_stream() { throw_unavailable(); }

void VulkanBackend::destroy_stream(DeviceStream* stream) {
    if (stream != nullptr) throw_unavailable();
}

DeviceEvent* VulkanBackend::create_event() { throw_unavailable(); }

void VulkanBackend::destroy_event(DeviceEvent* event) {
    if (event != nullptr) throw_unavailable();
}

void VulkanBackend::async_copy_host_to_device(void* dst_device,
                                              const void* src_host,
                                              std::size_t bytes,
                                              DeviceStream* stream) {
    (void)dst_device;
    (void)src_host;
    (void)bytes;
    (void)stream;
    throw_unavailable();
}

void VulkanBackend::async_copy_device_to_host(void* dst_host,
                                              const void* src_device,
                                              std::size_t bytes,
                                              DeviceStream* stream) {
    (void)dst_host;
    (void)src_device;
    (void)bytes;
    (void)stream;
    throw_unavailable();
}

void VulkanBackend::async_copy_device_to_device(void* dst_device,
                                                const void* src_device,
                                                std::size_t bytes,
                                                DeviceStream* stream) {
    (void)dst_device;
    (void)src_device;
    (void)bytes;
    (void)stream;
    throw_unavailable();
}

void VulkanBackend::sync_stream(DeviceStream* stream) {
    (void)stream;
    throw_unavailable();
}

void VulkanBackend::sync_all() { throw_unavailable(); }

void VulkanBackend::record_event(DeviceEvent* event, DeviceStream* stream) {
    (void)event;
    (void)stream;
    throw_unavailable();
}

void VulkanBackend::wait_event(DeviceEvent* event) {
    (void)event;
    throw_unavailable();
}

double VulkanBackend::event_elapsed_us(DeviceEvent* start, DeviceEvent* end) {
    (void)start;
    (void)end;
    throw_unavailable();
}

bool VulkanBackend::healthy() const { return false; }

std::string VulkanBackend::diagnostics() const {
    return std::string("vulkan backend unavailable: ") + kUnavailableReason;
}

std::string VulkanBackend::last_error() const {
    std::lock_guard lock(impl_->mu);
    return impl_->last_error;
}

void register_vulkan_backend(BackendRegistry& registry) {
    registry.register_backend(
        "vulkan", [] { return std::make_unique<VulkanBackend>(); },
        "Vulkan (disabled pending ABI-safe implementation)");
}

}  // namespace flashtier
