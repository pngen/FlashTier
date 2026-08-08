// Apple Metal backend fail-closed placeholder (macOS only).
//
// The previous implementation returned retained Objective-C MTLBuffer objects
// where DeviceBackend requires writable host byte pointers, and interpreted
// ordinary host pointers as Objective-C objects during transfers. Keep the
// backend registered on Apple platforms for configuration compatibility, but
// do not enumerate or open it until those semantics pass real-hardware
// conformance.

#if defined(__APPLE__)

#include "flashtier/backends/metal_backend.hpp"

#include <mutex>

#include "flashtier/error.hpp"

namespace flashtier {

namespace {

constexpr const char* kUnavailableReason =
    "disabled: Metal DeviceBackend memory semantics are not validated";

[[noreturn]] void throw_unavailable() {
    throw Error(ErrorCode::Unsupported,
                std::string("metal backend ") + kUnavailableReason);
}

}  // namespace

struct MetalBackend::Impl {
    mutable std::mutex mu;
    std::string last_error;
};

MetalBackend::MetalBackend() : impl_(std::make_unique<Impl>()) {}
MetalBackend::~MetalBackend() = default;

std::vector<DeviceInfo> MetalBackend::enumerate_devices() const { return {}; }

DeviceCapabilities MetalBackend::probe_capabilities() const {
    DeviceCapabilities caps;
    caps.note = std::string("metal: ") + kUnavailableReason;
    return caps;
}

void MetalBackend::open(int device_index) {
    (void)device_index;
    {
        std::lock_guard lock(impl_->mu);
        impl_->last_error = kUnavailableReason;
    }
    throw_unavailable();
}

void MetalBackend::close() {}
bool MetalBackend::is_open() const { return false; }
DeviceInfo MetalBackend::device_info() const { throw_unavailable(); }
DeviceCapabilities MetalBackend::capabilities() const { throw_unavailable(); }
uint64_t MetalBackend::total_memory() const { return 0; }
uint64_t MetalBackend::free_memory() const { return 0; }

void* MetalBackend::allocate(std::size_t bytes) {
    (void)bytes;
    throw_unavailable();
}

void MetalBackend::free(void* ptr) {
    if (ptr != nullptr) throw_unavailable();
}

void* MetalBackend::allocate_host_pinned(std::size_t bytes) {
    (void)bytes;
    throw_unavailable();
}

void MetalBackend::free_host_pinned(void* ptr) {
    if (ptr != nullptr) throw_unavailable();
}

void* MetalBackend::allocate_unified(std::size_t bytes) {
    (void)bytes;
    throw_unavailable();
}

void MetalBackend::free_unified(void* ptr) {
    if (ptr != nullptr) throw_unavailable();
}

void MetalBackend::prefetch_to_device(void* ptr, std::size_t bytes) {
    (void)ptr;
    (void)bytes;
    throw_unavailable();
}

void MetalBackend::advise_preferred_location(void* ptr, std::size_t bytes) {
    (void)ptr;
    (void)bytes;
    throw_unavailable();
}

DeviceStream* MetalBackend::create_stream() { throw_unavailable(); }

void MetalBackend::destroy_stream(DeviceStream* stream) {
    if (stream != nullptr) throw_unavailable();
}

DeviceEvent* MetalBackend::create_event() { throw_unavailable(); }

void MetalBackend::destroy_event(DeviceEvent* event) {
    if (event != nullptr) throw_unavailable();
}

void MetalBackend::async_copy_host_to_device(void* dst_device,
                                             const void* src_host,
                                             std::size_t bytes,
                                             DeviceStream* stream) {
    (void)dst_device;
    (void)src_host;
    (void)bytes;
    (void)stream;
    throw_unavailable();
}

void MetalBackend::async_copy_device_to_host(void* dst_host,
                                             const void* src_device,
                                             std::size_t bytes,
                                             DeviceStream* stream) {
    (void)dst_host;
    (void)src_device;
    (void)bytes;
    (void)stream;
    throw_unavailable();
}

void MetalBackend::async_copy_device_to_device(void* dst_device,
                                               const void* src_device,
                                               std::size_t bytes,
                                               DeviceStream* stream) {
    (void)dst_device;
    (void)src_device;
    (void)bytes;
    (void)stream;
    throw_unavailable();
}

void MetalBackend::sync_stream(DeviceStream* stream) {
    (void)stream;
    throw_unavailable();
}

void MetalBackend::sync_all() { throw_unavailable(); }

void MetalBackend::record_event(DeviceEvent* event, DeviceStream* stream) {
    (void)event;
    (void)stream;
    throw_unavailable();
}

void MetalBackend::wait_event(DeviceEvent* event) {
    (void)event;
    throw_unavailable();
}

double MetalBackend::event_elapsed_us(DeviceEvent* start, DeviceEvent* end) {
    (void)start;
    (void)end;
    throw_unavailable();
}

bool MetalBackend::healthy() const { return false; }

std::string MetalBackend::diagnostics() const {
    return std::string("metal backend unavailable: ") + kUnavailableReason;
}

std::string MetalBackend::last_error() const {
    std::lock_guard lock(impl_->mu);
    return impl_->last_error;
}

void register_metal_backend(BackendRegistry& registry) {
    registry.register_backend(
        "metal", [] { return std::make_unique<MetalBackend>(); },
        "Apple Metal (disabled pending contract-safe implementation)");
}

}  // namespace flashtier

#endif  // __APPLE__
