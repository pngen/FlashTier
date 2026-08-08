// Intel Level Zero backend fail-closed placeholder.
//
// The previous implementation called the Level Zero loader through locally
// declared ABI structures.  Those declarations did not match the official
// ABI and could let the loader write beyond the supplied objects.  Keep this
// backend registered for configuration compatibility, but never call into a
// Level Zero loader until the implementation is built against the official
// headers and passes the DeviceBackend conformance battery on real hardware.

#include "flashtier/backends/level_zero_backend.hpp"

#include <mutex>
#include <utility>

#include "flashtier/error.hpp"

namespace flashtier {

namespace {

constexpr const char* kUnavailableReason =
    "disabled: Level Zero ABI/runtime implementation is not validated";

[[noreturn]] void throw_unavailable() {
    throw Error(ErrorCode::Unsupported,
                std::string("level_zero backend ") + kUnavailableReason);
}

}  // namespace

struct LevelZeroBackend::Impl {
    mutable std::mutex mu;
    std::string last_error;
};

LevelZeroBackend::LevelZeroBackend() : impl_(std::make_unique<Impl>()) {}

LevelZeroBackend::~LevelZeroBackend() = default;

std::vector<DeviceInfo> LevelZeroBackend::enumerate_devices() const {
    return {};
}

DeviceCapabilities LevelZeroBackend::probe_capabilities() const {
    DeviceCapabilities caps;
    caps.note = std::string("level_zero: ") + kUnavailableReason;
    return caps;
}

void LevelZeroBackend::open(int device_index) {
    (void)device_index;
    {
        std::lock_guard lock(impl_->mu);
        impl_->last_error = kUnavailableReason;
    }
    throw_unavailable();
}

void LevelZeroBackend::close() {}

bool LevelZeroBackend::is_open() const {
    return false;
}

DeviceInfo LevelZeroBackend::device_info() const {
    throw_unavailable();
}

DeviceCapabilities LevelZeroBackend::capabilities() const {
    throw_unavailable();
}

uint64_t LevelZeroBackend::total_memory() const {
    return 0;
}

uint64_t LevelZeroBackend::free_memory() const {
    return 0;
}

void* LevelZeroBackend::allocate(std::size_t bytes) {
    (void)bytes;
    throw_unavailable();
}

void LevelZeroBackend::free(void* ptr) {
    if (ptr != nullptr) throw_unavailable();
}

void* LevelZeroBackend::allocate_host_pinned(std::size_t bytes) {
    (void)bytes;
    throw_unavailable();
}

void LevelZeroBackend::free_host_pinned(void* ptr) {
    if (ptr != nullptr) throw_unavailable();
}

void* LevelZeroBackend::allocate_unified(std::size_t bytes) {
    (void)bytes;
    throw_unavailable();
}

void LevelZeroBackend::free_unified(void* ptr) {
    if (ptr != nullptr) throw_unavailable();
}

void LevelZeroBackend::prefetch_to_device(void* ptr, std::size_t bytes) {
    (void)ptr;
    (void)bytes;
    throw_unavailable();
}

void LevelZeroBackend::advise_preferred_location(void* ptr, std::size_t bytes) {
    (void)ptr;
    (void)bytes;
    throw_unavailable();
}

DeviceStream* LevelZeroBackend::create_stream() {
    throw_unavailable();
}

void LevelZeroBackend::destroy_stream(DeviceStream* stream) {
    if (stream != nullptr) throw_unavailable();
}

DeviceEvent* LevelZeroBackend::create_event() {
    throw_unavailable();
}

void LevelZeroBackend::destroy_event(DeviceEvent* event) {
    if (event != nullptr) throw_unavailable();
}

void LevelZeroBackend::async_copy_host_to_device(void* dst_device,
                                                 const void* src_host,
                                                 std::size_t bytes,
                                                 DeviceStream* stream) {
    (void)dst_device;
    (void)src_host;
    (void)bytes;
    (void)stream;
    throw_unavailable();
}

void LevelZeroBackend::async_copy_device_to_host(void* dst_host,
                                                 const void* src_device,
                                                 std::size_t bytes,
                                                 DeviceStream* stream) {
    (void)dst_host;
    (void)src_device;
    (void)bytes;
    (void)stream;
    throw_unavailable();
}

void LevelZeroBackend::async_copy_device_to_device(void* dst_device,
                                                   const void* src_device,
                                                   std::size_t bytes,
                                                   DeviceStream* stream) {
    (void)dst_device;
    (void)src_device;
    (void)bytes;
    (void)stream;
    throw_unavailable();
}

void LevelZeroBackend::sync_stream(DeviceStream* stream) {
    (void)stream;
    throw_unavailable();
}

void LevelZeroBackend::sync_all() {
    throw_unavailable();
}

void LevelZeroBackend::record_event(DeviceEvent* event, DeviceStream* stream) {
    (void)event;
    (void)stream;
    throw_unavailable();
}

void LevelZeroBackend::wait_event(DeviceEvent* event) {
    (void)event;
    throw_unavailable();
}

double LevelZeroBackend::event_elapsed_us(DeviceEvent* start, DeviceEvent* end) {
    (void)start;
    (void)end;
    throw_unavailable();
}

bool LevelZeroBackend::healthy() const {
    return false;
}

std::string LevelZeroBackend::diagnostics() const {
    return std::string("level_zero backend unavailable: ") + kUnavailableReason;
}

std::string LevelZeroBackend::last_error() const {
    std::lock_guard lock(impl_->mu);
    return impl_->last_error;
}

void register_level_zero_backend(BackendRegistry& registry) {
    registry.register_backend(
        "level_zero", [] { return std::make_unique<LevelZeroBackend>(); },
        "Intel Level Zero (disabled pending ABI-safe implementation)");
}

}  // namespace flashtier
