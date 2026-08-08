#include "flashtier/backends/cpu_backend.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>

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
#include <unistd.h>
#endif

namespace flashtier {

namespace {

// One emulated "device": the local machine. Memory figures come from the OS
// at probe time (discovery, not guessing).
uint64_t physical_ram_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    return GlobalMemoryStatusEx(&ms) ? ms.ullTotalPhys : 0;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return pages > 0 && page_size > 0
               ? static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size)
               : 0;
#endif
}

void* aligned_alloc_bytes(std::size_t bytes) {
#if defined(_WIN32)
    return _aligned_malloc(bytes, 4096);
#else
    void* ptr = nullptr;
    return posix_memalign(&ptr, 4096, bytes) == 0 ? ptr : nullptr;
#endif
}

void aligned_free_bytes(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

class CpuStream final : public DeviceStream {
public:
    std::chrono::steady_clock::time_point last_timestamp{};
};

class CpuEvent final : public DeviceEvent {
public:
    std::chrono::steady_clock::time_point recorded{};
    bool was_recorded = false;
};

}  // namespace

CpuBackend::CpuBackend() {
    caps_.backend_available = true;
    caps_.explicit_allocation = true;
    caps_.async_host_to_device = true;
    caps_.async_device_to_host = true;
    caps_.device_to_device = true;
    caps_.pinned_host_allocation = true;
    caps_.unified_memory = true;  // device memory IS system memory here
    caps_.host_coherent_memory = true;
    caps_.memory_prefetch = true;   // no-op: memory is already host-coherent
    caps_.memory_advice = true;     // no-op: memory is already host-coherent
    caps_.event_timing = true;
    caps_.max_allocation_size = 1ull << 30;  // emulation cap: 1 GiB
    caps_.alignment_bytes = 4096;
    caps_.transfer_granularity = 1;
    caps_.queue_count = 4;
    caps_.note = "CPU emulation: memory is shared system RAM; no GPU execution claimed";
}

CpuBackend::~CpuBackend() {
    close();
}

std::vector<DeviceInfo> CpuBackend::enumerate_devices() const {
    DeviceInfo info;
    info.id = "cpu:0";
    info.name = "CPU emulation device";
    info.architecture = "portable";
    info.total_memory = physical_ram_bytes();
    info.free_memory = info.total_memory;  // OS-managed; free varies
    info.discrete = false;
    info.memory_shared = true;
    info.index = 0;
    return {info};
}

DeviceCapabilities CpuBackend::probe_capabilities() const {
    DeviceCapabilities caps = caps_;
    caps.device_available = physical_ram_bytes() > 0;
    return caps;
}

void CpuBackend::open(int device_index) {
    if (device_index != 0) {
        throw Error(ErrorCode::Config,
                    "cpu backend has exactly one device (index 0)",
                    "requested index " + std::to_string(device_index));
    }
    const DeviceInfo selected = enumerate_devices()[0];
    std::lock_guard lock(mu_);
    release_owned_resources_locked();
    info_ = selected;
    opened_ = true;
}

void CpuBackend::close() {
    std::lock_guard lock(mu_);
    release_owned_resources_locked();
    opened_ = false;
}

bool CpuBackend::is_open() const {
    std::lock_guard lock(mu_);
    return opened_;
}

DeviceInfo CpuBackend::device_info() const {
    std::lock_guard lock(mu_);
    if (!opened_) {
        throw Error(ErrorCode::State, "cpu backend is not open");
    }
    return info_;
}

DeviceCapabilities CpuBackend::capabilities() const {
    std::lock_guard lock(mu_);
    require_open_locked();
    DeviceCapabilities caps = caps_;
    caps.device_available = true;
    return caps;
}

uint64_t CpuBackend::total_memory() const {
    std::lock_guard lock(mu_);
    return opened_ ? info_.total_memory : 0;
}

uint64_t CpuBackend::free_memory() const {
    std::lock_guard lock(mu_);
    return opened_ && allocated_bytes_ < info_.total_memory
               ? info_.total_memory - allocated_bytes_
               : 0;
}

void* CpuBackend::allocate(std::size_t bytes) {
    if (bytes == 0) {
        throw Error(ErrorCode::InvalidArgument, "cpu backend: zero-byte allocation");
    }
    if (bytes > caps_.max_allocation_size) {
        throw Error(ErrorCode::Budget,
                    "cpu backend: allocation exceeds emulation limit",
                    std::to_string(bytes) + " > " + std::to_string(caps_.max_allocation_size));
    }
    std::lock_guard lock(mu_);
    require_open_locked();
    // Emulated device memory: total outstanding allocations are capped so
    // budget-exhaustion behavior is testable without touching real limits.
    if (bytes > caps_.max_allocation_size - allocated_bytes_) {
        throw Error(ErrorCode::Budget,
                    "cpu backend: emulated device memory exhausted",
                    std::to_string(bytes) + " requested with " +
                        std::to_string(allocated_bytes_) + " already allocated");
    }
    void* ptr = aligned_alloc_bytes(bytes);
    if (ptr == nullptr) {
        throw Error(ErrorCode::Budget, "cpu backend: allocation failed");
    }
    try {
        sizes_.emplace(ptr, static_cast<uint64_t>(bytes));
    } catch (...) {
        aligned_free_bytes(ptr);
        throw;
    }
    allocated_bytes_ += bytes;
    return ptr;
}

void CpuBackend::free(void* ptr) {
    if (ptr == nullptr) return;
    std::lock_guard lock(mu_);
    require_open_locked();
    auto it = sizes_.find(ptr);
    if (it == sizes_.end()) {
        throw Error(ErrorCode::State, "cpu backend: free of unknown pointer");
    }
    allocated_bytes_ -= it->second;
    sizes_.erase(it);
    aligned_free_bytes(ptr);
}

void* CpuBackend::allocate_host_pinned(std::size_t bytes) {
    return allocate(bytes);  // host-coherent by definition
}

void CpuBackend::free_host_pinned(void* ptr) {
    free(ptr);
}

void* CpuBackend::allocate_unified(std::size_t bytes) {
    return allocate(bytes);
}

void CpuBackend::free_unified(void* ptr) {
    free(ptr);
}

void CpuBackend::prefetch_to_device(void* ptr, std::size_t bytes) {
    std::lock_guard lock(mu_);
    require_open_locked();
    if (ptr == nullptr || bytes == 0 || !owns_range_locked(ptr, bytes)) {
        throw Error(ErrorCode::InvalidArgument,
                    "cpu backend: invalid prefetch range");
    }
    // No-op: memory is already host-coherent.
}

void CpuBackend::advise_preferred_location(void* ptr, std::size_t bytes) {
    std::lock_guard lock(mu_);
    require_open_locked();
    if (ptr == nullptr || bytes == 0 || !owns_range_locked(ptr, bytes)) {
        throw Error(ErrorCode::InvalidArgument,
                    "cpu backend: invalid advice range");
    }
    // No-op: memory is already host-coherent.
}

DeviceStream* CpuBackend::create_stream() {
    std::unique_ptr<DeviceStream> stream = std::make_unique<CpuStream>();
    std::lock_guard lock(mu_);
    require_open_locked();
    streams_.insert(stream.get());
    return stream.release();
}

void CpuBackend::destroy_stream(DeviceStream* stream) {
    if (stream == nullptr) return;
    std::lock_guard lock(mu_);
    if (streams_.erase(stream) == 0) {
        throw Error(ErrorCode::State,
                    "cpu backend: destroy of unknown stream");
    }
    delete stream;
}

DeviceEvent* CpuBackend::create_event() {
    std::unique_ptr<DeviceEvent> event = std::make_unique<CpuEvent>();
    std::lock_guard lock(mu_);
    require_open_locked();
    events_.insert(event.get());
    return event.release();
}

void CpuBackend::destroy_event(DeviceEvent* event) {
    if (event == nullptr) return;
    std::lock_guard lock(mu_);
    if (events_.erase(event) == 0) {
        throw Error(ErrorCode::State,
                    "cpu backend: destroy of unknown event");
    }
    delete event;
}

void CpuBackend::async_copy_host_to_device(void* dst_device, const void* src_host,
                                           std::size_t bytes, DeviceStream* stream) {
    std::lock_guard lock(mu_);
    require_open_locked();
    if (stream != nullptr && streams_.count(stream) == 0) {
        throw Error(ErrorCode::State, "cpu backend: unknown stream");
    }
    if (bytes == 0) return;
    if (src_host == nullptr || !owns_range_locked(dst_device, bytes)) {
        throw Error(ErrorCode::InvalidArgument,
                    "cpu backend: invalid host-to-device copy range");
    }
    std::memcpy(dst_device, src_host, bytes);
    if (stream != nullptr) {
        static_cast<CpuStream*>(stream)->last_timestamp =
            std::chrono::steady_clock::now();
    }
}

void CpuBackend::async_copy_device_to_host(void* dst_host, const void* src_device,
                                           std::size_t bytes, DeviceStream* stream) {
    std::lock_guard lock(mu_);
    require_open_locked();
    if (stream != nullptr && streams_.count(stream) == 0) {
        throw Error(ErrorCode::State, "cpu backend: unknown stream");
    }
    if (bytes == 0) return;
    if (dst_host == nullptr || !owns_range_locked(src_device, bytes)) {
        throw Error(ErrorCode::InvalidArgument,
                    "cpu backend: invalid device-to-host copy range");
    }
    std::memcpy(dst_host, src_device, bytes);
    if (stream != nullptr) {
        static_cast<CpuStream*>(stream)->last_timestamp =
            std::chrono::steady_clock::now();
    }
}

void CpuBackend::async_copy_device_to_device(void* dst_device, const void* src_device,
                                             std::size_t bytes, DeviceStream* stream) {
    std::lock_guard lock(mu_);
    require_open_locked();
    if (stream != nullptr && streams_.count(stream) == 0) {
        throw Error(ErrorCode::State, "cpu backend: unknown stream");
    }
    if (bytes == 0) return;
    if (!owns_range_locked(dst_device, bytes) ||
        !owns_range_locked(src_device, bytes)) {
        throw Error(ErrorCode::InvalidArgument,
                    "cpu backend: invalid device-to-device copy range");
    }
    std::memcpy(dst_device, src_device, bytes);
    if (stream != nullptr) {
        static_cast<CpuStream*>(stream)->last_timestamp =
            std::chrono::steady_clock::now();
    }
}

void CpuBackend::sync_stream(DeviceStream* stream) {
    std::lock_guard lock(mu_);
    require_open_locked();
    if (stream != nullptr && streams_.count(stream) == 0) {
        throw Error(ErrorCode::State, "cpu backend: unknown stream");
    }
    // Copies are synchronous by construction.
}

void CpuBackend::sync_all() {
    std::lock_guard lock(mu_);
    require_open_locked();
    // Copies are synchronous by construction.
}

void CpuBackend::record_event(DeviceEvent* event, DeviceStream* stream) {
    std::lock_guard lock(mu_);
    require_open_locked();
    if (event == nullptr || events_.count(event) == 0) {
        throw Error(ErrorCode::State, "cpu backend: unknown event");
    }
    if (stream != nullptr && streams_.count(stream) == 0) {
        throw Error(ErrorCode::State, "cpu backend: unknown stream");
    }
    auto* e = static_cast<CpuEvent*>(event);
    e->recorded = std::chrono::steady_clock::now();
    e->was_recorded = true;
    if (stream != nullptr) {
        static_cast<CpuStream*>(stream)->last_timestamp = e->recorded;
    }
}

void CpuBackend::wait_event(DeviceEvent* event) {
    std::lock_guard lock(mu_);
    require_open_locked();
    if (event == nullptr || events_.count(event) == 0 ||
        !static_cast<CpuEvent*>(event)->was_recorded) {
        throw Error(ErrorCode::State, "cpu backend: event was not recorded");
    }
    // Already satisfied when recorded (synchronous copies).
}

double CpuBackend::event_elapsed_us(DeviceEvent* start, DeviceEvent* end) {
    std::lock_guard lock(mu_);
    require_open_locked();
    if (start == nullptr || end == nullptr || events_.count(start) == 0 ||
        events_.count(end) == 0 ||
        !static_cast<CpuEvent*>(start)->was_recorded ||
        !static_cast<CpuEvent*>(end)->was_recorded) {
        throw Error(ErrorCode::State,
                    "cpu backend: elapsed time requires recorded events");
    }
    const auto a = static_cast<CpuEvent*>(start)->recorded;
    const auto b = static_cast<CpuEvent*>(end)->recorded;
    return std::chrono::duration<double, std::micro>(b - a).count();
}

std::string CpuBackend::diagnostics() const {
    return "cpu backend: emulation device, memory = " +
           std::to_string(physical_ram_bytes()) + " bytes";
}

std::string CpuBackend::last_error() const {
    return "cpu backend: no vendor errors (host memory)";
}

bool CpuBackend::healthy() const {
    std::lock_guard lock(mu_);
    return opened_;
}

void CpuBackend::require_open_locked() const {
    if (!opened_) {
        throw Error(ErrorCode::State, "cpu backend is not open");
    }
}

bool CpuBackend::owns_range_locked(const void* ptr, std::size_t bytes) const {
    if (ptr == nullptr) return false;
    const auto address = reinterpret_cast<std::uintptr_t>(ptr);
    for (const auto& [base_ptr, allocation_bytes] : sizes_) {
        const auto base = reinterpret_cast<std::uintptr_t>(base_ptr);
        if (address < base) continue;
        const uint64_t offset = static_cast<uint64_t>(address - base);
        if (offset <= allocation_bytes && bytes <= allocation_bytes - offset) {
            return true;
        }
    }
    return false;
}

void CpuBackend::release_owned_resources_locked() noexcept {
    for (DeviceEvent* event : events_) delete event;
    events_.clear();
    for (DeviceStream* stream : streams_) delete stream;
    streams_.clear();
    for (const auto& [ptr, bytes] : sizes_) {
        (void)bytes;
        aligned_free_bytes(ptr);
    }
    sizes_.clear();
    allocated_bytes_ = 0;
}

void register_cpu_backend(BackendRegistry& registry) {
    registry.register_backend(
        "cpu", [] { return std::make_unique<CpuBackend>(); },
        "Portable (CPU emulation)");
}

}  // namespace flashtier
