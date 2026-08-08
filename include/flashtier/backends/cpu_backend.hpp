#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "flashtier/backends/backend_registry.hpp"
#include "flashtier/backends/device_backend.hpp"

namespace flashtier {

// CPU emulation backend. Treats host RAM as "device memory" through aligned
// allocations and memcpy-based transfers. This is a diagnostic/emulation
// path (--backend cpu) and the basis of portable conformance runs; it never
// claims GPU execution. Memory is host-coherent by definition.
//
// The governed runtime does not use this backend as a device tier in
// default CPU-only mode: auto-selection returns "cpu" only when no
// accelerator backend exists, and the runtime then runs the host+NVMe tiers
// without a device tier (see Runtime::start).
class CpuBackend final : public DeviceBackend {
public:
    CpuBackend();
    ~CpuBackend() override;

    CpuBackend(const CpuBackend&) = delete;
    CpuBackend& operator=(const CpuBackend&) = delete;

    std::string backend_name() const override { return "cpu"; }
    BackendVendor vendor() const override { return BackendVendor::Portable; }
    std::string vendor_display_name() const override { return "Portable (CPU emulation)"; }

    std::vector<DeviceInfo> enumerate_devices() const override;
    DeviceCapabilities probe_capabilities() const override;

    void open(int device_index) override;
    void close() override;
    bool is_open() const override;

    DeviceInfo device_info() const override;
    DeviceCapabilities capabilities() const override;
    uint64_t total_memory() const override;
    uint64_t free_memory() const override;

    void* allocate(std::size_t bytes) override;
    void free(void* ptr) override;
    void* allocate_host_pinned(std::size_t bytes) override;
    void free_host_pinned(void* ptr) override;
    void* allocate_unified(std::size_t bytes) override;
    void free_unified(void* ptr) override;
    void prefetch_to_device(void* ptr, std::size_t bytes) override;
    void advise_preferred_location(void* ptr, std::size_t bytes) override;

    DeviceStream* create_stream() override;
    void destroy_stream(DeviceStream* stream) override;
    DeviceEvent* create_event() override;
    void destroy_event(DeviceEvent* event) override;

    void async_copy_host_to_device(void* dst_device, const void* src_host,
                                   std::size_t bytes, DeviceStream* stream) override;
    void async_copy_device_to_host(void* dst_host, const void* src_device,
                                   std::size_t bytes, DeviceStream* stream) override;
    void async_copy_device_to_device(void* dst_device, const void* src_device,
                                     std::size_t bytes, DeviceStream* stream) override;

    void sync_stream(DeviceStream* stream) override;
    void sync_all() override;
    void record_event(DeviceEvent* event, DeviceStream* stream) override;
    void wait_event(DeviceEvent* event) override;
    double event_elapsed_us(DeviceEvent* start, DeviceEvent* end) override;

    bool healthy() const override;
    std::string diagnostics() const override;
    std::string last_error() const override;

private:
    void require_open_locked() const;
    bool owns_range_locked(const void* ptr, std::size_t bytes) const;
    void release_owned_resources_locked() noexcept;

    mutable std::mutex mu_;
    DeviceInfo info_;
    DeviceCapabilities caps_;
    bool opened_ = false;
    uint64_t allocated_bytes_ = 0;
    std::unordered_map<void*, uint64_t> sizes_;
    std::unordered_set<DeviceStream*> streams_;
    std::unordered_set<DeviceEvent*> events_;
};

void register_cpu_backend(BackendRegistry& registry);

}  // namespace flashtier
