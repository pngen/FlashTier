#pragma once

#include <memory>
#include <string>

#include "flashtier/backends/backend_registry.hpp"
#include "flashtier/backends/device_backend.hpp"

namespace flashtier {

// AMD GPU backend through HIP/ROCm.
//
// Compile-gated by FLASHTIER_ENABLE_HIP (requires a HIP/ROCm toolchain).
// Implements the same vendor-neutral DeviceBackend contract as the CUDA
// backend; HIP errors map to ErrorCode::Device typed errors. Local
// validation status: depends on installed AMD hardware/ROCm; nothing is
// claimed without real hardware execution.
class HipBackend final : public DeviceBackend {
public:
    HipBackend();
    ~HipBackend() override;

    HipBackend(const HipBackend&) = delete;
    HipBackend& operator=(const HipBackend&) = delete;

    std::string backend_name() const override { return "hip"; }
    BackendVendor vendor() const override { return BackendVendor::Amd; }
    std::string vendor_display_name() const override { return "AMD (HIP/ROCm)"; }

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
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

void register_hip_backend(BackendRegistry& registry);

}  // namespace flashtier
