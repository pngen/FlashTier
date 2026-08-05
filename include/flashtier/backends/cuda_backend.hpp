#pragma once

#include <memory>
#include <string>

#include "flashtier/backends/backend_registry.hpp"
#include "flashtier/backends/device_backend.hpp"

namespace flashtier {

#if FLASHTIER_HAVE_CUDA

// NVIDIA CUDA backend implementing the vendor-neutral DeviceBackend
// contract. This header is CUDA-header-free: vendor handles live inside
// the implementation (compiled with nvcc).
class CudaBackend final : public DeviceBackend {
public:
    CudaBackend();
    ~CudaBackend() override;

    CudaBackend(const CudaBackend&) = delete;
    CudaBackend& operator=(const CudaBackend&) = delete;

    std::string backend_name() const override { return "cuda"; }
    BackendVendor vendor() const override { return BackendVendor::Nvidia; }
    std::string vendor_display_name() const override { return "NVIDIA (CUDA)"; }

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

    // Backend-specific diagnostics: deterministic device-side fill kernel
    // (matches the host-side integrity pattern).
    void fill_device_pattern(void* device_ptr, std::size_t bytes,
                             std::uint64_t seed, std::uint64_t page_id,
                             DeviceStream* stream);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#else  // CPU-only stub keeps headers includable

class CudaBackend final : public DeviceBackend {
public:
    CudaBackend() = default;
    std::string backend_name() const override { return "cuda"; }
    BackendVendor vendor() const override { return BackendVendor::Nvidia; }
    std::string vendor_display_name() const override { return "NVIDIA (CUDA)"; }
    std::vector<DeviceInfo> enumerate_devices() const override { return {}; }
    DeviceCapabilities probe_capabilities() const override { return {}; }
    void open(int) override {
        throw Error(ErrorCode::Unsupported,
                    "cuda backend not compiled into this build");
    }
    void close() override {}
    bool is_open() const override { return false; }
    DeviceInfo device_info() const override { return {}; }
    DeviceCapabilities capabilities() const override { return {}; }
    uint64_t total_memory() const override { return 0; }
    uint64_t free_memory() const override { return 0; }
    void* allocate(std::size_t) override { return nullptr; }
    void free(void*) override {}
    void* allocate_host_pinned(std::size_t) override { return nullptr; }
    void free_host_pinned(void*) override {}
    void* allocate_unified(std::size_t) override {
        throw Error(ErrorCode::Unsupported, "cuda backend not compiled");
    }
    void free_unified(void*) override {}
    void prefetch_to_device(void*, std::size_t) override {
        throw Error(ErrorCode::Unsupported, "cuda backend not compiled");
    }
    void advise_preferred_location(void*, std::size_t) override {
        throw Error(ErrorCode::Unsupported, "cuda backend not compiled");
    }
    DeviceStream* create_stream() override { return nullptr; }
    void destroy_stream(DeviceStream*) override {}
    DeviceEvent* create_event() override { return nullptr; }
    void destroy_event(DeviceEvent*) override {}
    void async_copy_host_to_device(void*, const void*, std::size_t, DeviceStream*) override {}
    void async_copy_device_to_host(void*, const void*, std::size_t, DeviceStream*) override {}
    void async_copy_device_to_device(void*, const void*, std::size_t, DeviceStream*) override {}
    void sync_stream(DeviceStream*) override {}
    void sync_all() override {}
    void record_event(DeviceEvent*, DeviceStream*) override {}
    void wait_event(DeviceEvent*) override {}
    double event_elapsed_us(DeviceEvent*, DeviceEvent*) override { return 0.0; }
    bool healthy() const override { return false; }
    std::string diagnostics() const override { return "cuda backend not compiled"; }
    std::string last_error() const override { return ""; }
};

#endif  // FLASHTIER_HAVE_CUDA

void register_cuda_backend(BackendRegistry& registry);

}  // namespace flashtier
