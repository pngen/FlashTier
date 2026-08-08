#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "flashtier/error.hpp"

namespace flashtier {

// Vendor-neutral accelerator contract. Every accelerator backend (CUDA,
// HIP/ROCm, Level Zero, Vulkan, Metal, CPU emulation, test mocks)
// implements this interface. The governed runtime, planner, page table,
// policies, telemetry, integrity system, NVMe backend, and host-memory
// backend never call vendor APIs directly; all accelerator-specific
// behavior sits behind this contract.
//
// Typed opaque handles (DeviceStream, DeviceEvent) isolate vendor handle
// types (cudaStream_t, hipStream_t, ze_command_queue_handle_t, VkQueue,
// MTLCommandQueue, ...) from the core runtime.

enum class BackendVendor : int {
    Unknown = 0,
    Nvidia = 1,
    Amd = 2,
    Intel = 3,
    Apple = 4,
    Portable = 5,  // CPU emulation / portable host path
    Virtual = 6,   // test mocks and simulators
};

const char* backend_vendor_name(BackendVendor vendor) noexcept;

// One accelerator discovered through a backend. `discrete` and
// `memory_shared` come from device discovery, never from product names.
struct DeviceInfo {
    std::string id;           // backend-unique identity, e.g. "cuda:0", "level_zero:0"
    std::string name;         // product name reported by the driver
    std::string architecture; // compute capability / arch id (e.g. "12.0", "gfx1100")
    uint64_t total_memory = 0;
    uint64_t free_memory = 0;  // 0 = unknown
    bool discrete = true;      // dedicated accelerator with device-local memory
    bool memory_shared = false;  // unified/shared-memory architecture (integrated GPUs)
    int index = 0;             // vendor device index within the backend
};

// Structured capability model. Every field is discovered at runtime by the
// backend; nothing is guessed from marketing names.
struct DeviceCapabilities {
    bool backend_available = false;
    bool device_available = false;
    bool explicit_allocation = false;
    bool async_host_to_device = false;
    bool async_device_to_host = false;
    bool device_to_device = false;
    bool pinned_host_allocation = false;
    bool unified_memory = false;
    bool concurrent_managed_access = false;
    bool memory_prefetch = false;
    bool memory_advice = false;
    bool direct_storage = false;
    bool peer_to_peer = false;
    bool multi_device = false;
    bool hardware_page_fault = false;
    bool host_coherent_memory = false;
    bool event_timing = false;
    uint64_t max_allocation_size = 0;
    uint64_t alignment_bytes = 0;
    uint64_t transfer_granularity = 0;
    uint32_t queue_count = 0;
    std::string note;  // backend-specific capability notes
};

// Opaque execution queue / stream. Vendor handles live inside.
class DeviceStream {
public:
    virtual ~DeviceStream() = default;
};

// Opaque event / fence. Supports synchronization and, when
// DeviceCapabilities::event_timing is set, elapsed-time measurement.
class DeviceEvent {
public:
    virtual ~DeviceEvent() = default;
};

class DeviceBackend {
public:
    virtual ~DeviceBackend() = default;

    // ---- identity ----------------------------------------------------------
    virtual std::string backend_name() const = 0;          // "cuda", "hip", "level_zero", ...
    virtual BackendVendor vendor() const = 0;
    virtual std::string vendor_display_name() const = 0;

    // ---- discovery (safe without open) -------------------------------------
    virtual std::vector<DeviceInfo> enumerate_devices() const = 0;

    // Capability probe without opening a specific device. Uses the default
    // device when one exists; valid only when devices were enumerated.
    virtual DeviceCapabilities probe_capabilities() const = 0;

    // ---- lifecycle ----------------------------------------------------------
    // Throws ErrorCode::Config when the index is invalid, ErrorCode::Cuda /
    // backend-specific typed errors on init failure. Partial initialization
    // failures must clean up after themselves.
    virtual void open(int device_index) = 0;
    virtual void close() = 0;  // idempotent; safe from the destructor
    virtual bool is_open() const = 0;

    // ---- selected device ----------------------------------------------------
    virtual DeviceInfo device_info() const = 0;
    virtual DeviceCapabilities capabilities() const = 0;
    virtual uint64_t total_memory() const = 0;
    virtual uint64_t free_memory() const = 0;

    // ---- device memory ------------------------------------------------------
    virtual void* allocate(std::size_t bytes) = 0;  // throws InvalidArgument on 0 or > max_allocation_size
    virtual void free(void* ptr) = 0;

    // ---- host memory ----------------------------------------------------------
    // Pinned/page-locked host allocation. Throws ErrorCode::Unsupported when
    // the backend does not advertise pinned_host_allocation.
    virtual void* allocate_host_pinned(std::size_t bytes) = 0;
    virtual void free_host_pinned(void* ptr) = 0;

    // ---- unified / shared memory (optional) ---------------------------------
    virtual void* allocate_unified(std::size_t bytes) = 0;  // Unsupported when not advertised
    virtual void free_unified(void* ptr) = 0;
    virtual void prefetch_to_device(void* ptr, std::size_t bytes) = 0;  // Unsupported when not advertised
    virtual void advise_preferred_location(void* ptr, std::size_t bytes) = 0;

    // ---- queues / streams ----------------------------------------------------
    virtual DeviceStream* create_stream() = 0;
    virtual void destroy_stream(DeviceStream* stream) = 0;

    // ---- events ---------------------------------------------------------------
    virtual DeviceEvent* create_event() = 0;
    virtual void destroy_event(DeviceEvent* event) = 0;

    // ---- transfers (asynchronous; caller synchronizes) -----------------------
    virtual void async_copy_host_to_device(void* dst_device, const void* src_host,
                                           std::size_t bytes, DeviceStream* stream) = 0;
    virtual void async_copy_device_to_host(void* dst_host, const void* src_device,
                                           std::size_t bytes, DeviceStream* stream) = 0;
    // Optional device-to-device. Unsupported when not advertised.
    virtual void async_copy_device_to_device(void* dst_device, const void* src_device,
                                             std::size_t bytes, DeviceStream* stream) = 0;

    // ---- synchronization -------------------------------------------------------
    virtual void sync_stream(DeviceStream* stream) = 0;
    virtual void sync_all() = 0;

    // ---- events ---------------------------------------------------------------
    virtual void record_event(DeviceEvent* event, DeviceStream* stream) = 0;
    virtual void wait_event(DeviceEvent* event) = 0;
    // Requires event_timing capability.
    virtual double event_elapsed_us(DeviceEvent* start, DeviceEvent* end) = 0;

    // ---- health / diagnostics ---------------------------------------------------
    virtual bool healthy() const = 0;
    virtual std::string diagnostics() const = 0;   // driver/runtime versions etc.
    virtual std::string last_error() const = 0;

    // ---- convenience synchronous copies (defaults compose async + sync) ---------
    virtual void copy_host_to_device_sync(void* dst_device, const void* src_host,
                                          std::size_t bytes) {
        DeviceStream* stream = create_stream();
        try {
            async_copy_host_to_device(dst_device, src_host, bytes, stream);
            sync_stream(stream);
        } catch (...) {
            // Preserve the transfer/synchronization failure while still
            // releasing the helper-owned stream. A cleanup failure must not
            // replace the causal exception.
            try {
                destroy_stream(stream);
            } catch (...) {
            }
            throw;
        }
        destroy_stream(stream);
    }

    virtual void copy_device_to_host_sync(void* dst_host, const void* src_device,
                                          std::size_t bytes) {
        DeviceStream* stream = create_stream();
        try {
            async_copy_device_to_host(dst_host, src_device, bytes, stream);
            sync_stream(stream);
        } catch (...) {
            try {
                destroy_stream(stream);
            } catch (...) {
            }
            throw;
        }
        destroy_stream(stream);
    }
};

}  // namespace flashtier
