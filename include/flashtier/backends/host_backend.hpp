#pragma once

#include <cstdint>
#include <mutex>
#include <unordered_map>

#include "flashtier/tier.hpp"

namespace flashtier {

class DeviceBackend;

// Pinned (or explicitly pageable) host memory backend.
//
// Pinned allocation comes from the active device backend's
// allocate_host_pinned (CUDA, HIP, Level Zero, Vulkan, ...) when the
// backend advertises it; otherwise an aligned heap fallback is used. A
// pageable fallback exists but is only used when explicitly enabled via
// Config::use_pageable_fallback — it never silently substitutes for pinned
// memory in performance claims. Telemetry records which kind of host
// memory was used. This class never calls vendor APIs directly.
class HostBackend {
public:
    HostBackend();
    ~HostBackend();

    HostBackend(const HostBackend&) = delete;
    HostBackend& operator=(const HostBackend&) = delete;

    // Configure the budget.
    void set_budget(uint64_t limit_bytes, double reserve_margin);

    // The device backend whose pinned allocator is used. Null disables
    // pinned allocation (aligned heap fallback only).
    void set_device_backend(DeviceBackend* backend);

    // Allocate `bytes` (already aligned to page size by the caller).
    // Returns nullptr on budget breach; throws Error on allocator failure.
    void* allocate(uint64_t bytes, bool pageable);

    void free(void* ptr, uint64_t bytes);

    uint64_t used() const noexcept;
    uint64_t limit() const noexcept;
    uint64_t headroom() const noexcept;
    uint64_t high_water() const noexcept;
    bool uses_pinned() const noexcept;
    Tier allocation_tier(void* ptr) const;

private:
    struct Allocation {
        uint64_t bytes = 0;
        DeviceBackend* device = nullptr;  // non-null when vendor-pinned
    };

    void* allocate_fallback(uint64_t bytes);
    void free_fallback(void* ptr);
    uint64_t usable_limit_locked() const noexcept;

    mutable std::mutex mu_;
    DeviceBackend* device_ = nullptr;
    bool pinned_available_ = false;  // snapshot taken at set_device_backend
    uint64_t limit_ = 0;
    uint64_t used_ = 0;
    uint64_t high_water_ = 0;
    double margin_ = 0.0;
    std::unordered_map<void*, Allocation> allocations_;
};

}  // namespace flashtier
