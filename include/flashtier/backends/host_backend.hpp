#pragma once

#include <cstdint>
#include <mutex>

#include "flashtier/tier.hpp"

namespace flashtier {

// Pinned (or explicitly pageable) host memory backend.
//
// With CUDA available, allocations use cudaMallocHost (pinned). A pageable
// fallback (aligned heap allocation) exists but is only used when
// explicitly enabled via Config::use_pageable_fallback — it never silently
// substitutes for pinned memory in performance claims. Telemetry records
// which kind of host memory was used.
class HostBackend {
public:
    HostBackend();
    ~HostBackend();

    HostBackend(const HostBackend&) = delete;
    HostBackend& operator=(const HostBackend&) = delete;

    // Configure the budget. Errors on negative reserve.
    void set_budget(uint64_t limit_bytes, double reserve_margin);

    // Allocate `bytes` (already aligned to page size by the caller).
    // Returns nullptr on budget breach; throws Error on allocator failure.
    void* allocate(uint64_t bytes, bool pageable);

    void free(void* ptr, uint64_t bytes);

    uint64_t used() const noexcept;
    uint64_t limit() const noexcept;
    uint64_t headroom() const noexcept;
    uint64_t high_water() const noexcept;
    bool uses_pinned() const noexcept { return cuda_pinned_; }

private:
    mutable std::mutex mu_;
    uint64_t limit_ = 0;
    uint64_t used_ = 0;
    uint64_t high_water_ = 0;
    double margin_ = 0.0;
    bool cuda_pinned_ = false;
};

}  // namespace flashtier
