#include "flashtier/backends/host_backend.hpp"

#include <algorithm>
#include <cstdlib>

#include "flashtier/backends/device_backend.hpp"
#include "flashtier/error.hpp"

namespace flashtier {

HostBackend::HostBackend() = default;

HostBackend::~HostBackend() = default;

void HostBackend::set_budget(uint64_t limit_bytes, double reserve_margin) {
    std::lock_guard lock(mu_);
    limit_ = limit_bytes;
    margin_ = reserve_margin;
}

void HostBackend::set_device_backend(DeviceBackend* backend) {
    std::lock_guard lock(mu_);
    device_ = backend;
    // Capability queries can be expensive (e.g. a live driver probe on the
    // CUDA backend); snapshot the pinned flag once instead of querying it
    // on every allocation.
    pinned_available_ = device_ != nullptr && device_->capabilities().pinned_host_allocation;
}

void* HostBackend::allocate_fallback(uint64_t bytes) {
#if defined(_WIN32)
    void* ptr = _aligned_malloc(bytes, 4096);
#else
    void* ptr = aligned_alloc(4096, bytes);
#endif
    if (ptr == nullptr) {
        throw Error(ErrorCode::Internal, "aligned host allocation failed");
    }
    return ptr;
}

void HostBackend::free_fallback(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

void* HostBackend::allocate(uint64_t bytes, bool pageable) {
    bool use_device_pinned = false;
    {
        std::lock_guard lock(mu_);
        const uint64_t usable = limit_ - static_cast<uint64_t>(static_cast<double>(limit_) * margin_);
        if (bytes > usable || used_ > usable - bytes) {
            return nullptr;  // budget breach; caller decides
        }
        use_device_pinned = !pageable && pinned_available_;
    }

    void* ptr = nullptr;
    if (use_device_pinned) {
        ptr = device_->allocate_host_pinned(bytes);
    } else {
        ptr = allocate_fallback(bytes);
    }

    std::lock_guard lock(mu_);
    if (use_device_pinned) {
        pinned_ptrs_.insert(ptr);
    }
    used_ += bytes;
    high_water_ = std::max(high_water_, used_);
    return ptr;
}

void HostBackend::free(void* ptr, uint64_t bytes) {
    if (ptr == nullptr) return;
    bool was_pinned = false;
    {
        std::lock_guard lock(mu_);
        was_pinned = pinned_ptrs_.erase(ptr) != 0;
    }
    if (was_pinned) {
        device_->free_host_pinned(ptr);
    } else {
        free_fallback(ptr);
    }
    std::lock_guard lock(mu_);
    used_ = (bytes >= used_) ? 0 : used_ - bytes;
}

uint64_t HostBackend::used() const noexcept {
    std::lock_guard lock(mu_);
    return used_;
}

uint64_t HostBackend::limit() const noexcept {
    std::lock_guard lock(mu_);
    return limit_;
}

uint64_t HostBackend::headroom() const noexcept {
    std::lock_guard lock(mu_);
    const uint64_t usable = limit_ - static_cast<uint64_t>(static_cast<double>(limit_) * margin_);
    return (used_ >= usable) ? 0 : usable - used_;
}

uint64_t HostBackend::high_water() const noexcept {
    std::lock_guard lock(mu_);
    return high_water_;
}

bool HostBackend::uses_pinned() const noexcept {
    std::lock_guard lock(mu_);
    return pinned_available_;
}

}  // namespace flashtier
