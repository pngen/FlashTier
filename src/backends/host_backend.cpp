#include "flashtier/backends/host_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

#include "flashtier/backends/device_backend.hpp"
#include "flashtier/error.hpp"

namespace flashtier {

namespace {

uint64_t usable_limit(uint64_t limit, double margin) noexcept {
    if (margin <= 0.0) return limit;
    if (margin >= 1.0) return 0;
    const long double usable =
        static_cast<long double>(limit) * (1.0L - static_cast<long double>(margin));
    if (usable >= static_cast<long double>(limit)) return limit;
    return static_cast<uint64_t>(usable);
}

}  // namespace

HostBackend::HostBackend() = default;

HostBackend::~HostBackend() {
    std::lock_guard lock(mu_);
    for (const auto& [ptr, allocation] : allocations_) {
        try {
            if (allocation.device != nullptr) {
                allocation.device->free_host_pinned(ptr);
            } else {
                free_fallback(ptr);
            }
        } catch (...) {
            // Destructors cannot report allocator teardown failures. Normal
            // lifecycle code must release allocations explicitly so failures
            // remain observable there.
        }
    }
    allocations_.clear();
    used_ = 0;
}

void HostBackend::set_budget(uint64_t limit_bytes, double reserve_margin) {
    if (!std::isfinite(reserve_margin) || reserve_margin < 0.0 ||
        reserve_margin > 1.0) {
        throw Error(ErrorCode::InvalidArgument,
                    "host backend reserve margin must be in [0, 1]");
    }
    std::lock_guard lock(mu_);
    const uint64_t usable = usable_limit(limit_bytes, reserve_margin);
    if (used_ > usable) {
        throw Error(ErrorCode::State,
                    "host backend budget cannot be reduced below live allocations");
    }
    limit_ = limit_bytes;
    margin_ = reserve_margin;
}

void HostBackend::set_device_backend(DeviceBackend* backend) {
    const bool pinned_available =
        backend != nullptr && backend->capabilities().pinned_host_allocation;
    std::lock_guard lock(mu_);
    if (backend != device_) {
        const bool pinned_live = std::any_of(
            allocations_.begin(), allocations_.end(),
            [](const auto& entry) { return entry.second.device != nullptr; });
        if (pinned_live) {
            throw Error(ErrorCode::State,
                        "cannot change host allocator backend while pinned allocations are live");
        }
    }
    device_ = backend;
    // Capability queries can be expensive (e.g. a live driver probe on the
    // CUDA backend); snapshot the pinned flag once instead of querying it
    // on every allocation.
    pinned_available_ = pinned_available;
}

void* HostBackend::allocate_fallback(uint64_t bytes) {
#if defined(_WIN32)
    void* ptr = _aligned_malloc(static_cast<std::size_t>(bytes), 4096);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 4096, static_cast<std::size_t>(bytes)) != 0) {
        ptr = nullptr;
    }
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
    if (bytes == 0 || bytes > std::numeric_limits<std::size_t>::max()) {
        throw Error(ErrorCode::InvalidArgument,
                    "host backend allocation size is invalid");
    }

    std::lock_guard lock(mu_);
    const uint64_t usable = usable_limit_locked();
    if (bytes > usable || used_ > usable - bytes) {
        return nullptr;  // budget breach; caller decides
    }
    if (!pageable && !pinned_available_) {
        throw Error(ErrorCode::Unsupported,
                    "host backend: pinned allocation requested but the active "
                    "device backend does not provide one; pageable fallback "
                    "must be enabled explicitly");
    }

    DeviceBackend* allocation_device =
        !pageable && pinned_available_ ? device_ : nullptr;
    void* ptr = allocation_device != nullptr
                    ? allocation_device->allocate_host_pinned(
                          static_cast<std::size_t>(bytes))
                    : allocate_fallback(bytes);
    if (ptr == nullptr) {
        throw Error(ErrorCode::Internal,
                    "host allocator returned a null pointer");
    }
    try {
        const bool inserted =
            allocations_.emplace(ptr, Allocation{bytes, allocation_device}).second;
        if (!inserted) {
            throw Error(ErrorCode::State,
                        "host allocator returned a pointer that is already live");
        }
    } catch (...) {
        try {
            if (allocation_device != nullptr) {
                allocation_device->free_host_pinned(ptr);
            } else {
                free_fallback(ptr);
            }
        } catch (...) {
        }
        throw;
    }
    used_ += bytes;
    high_water_ = std::max(high_water_, used_);
    return ptr;
}

void HostBackend::free(void* ptr, uint64_t bytes) {
    if (ptr == nullptr) return;
    std::lock_guard lock(mu_);
    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
        throw Error(ErrorCode::State,
                    "host backend: free of unknown pointer");
    }
    if (bytes != it->second.bytes) {
        throw Error(ErrorCode::InvalidArgument,
                    "host backend: free size does not match allocation",
                    std::to_string(bytes) + " != " +
                        std::to_string(it->second.bytes));
    }
    if (it->second.device != nullptr) {
        it->second.device->free_host_pinned(ptr);
    } else {
        free_fallback(ptr);
    }
    used_ -= it->second.bytes;
    allocations_.erase(it);
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
    const uint64_t usable = usable_limit_locked();
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

Tier HostBackend::allocation_tier(void* ptr) const {
    std::lock_guard lock(mu_);
    const auto it = allocations_.find(ptr);
    if (ptr == nullptr || it == allocations_.end()) {
        throw Error(ErrorCode::State,
                    "host backend: tier query for unknown pointer");
    }
    return it->second.device != nullptr ? Tier::HostPinned : Tier::HostPageable;
}

uint64_t HostBackend::usable_limit_locked() const noexcept {
    return usable_limit(limit_, margin_);
}

}  // namespace flashtier
