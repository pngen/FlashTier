#include "flashtier/backends/host_backend.hpp"

#include <algorithm>
#include <cstdlib>

#include "flashtier/error.hpp"

#if FLASHTIER_HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace flashtier {

HostBackend::HostBackend()
#if FLASHTIER_HAVE_CUDA
    : cuda_pinned_(true)
#endif
{
}

HostBackend::~HostBackend() = default;

void HostBackend::set_budget(uint64_t limit_bytes, double reserve_margin) {
    std::lock_guard lock(mu_);
    limit_ = limit_bytes;
    margin_ = reserve_margin;
}

void* HostBackend::allocate(uint64_t bytes, bool pageable) {
    (void)pageable;  // used only in CUDA builds
    {
        std::lock_guard lock(mu_);
        const uint64_t usable = limit_ - static_cast<uint64_t>(static_cast<double>(limit_) * margin_);
        if (bytes > usable || used_ > usable - bytes) {
            return nullptr;  // budget breach; caller decides
        }
    }

    void* ptr = nullptr;
#if FLASHTIER_HAVE_CUDA
    if (cuda_pinned_ && !pageable) {
        cudaError_t e = cudaMallocHost(&ptr, bytes);
        if (e != cudaSuccess) {
            throw Error(ErrorCode::Cuda,
                        "cudaMallocHost failed for pinned host memory",
                        cudaGetErrorString(e));
        }
    } else
#endif
    {
#if defined(_WIN32)
        ptr = _aligned_malloc(bytes, 4096);
#else
        ptr = aligned_alloc(4096, bytes);
#endif
        if (ptr == nullptr) {
            throw Error(ErrorCode::Internal, "aligned host allocation failed");
        }
    }

    std::lock_guard lock(mu_);
    used_ += bytes;
    high_water_ = std::max(high_water_, used_);
    return ptr;
}

void HostBackend::free(void* ptr, uint64_t bytes) {
    if (ptr == nullptr) return;
#if FLASHTIER_HAVE_CUDA
    if (cuda_pinned_) {
        cudaError_t e = cudaFreeHost(ptr);
        if (e != cudaSuccess) {
            throw Error(ErrorCode::Cuda, "cudaFreeHost failed",
                        cudaGetErrorString(e));
        }
    } else
#endif
    {
#if defined(_WIN32)
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
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

}  // namespace flashtier
