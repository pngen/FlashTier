// Bounded Unified Memory comparison benchmark. Compiled with nvcc only
// when FLASHTIER_ENABLE_CUDA=ON.
//
// Implements the driver-managed side of the Unified Memory comparison
// (cudaMallocManaged plus optional cudaMemPrefetchAsync/cudaMemAdvise).
// It shares working-set and page geometry with the explicit benchmark, but
// deliberately reports its own one-uint64-per-page touch volume: the explicit
// path writes full page payloads, so this is not a byte-for-byte work claim.
//
// Some WDDM drivers report managedMemory but reject prefetch/advice at
// runtime; hint failures are recorded in UmResult, never silently
// swallowed.

#include "flashtier/backends/unified_memory.hpp"

#include <chrono>
#include <cstddef>
#include <limits>
#include <cuda_runtime.h>

#include "flashtier/error.hpp"
#include "flashtier/integrity.hpp"

namespace flashtier {

namespace {

void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw Error(ErrorCode::Cuda,
                    std::string(what) + " failed: " + cudaGetErrorString(e));
    }
}

class DeviceRestore final {
public:
    explicit DeviceRestore(int device_id) {
        check_cuda(cudaGetDevice(&previous_), "cudaGetDevice");
        if (previous_ != device_id) {
            check_cuda(cudaSetDevice(device_id), "cudaSetDevice");
            restore_ = true;
        }
    }

    ~DeviceRestore() {
        if (restore_) {
            (void)cudaSetDevice(previous_);
        }
    }

    DeviceRestore(const DeviceRestore&) = delete;
    DeviceRestore& operator=(const DeviceRestore&) = delete;

private:
    int previous_ = 0;
    bool restore_ = false;
};

}  // namespace

UmResult run_unified_memory_benchmark(int device_id, uint64_t working_set_bytes,
                                      uint64_t page_size, uint64_t prefetch_distance,
                                      uint64_t iterations, uint64_t seed) {
    if (working_set_bytes == 0 || page_size == 0 ||
        working_set_bytes % page_size != 0) {
        throw Error(ErrorCode::InvalidArgument,
                    "unified memory benchmark requires nonzero aligned sizes");
    }
    if (page_size < sizeof(uint64_t) || page_size % sizeof(uint64_t) != 0) {
        throw Error(ErrorCode::InvalidArgument,
                    "unified memory benchmark page size must be a multiple of 8 bytes");
    }
    if (iterations == 0) {
        throw Error(ErrorCode::InvalidArgument,
                    "unified memory benchmark requires at least one iteration");
    }
    if (working_set_bytes > std::numeric_limits<std::size_t>::max()) {
        throw Error(ErrorCode::InvalidArgument,
                    "unified memory benchmark working set exceeds addressable memory");
    }
    const uint64_t page_count = working_set_bytes / page_size;
    if (iterations > std::numeric_limits<uint64_t>::max() / page_count) {
        throw Error(ErrorCode::InvalidArgument,
                    "unified memory benchmark operation count overflows");
    }
    const uint64_t ops = page_count * iterations;
    if (ops > std::numeric_limits<uint64_t>::max() / sizeof(uint64_t)) {
        throw Error(ErrorCode::InvalidArgument,
                    "unified memory benchmark byte count overflows");
    }

    DeviceRestore restore_device(device_id);
    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, device_id), "cudaGetDeviceProperties");
    if (prop.managedMemory == 0) {
        throw Error(ErrorCode::Unsupported,
                    "device does not support unified memory");
    }

    UmResult out;
    out.device_name = prop.name;
    out.total_vram_bytes = static_cast<uint64_t>(prop.totalGlobalMem);
    out.working_set_bytes = working_set_bytes;

    void* ptr = nullptr;
    check_cuda(cudaMallocManaged(&ptr, working_set_bytes), "cudaMallocManaged");
    try {
        // Best-effort driver hints; failures are recorded, not fatal. Some
        // WDDM drivers reject these APIs even when managedMemory is set.
        out.advice_used =
            cudaMemAdvise(ptr, working_set_bytes,
                          cudaMemAdviseSetPreferredLocation, device_id) == cudaSuccess;
        out.prefetch_used =
            cudaMemPrefetchAsync(ptr, working_set_bytes, device_id) == cudaSuccess;
        cudaGetLastError();  // clear stale error state from failed hints

        // A successful prefetch is asynchronous. Complete it before the CPU
        // starts touching managed memory; host access while a device operation
        // is active is invalid on devices without concurrent managed access.
        if (out.prefetch_used) {
            check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(prefetch)");
        }

        const auto t0 = std::chrono::steady_clock::now();

        for (uint64_t it = 0; it < iterations; ++it) {
            for (uint64_t p = 0; p < page_count; ++p) {
                // Deterministic touch: write one 8-byte pattern per page.
                const uint64_t v = seed ^ (p * 0x9E3779B97F4A7C15ull) ^ it;
                auto* dst = static_cast<uint64_t*>(ptr) + (p * (page_size / 8));
                *dst = v;

                // Prefetch a few pages ahead (best effort). Avoid overflow in
                // the caller-controlled distance before forming the index.
                if (out.prefetch_used && prefetch_distance < page_count - p) {
                    const uint64_t ahead = p + prefetch_distance;
                    const cudaError_t hint = cudaMemPrefetchAsync(
                        static_cast<char*>(ptr) + ahead * page_size,
                        page_size, device_id);
                    if (hint != cudaSuccess) {
                        out.prefetch_used = false;
                        cudaGetLastError();
                    } else if (prop.concurrentManagedAccess == 0) {
                        // On non-concurrent devices, do not permit the next
                        // host touch to overlap a managed-memory device op.
                        check_cuda(cudaDeviceSynchronize(),
                                   "cudaDeviceSynchronize(prefetch-ahead)");
                    }
                }
            }
            // Drain once per iteration so timing reflects driver-managed moves.
            check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        }

        const auto t1 = std::chrono::steady_clock::now();
        out.duration_s = std::chrono::duration<double>(t1 - t0).count();
        out.ops = ops;
        out.ops_per_s = out.duration_s > 0.0
                            ? static_cast<double>(ops) / out.duration_s
                            : 0.0;
        // Only one uint64_t is explicitly accessed per operation. Managed
        // migration granularity is driver/platform dependent and is not
        // falsely reported as measured transfer volume.
        out.bytes_moved = ops * sizeof(uint64_t);
        out.bandwidth_gb_s = out.duration_s > 0.0
                                 ? static_cast<double>(out.bytes_moved) / 1e9 /
                                       out.duration_s
                                 : 0.0;

        // Only the last iteration's value can remain in each location. The
        // previous code compared that final value against every historical
        // iteration and therefore manufactured one mismatch per page whenever
        // iterations was greater than one.
        const uint64_t final_iteration = iterations - 1;
        for (uint64_t p = 0; p < page_count; ++p) {
            const uint64_t expected =
                seed ^ (p * 0x9E3779B97F4A7C15ull) ^ final_iteration;
            const auto* src =
                static_cast<const uint64_t*>(ptr) + (p * (page_size / 8));
            if (*src != expected) ++out.mismatches;
        }

        check_cuda(cudaFree(ptr), "cudaFree(managed)");
        ptr = nullptr;
    } catch (...) {
        if (ptr != nullptr) {
            (void)cudaFree(ptr);
        }
        throw;
    }
    return out;
}

}  // namespace flashtier
