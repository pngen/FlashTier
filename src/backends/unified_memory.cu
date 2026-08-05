// Bounded Unified Memory comparison benchmark. Compiled with nvcc only
// when FLASHTIER_ENABLE_CUDA=ON.
//
// Compares the driver-managed oversubscription path (cudaMallocManaged +
// cudaMemPrefetchAsync) against FlashTier's explicit strategy. Uses the
// same deterministic access pattern and working-set geometry as the
// explicit oversubscription benchmark so the comparison is apples-to-apples
// on the same machine.

#include "unified_memory.hpp"

#include <chrono>
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

}  // namespace

UmResult run_unified_memory_benchmark(int device_id, uint64_t working_set_bytes,
                                      uint64_t page_size, uint64_t prefetch_distance,
                                      uint64_t iterations, uint64_t seed) {
    if (working_set_bytes == 0 || page_size == 0 ||
        working_set_bytes % page_size != 0) {
        throw Error(ErrorCode::InvalidArgument,
                    "unified memory benchmark requires nonzero aligned sizes");
    }

    check_cuda(cudaSetDevice(device_id), "cudaSetDevice");
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
    // Ask the driver to keep the buffer near the device.
    check_cuda(cudaMemAdvise(ptr, working_set_bytes,
                             cudaMemAdviseSetPreferredLocation, device_id),
               "cudaMemAdvise");
    check_cuda(cudaMemPrefetchAsync(ptr, working_set_bytes, device_id),
               "cudaMemPrefetchAsync");

    const uint64_t page_count = working_set_bytes / page_size;
    const uint64_t ops = page_count * iterations;

    const auto t0 = std::chrono::steady_clock::now();

    for (uint64_t it = 0; it < iterations; ++it) {
        for (uint64_t p = 0; p < page_count; ++p) {
            // Deterministic touch: write one 8-byte pattern per page.
            const uint64_t block = p;
            const uint64_t v = seed ^ (block * 0x9E3779B97F4A7C15ull) ^ it;
            auto* dst = static_cast<uint64_t*>(ptr) + (p * (page_size / 8));
            *dst = v;

            // Prefetch a few pages ahead.
            const uint64_t ahead = p + prefetch_distance;
            if (ahead < page_count) {
                cudaMemPrefetchAsync(static_cast<char*>(ptr) + ahead * page_size,
                                     page_size, device_id);
            }
        }
        // Drain once per iteration so timing reflects driver-managed moves.
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }

    // Verify the deterministic pattern survived.
    uint64_t mismatches = 0;
    for (uint64_t it = 0; it < iterations; ++it) {
        for (uint64_t p = 0; p < page_count; ++p) {
            const uint64_t expected = seed ^ (p * 0x9E3779B97F4A7C15ull) ^ it;
            const auto* src = static_cast<const uint64_t*>(ptr) + (p * (page_size / 8));
            if (*src != expected) ++mismatches;
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    out.duration_s =
        std::chrono::duration<double>(t1 - t0).count();
    out.ops = ops;
    out.ops_per_s = static_cast<double>(ops) / out.duration_s;
    out.bytes_moved = working_set_bytes * iterations;  // one write per page per iteration
    out.bandwidth_gb_s = static_cast<double>(out.bytes_moved) / 1e9 / out.duration_s;
    out.mismatches = mismatches;

    check_cuda(cudaFree(ptr), "cudaFree(managed)");
    return out;
}

}  // namespace flashtier
