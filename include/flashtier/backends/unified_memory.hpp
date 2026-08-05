#pragma once

#include <cstdint>
#include <string>

#include "flashtier/error.hpp"

namespace flashtier {

struct UmResult {
    std::string device_name;
    uint64_t total_vram_bytes = 0;
    uint64_t working_set_bytes = 0;
    double duration_s = 0.0;
    uint64_t ops = 0;
    double ops_per_s = 0.0;
    uint64_t bytes_moved = 0;
    double bandwidth_gb_s = 0.0;
    uint64_t mismatches = 0;
    bool advice_used = false;   // cudaMemAdvise succeeded on this platform
    bool prefetch_used = false; // cudaMemPrefetchAsync succeeded on this platform
};

#if FLASHTIER_HAVE_CUDA

// Bounded Unified Memory oversubscription comparison. Throws
// ErrorCode::Unsupported when the device has no managed memory.
UmResult run_unified_memory_benchmark(int device_id, uint64_t working_set_bytes,
                                      uint64_t page_size,
                                      uint64_t prefetch_distance,
                                      uint64_t iterations, uint64_t seed);

#else

inline UmResult run_unified_memory_benchmark(int, uint64_t, uint64_t,
                                             uint64_t, uint64_t, uint64_t) {
    throw Error(ErrorCode::Unsupported,
                "unified memory benchmark requires a CUDA build");
}

#endif  // FLASHTIER_HAVE_CUDA

}  // namespace flashtier
