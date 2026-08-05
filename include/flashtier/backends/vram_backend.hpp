#pragma once

#include <cstdint>
#include <string>

namespace flashtier {

// CUDA VRAM backend: explicit device allocations, streams, async copies,
// events, bandwidth measurement, and GPU fill/verify kernels. Every CUDA
// call is checked and mapped to ErrorCode::Cuda. Never silently ignores an
// allocation or transfer failure.
//
// This backend is compiled only when FLASHTIER_HAVE_CUDA is defined (the
// header remains includable in CPU-only builds via the inline stubs).
#if FLASHTIER_HAVE_CUDA

#include <cuda_runtime.h>

class VramBackend {
public:
    struct CopyResult {
        double duration_us = 0.0;
        double bandwidth_gb_s = 0.0;
    };

    explicit VramBackend(int device_id);
    ~VramBackend();

    VramBackend(const VramBackend&) = delete;
    VramBackend& operator=(const VramBackend&) = delete;

    // Select the device, query properties, create streams.
    void open();
    void close();

    const cudaDeviceProp& prop() const;

    uint64_t total_mem_bytes() const;
    uint64_t free_mem_bytes() const;  // queries the driver each call

    void* alloc(std::size_t bytes);   // throws ErrorCode::Cuda on failure
    void free(void* ptr);

    // Asynchronous copies on a runtime-owned stream; call sync() before
    // consuming results.
    void async_h2d(void* dst_device, const void* src_host, std::size_t bytes, int stream = 0);
    void async_d2h(void* dst_host, const void* src_device, std::size_t bytes, int stream = 0);
    void sync(int stream = 0);
    void sync_all();

    // Record an event on a stream and measure elapsed time between events.
    void record_start(int stream = 0);
    void record_end(int stream = 0);
    double elapsed_us() const;

    // Self-timed bandwidth measurements (iterations, best-of).
    CopyResult measure_h2d(std::size_t bytes, int iterations = 8);
    CopyResult measure_d2h(std::size_t bytes, int iterations = 8);

    // GPU kernels (fill/verify used by benchmarks and integrity checks).
    void fill_device_pattern(void* device_ptr, std::size_t bytes,
                             std::uint64_t seed, std::uint64_t page_id,
                             int stream = 0);

    // Unified Memory support (comparison benchmark only; the governed
    // runtime does not rely on it).
    bool unified_memory_available() const;
    void* alloc_managed(std::size_t bytes);
    void free_managed(void* ptr);
    void prefetch_managed(void* ptr, std::size_t bytes, int device);
    void advise_preferred_location(void* ptr, std::size_t bytes, int device);

    int device_id() const noexcept { return device_id_; }

private:
    int device_id_;
    cudaDeviceProp prop_{};
    cudaStream_t stream_[2] = {nullptr, nullptr};
    cudaEvent_t start_event_ = nullptr;
    cudaEvent_t end_event_ = nullptr;
    bool opened_ = false;
};

#else  // CPU-only stub

class VramBackend {
public:
    struct CopyResult {
        double duration_us = 0.0;
        double bandwidth_gb_s = 0.0;
    };
    explicit VramBackend(int) {}
    void open() {}
    void close() {}
    uint64_t total_mem_bytes() const { return 0; }
    uint64_t free_mem_bytes() const { return 0; }
    void* alloc(std::size_t) { return nullptr; }
    void free(void*) {}
    void async_h2d(void*, const void*, std::size_t, int = 0) {}
    void async_d2h(void*, const void*, std::size_t, int = 0) {}
    void sync(int = 0) {}
    void sync_all() {}
    bool unified_memory_available() const { return false; }
};

#endif  // FLASHTIER_HAVE_CUDA

}  // namespace flashtier
