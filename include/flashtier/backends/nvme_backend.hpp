#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "flashtier/backends/storage_backend.hpp"

namespace flashtier {

// File-backed NVMe page store.
//
// Windows: overlapped I/O driven by an I/O completion port with a bounded
// worker pool; offsets and sizes are page-aligned (the OS buffered path
// accepts any buffer alignment, so callers may use pinned or heap buffers).
// Linux: threaded pread/pwrite with the same AsyncOp interface.
//
// The store works on any filesystem path; a fast local NVMe volume is
// recommended but not verified or required. Store metadata (magic, version,
// page size, capacity, generation) is written and validated at open time so
// stale or incompatible stores are rejected.
class NvmeBackend final : public StorageBackend {
public:
    NvmeBackend();
    ~NvmeBackend() override;

    NvmeBackend(const NvmeBackend&) = delete;
    NvmeBackend& operator=(const NvmeBackend&) = delete;

    // StorageBackend interface.
    Info open(const std::string& path, uint64_t capacity_bytes,
              uint64_t page_size) override;
    bool allocate_extent(uint64_t& offset_out) override;
    void free_extent(uint64_t offset) override;
    uint64_t extents_used() const noexcept override;
    AsyncOpPtr read_async(uint64_t offset, void* dst, std::size_t bytes) override;
    AsyncOpPtr write_async(uint64_t offset, const void* src, std::size_t bytes) override;
    void flush() override;
    void cancel_all() override;
    void close() override;

    // True when the platform backend was initialized (open succeeded).
    bool is_open() const noexcept { return opened_; }

    // Wipe the backing file (used by tests for corrupted-store scenarios).
    static void destroy_file(const std::string& path);

private:
    // Implementation is public: platform worker helpers need it. The class
    // surface below remains the only supported API.
public:
    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
    bool opened_ = false;
};

}  // namespace flashtier
