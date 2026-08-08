#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "flashtier/error.hpp"

namespace flashtier {

// Completion handle for asynchronous store operations. Thread-safe.
class AsyncOp {
public:
    enum class Status : int { Pending = 0, Complete = 1, Cancelled = 2, Failed = 3 };

    virtual ~AsyncOp() = default;

    // Block until the operation completes, is cancelled, or fails. Throws
    // the operation's error on failure.
    virtual void wait() = 0;

    virtual Status status() const noexcept = 0;
    virtual uint64_t bytes_done() const noexcept = 0;
    virtual bool is_complete() const noexcept = 0;
};

using AsyncOpPtr = std::shared_ptr<AsyncOp>;

// Abstract backing-store interface. v0.1 ships the file-backed NvmeBackend;
// future DirectStorage / GPUDirect Storage / HBF backends implement this
// interface and may bypass the host staging step.
class StorageBackend {
public:
    struct Info {
        uint64_t capacity_bytes = 0;  // usable payload bytes
        uint64_t page_size = 0;
        uint64_t extent_count = 0;
        std::string path;
    };

    virtual ~StorageBackend() = default;

    // Open or create the store at `path` with the given capacity and page
    // size. Validates existing store metadata; rejects stale or
    // incompatible stores with ErrorCode::StoreCorrupt.
    virtual Info open(const std::string& path, uint64_t capacity_bytes,
                      uint64_t page_size) = 0;

    // Allocate one extent of `page_size` bytes. Returns false when full.
    virtual bool allocate_extent(uint64_t& offset_out) = 0;

    virtual void free_extent(uint64_t offset) = 0;

    // Allocate/free a contiguous run of extents atomically. The default
    // implementation preserves compatibility for backends that only support
    // single-extent operations; multi-extent requests are rejected. A
    // successful allocation owns every extent in
    // [offset_out, offset_out + extent_count * page_size).
    virtual bool allocate_extents(uint64_t extent_count, uint64_t& offset_out);
    virtual void free_extents(uint64_t offset, uint64_t extent_count);

    virtual uint64_t extents_used() const noexcept = 0;

    // Async reads/writes. The operation runs in the backend and completes
    // via the returned handle.
    virtual AsyncOpPtr read_async(uint64_t offset, void* dst, std::size_t bytes) = 0;
    virtual AsyncOpPtr write_async(uint64_t offset, const void* src, std::size_t bytes) = 0;

    // Flush metadata and file buffers to stable storage.
    virtual void flush() = 0;

    // Cancel in-flight operations where supported. Completes promptly.
    virtual void cancel_all() = 0;

    // Close the store. Idempotent. No-op after close.
    virtual void close() = 0;
};

// Verify store header integrity for an existing file without opening a
// backend (used by tests and `flashtier verify`).
bool store_header_valid(const std::string& path, uint64_t page_size, uint64_t capacity_bytes);

}  // namespace flashtier
