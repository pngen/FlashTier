#include "flashtier/backends/nvme_backend.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#include "flashtier/error.hpp"
#include "store_common.hpp"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <sys/stat.h>
#include <sys/file.h>
#endif

namespace flashtier {

namespace {

void validate_store_geometry(const std::string& path, uint64_t capacity_bytes,
                             uint64_t page_size, uint64_t max_file_size,
                             uint64_t max_io_size) {
    if (path.empty()) {
        throw Error(ErrorCode::InvalidArgument, "store path must not be empty");
    }
    if (path.find('\0') != std::string::npos) {
        throw Error(ErrorCode::InvalidArgument,
                    "store path must not contain an embedded NUL byte");
    }
    if (capacity_bytes == 0 || page_size < detail::kHeaderBlockSize ||
        capacity_bytes % page_size != 0) {
        throw Error(ErrorCode::InvalidArgument,
                    "store capacity must be a nonzero multiple of a page size "
                    "of at least 4096 bytes");
    }
    if (page_size > max_io_size) {
        throw Error(ErrorCode::InvalidArgument,
                    "store page size exceeds the platform I/O limit",
                    std::to_string(page_size));
    }
    if (capacity_bytes > std::numeric_limits<uint64_t>::max() - page_size ||
        capacity_bytes + page_size > max_file_size) {
        throw Error(ErrorCode::InvalidArgument,
                    "store geometry exceeds the platform file-size limit");
    }
    const uint64_t extent_count = capacity_bytes / page_size;
    if (extent_count == 0 || extent_count > std::numeric_limits<std::size_t>::max()) {
        throw Error(ErrorCode::InvalidArgument,
                    "store extent count exceeds the platform metadata limit");
    }
}

void validate_io_request(uint64_t page_size, uint64_t file_size, uint64_t offset,
                         const void* buffer, std::size_t bytes,
                         uint64_t max_io_size) {
    if (buffer == nullptr) {
        throw Error(ErrorCode::InvalidArgument, "store I/O buffer must not be null");
    }
    if (bytes == 0 || static_cast<uint64_t>(bytes) > max_io_size) {
        throw Error(ErrorCode::InvalidArgument,
                    "store I/O size is zero or exceeds the platform limit",
                    std::to_string(bytes));
    }
    const uint64_t io_bytes = static_cast<uint64_t>(bytes);
    if (offset < page_size || offset % page_size != 0 ||
        io_bytes % page_size != 0 || offset >= file_size ||
        io_bytes > file_size - offset) {
        throw Error(ErrorCode::InvalidArgument,
                    "store I/O is outside allocated payload or is misaligned",
                    "offset " + std::to_string(offset) +
                        " bytes " + std::to_string(bytes));
    }
}

#if defined(_WIN32)

std::string last_error_message(DWORD code) {
    char buf[512] = {};
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                   nullptr, code, 0, buf, sizeof(buf), nullptr);
    return buf;
}

std::string windows_error(DWORD code, const char* what) {
    return std::string(what) + " failed with error " + std::to_string(code) +
           ": " + last_error_message(code);
}

std::wstring utf8_path_to_wide(const std::string& path) {
    if (path.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw Error(ErrorCode::InvalidArgument,
                    "store path is too long to convert from UTF-8");
    }
    const int input_size = static_cast<int>(path.size());
    const int wide_size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                               path.data(), input_size,
                                               nullptr, 0);
    if (wide_size <= 0) {
        throw Error(ErrorCode::InvalidArgument,
                    "store path is not valid UTF-8",
                    windows_error(GetLastError(), "MultiByteToWideChar"));
    }
    std::wstring wide(static_cast<std::size_t>(wide_size), L'\0');
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                            path.data(), input_size,
                            wide.data(), wide_size) != wide_size) {
        throw Error(ErrorCode::InvalidArgument,
                    "store path could not be converted from UTF-8",
                    windows_error(GetLastError(), "MultiByteToWideChar"));
    }
    return wide;
}

#endif

// ---------------------------------------------------------------------------
// Async operation state
// ---------------------------------------------------------------------------

class AsyncOpImpl final : public AsyncOp {
public:
    void wait() override {
        std::unique_lock lock(mu_);
        cv_.wait(lock, [this] { return status_ != Status::Pending; });
        if (status_ == Status::Failed && error_) {
            throw *error_;
        }
    }

    Status status() const noexcept override {
        std::lock_guard lock(mu_);
        return status_;
    }

    uint64_t bytes_done() const noexcept override {
        std::lock_guard lock(mu_);
        return bytes_done_;
    }

    bool is_complete() const noexcept override {
        std::lock_guard lock(mu_);
        return status_ != Status::Pending;
    }

    void complete(uint64_t bytes) {
        {
            std::lock_guard lock(mu_);
            if (status_ != Status::Pending) return;
            bytes_done_ = bytes;
            status_ = Status::Complete;
        }
        cv_.notify_all();
    }

    void fail(Error err) {
        {
            std::lock_guard lock(mu_);
            if (status_ != Status::Pending) return;
            error_.emplace(std::move(err));
            status_ = Status::Failed;
        }
        cv_.notify_all();
    }

    void cancel() {
        {
            std::lock_guard lock(mu_);
            if (status_ != Status::Pending) return;
            status_ = Status::Cancelled;
        }
        cv_.notify_all();
    }

private:
    mutable std::mutex mu_;
    std::condition_variable cv_;
    Status status_ = Status::Pending;
    uint64_t bytes_done_ = 0;
    std::optional<Error> error_;
};

}  // namespace

// ---------------------------------------------------------------------------
// Platform implementation
// ---------------------------------------------------------------------------

#if defined(_WIN32)

struct NvmeBackend::Impl {
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE iocp = nullptr;
    std::vector<std::thread> workers;
    std::atomic<bool> stopping{false};

    uint64_t page_size = 0;
    uint64_t capacity_bytes = 0;
    uint64_t file_size = 0;  // capacity_bytes + one header page
    uint64_t extent_count = 0;
    std::atomic<uint64_t> used_extents{0};

    std::mutex mu;                       // guards free list + ops
    std::vector<bool> used;
    std::vector<uint32_t> active_io;
    std::vector<uint64_t> free_list;
    std::unordered_map<OVERLAPPED*, std::weak_ptr<AsyncOpImpl>> pending;

    std::string path;

    // Single authoritative shutdown path (idempotent):
    //   1. stop accepting new work;
    //   2. cancel pending operations with CancelIoEx;
    //   3. post one shutdown completion packet per worker;
    //   4. join every worker;
    //   5. clear pending state;
    //   6. close the file and IOCP handles.
    void stop_workers() {
        std::vector<std::thread> local;
        {
            std::lock_guard lock(mu);
            if (workers.empty() && file == INVALID_HANDLE_VALUE && iocp == nullptr) {
                return;  // already stopped or never started
            }
            stopping = true;
            local.swap(workers);
        }

        if (file != INVALID_HANDLE_VALUE) {
            std::vector<OVERLAPPED*> to_cancel;
            {
                std::lock_guard lock(mu);
                to_cancel.reserve(pending.size());
                for (const auto& [ov, weak] : pending) {
                    (void)weak;
                    to_cancel.push_back(ov);
                }
            }
            for (OVERLAPPED* ov : to_cancel) {
                CancelIoEx(file, ov);  // failure = op already completed
            }
        }

        if (iocp != nullptr) {
            // Wake each worker. A worker only exits after the pending map is
            // empty, so a sentinel can never overtake cancellation
            // completions and strand an AsyncOp in Pending.
            const std::size_t n = local.size();
            for (std::size_t i = 0; i < n; ++i) {
                (void)PostQueuedCompletionStatus(iocp, 0, 0, nullptr);
            }
        }

        for (auto& t : local) {
            if (t.joinable()) t.join();
        }

        {
            std::lock_guard lock(mu);
            pending.clear();
        }
        if (file != INVALID_HANDLE_VALUE) {
            CloseHandle(file);
            file = INVALID_HANDLE_VALUE;
        }
        if (iocp != nullptr) {
            CloseHandle(iocp);
            iocp = nullptr;
        }
    }

    ~Impl() {
        try {
            stop_workers();
        } catch (...) {
            // Destructor must not throw; close() is the propagation path.
        }
    }
};

namespace {

struct IocpOverlapped {
    OVERLAPPED ov{};
    NvmeBackend::Impl* impl = nullptr;
    std::shared_ptr<AsyncOpImpl> op;
    uint8_t* dst = nullptr;
    const uint8_t* src = nullptr;
    std::size_t bytes = 0;
    uint64_t first_extent = 0;
    uint64_t extent_count = 0;
    bool is_read = false;
};

IocpOverlapped* container_of(OVERLAPPED* ov) {
    return reinterpret_cast<IocpOverlapped*>(reinterpret_cast<char*>(ov) -
                                             offsetof(IocpOverlapped, ov));
}

bool iocp_shutdown_drained(NvmeBackend::Impl* impl) {
    std::lock_guard lock(impl->mu);
    return impl->stopping && impl->pending.empty();
}

void retire_iocp_operation(NvmeBackend::Impl* impl, IocpOverlapped* io) {
    std::lock_guard lock(impl->mu);
    impl->pending.erase(&io->ov);
    for (uint64_t i = 0; i < io->extent_count; ++i) {
        const uint64_t idx = io->first_extent + i;
        if (idx < impl->active_io.size() && impl->active_io[idx] != 0) {
            --impl->active_io[idx];
        }
    }
}

// Synchronous I/O on an overlapped handle: submit with an OVERLAPPED whose
// event is null and wait via GetOverlappedResult.
bool sync_read(HANDLE file, void* buf, DWORD bytes, DWORD& done) {
    OVERLAPPED ov{};
    const BOOL ok = ReadFile(file, buf, bytes, &done, &ov);
    if (ok) return true;
    if (GetLastError() == ERROR_IO_PENDING) {
        return GetOverlappedResult(file, &ov, &done, TRUE) != FALSE;
    }
    return false;
}

bool sync_write(HANDLE file, const void* buf, DWORD bytes, DWORD& done) {
    OVERLAPPED ov{};
    const BOOL ok = WriteFile(file, buf, bytes, &done, &ov);
    if (ok) return true;
    if (GetLastError() == ERROR_IO_PENDING) {
        return GetOverlappedResult(file, &ov, &done, TRUE) != FALSE;
    }
    return false;
}

// Completion worker. Each completed/cancelled/failed OVERLAPPED is
// processed exactly once and erased from the pending map. Shutdown:
// cancellation completions drain before workers exit. Sentinels only wake
// the pool; the pending map is the authoritative drain condition.
void iocp_worker(NvmeBackend::Impl* impl) {
    while (true) {
        DWORD bytes = 0;
        ULONG_PTR key = 0;
        OVERLAPPED* ov = nullptr;
        if (!GetQueuedCompletionStatus(impl->iocp, &bytes, &key, &ov, 200)) {
            const DWORD err = GetLastError();
            if (err == WAIT_TIMEOUT) {
                if (iocp_shutdown_drained(impl)) break;
                continue;
            }
            if (ov != nullptr) {
                IocpOverlapped* io = container_of(ov);
                retire_iocp_operation(impl, io);
                if (err == ERROR_OPERATION_ABORTED) {
                    io->op->cancel();
                } else {
                    io->op->fail(Error(ErrorCode::Io,
                                       windows_error(err, "overlapped I/O")));
                }
                delete io;
                continue;
            }
            if (iocp_shutdown_drained(impl)) break;
            continue;
        }
        if (ov == nullptr) {
            if (iocp_shutdown_drained(impl)) break;
            continue;
        }
        IocpOverlapped* io = container_of(ov);
        retire_iocp_operation(impl, io);
        if (io->op->status() == AsyncOp::Status::Cancelled) {
            // Operation was cancelled; do not touch user buffers.
        } else if (bytes != io->bytes) {
            io->op->fail(Error(ErrorCode::Io,
                               "short overlapped store I/O",
                               "expected " + std::to_string(io->bytes) +
                                   " bytes, completed " + std::to_string(bytes)));
        } else {
            io->op->complete(bytes);
        }
        delete io;
    }
}

}  // namespace

NvmeBackend::NvmeBackend() : impl_(std::make_unique<Impl>()) {}

NvmeBackend::~NvmeBackend() {
    try {
        close();
    } catch (...) {
        // Destructors are noexcept; close() from explicit callers is the
        // error-propagation path.
    }
}

StorageBackend::Info NvmeBackend::open(const std::string& path,
                                       uint64_t capacity_bytes,
                                       uint64_t page_size) {
    std::lock_guard lifecycle_lock(lifecycle_mu_);
    close();  // release live handles from a previous (possibly failed) open
    validate_store_geometry(path, capacity_bytes, page_size,
                            static_cast<uint64_t>(std::numeric_limits<LONGLONG>::max()),
                            static_cast<uint64_t>(std::numeric_limits<DWORD>::max()));
    const std::wstring wide_path = utf8_path_to_wide(path);

    impl_->path = path;
    impl_->page_size = page_size;
    impl_->capacity_bytes = capacity_bytes;
    impl_->file_size = capacity_bytes + page_size;  // header page + payload
    impl_->extent_count = capacity_bytes / page_size;
    try {
        impl_->used.assign(static_cast<std::size_t>(impl_->extent_count), false);
        impl_->active_io.assign(static_cast<std::size_t>(impl_->extent_count), 0);
        impl_->free_list.clear();
        impl_->free_list.reserve(static_cast<std::size_t>(impl_->extent_count));
        for (uint64_t idx = 0; idx < impl_->extent_count; ++idx) {
            impl_->free_list.push_back(idx);  // every extent starts free
        }
    } catch (const std::bad_alloc&) {
        throw Error(ErrorCode::InvalidArgument,
                    "store geometry requires too much allocation metadata");
    } catch (const std::length_error&) {
        throw Error(ErrorCode::InvalidArgument,
                    "store extent count exceeds the allocation metadata limit");
    }
    impl_->used_extents = 0;

    bool created = true;
    impl_->file = CreateFileW(wide_path.c_str(),
                              GENERIC_READ | GENERIC_WRITE,
                              FILE_SHARE_READ,
                              nullptr, CREATE_NEW,
                              FILE_FLAG_OVERLAPPED,
                              nullptr);
    if (impl_->file == INVALID_HANDLE_VALUE) {
        const DWORD create_error = GetLastError();
        if (create_error != ERROR_FILE_EXISTS && create_error != ERROR_ALREADY_EXISTS) {
            throw Error(ErrorCode::Io, windows_error(create_error, "CreateFile"), path);
        }
        created = false;
        impl_->file = CreateFileW(wide_path.c_str(),
                                  GENERIC_READ | GENERIC_WRITE,
                                  FILE_SHARE_READ,
                                  nullptr, OPEN_EXISTING,
                                  FILE_FLAG_OVERLAPPED,
                                  nullptr);
        if (impl_->file == INVALID_HANDLE_VALUE) {
            throw Error(ErrorCode::Io,
                        windows_error(GetLastError(), "open existing store"), path);
        }
    }

    if (!created) {
        detail::StoreHeader existing{};
        DWORD got = 0;
        if (!sync_read(impl_->file, &existing, sizeof(existing), got)) {
            const DWORD err = GetLastError();
            close();
            throw Error(ErrorCode::Io, windows_error(err, "ReadFile(header)"), path);
        }
        if (got != sizeof(existing)) {
            close();
            throw Error(ErrorCode::StoreCorrupt,
                        "existing store has truncated header", path);
        }
        const std::string reason =
            detail::header_mismatch_reason(existing, page_size, capacity_bytes);
        if (!reason.empty()) {
            close();
            throw Error(ErrorCode::StoreCorrupt,
                        "existing store rejected: " + reason, path);
        }

        LARGE_INTEGER actual_size{};
        if (!GetFileSizeEx(impl_->file, &actual_size)) {
            const DWORD err = GetLastError();
            close();
            throw Error(ErrorCode::Io, windows_error(err, "GetFileSizeEx"), path);
        }
        if (actual_size.QuadPart != static_cast<LONGLONG>(impl_->file_size)) {
            const LONGLONG got_size = actual_size.QuadPart;
            close();
            throw Error(ErrorCode::StoreCorrupt,
                        "existing store file size does not match its header",
                        "expected " + std::to_string(impl_->file_size) +
                            " bytes, found " + std::to_string(got_size));
        }
        // The free map is rebuilt from the runtime page table on each run;
        // the header carries identity, not extent usage.
    } else {
        LARGE_INTEGER size{};
        size.QuadPart = static_cast<LONGLONG>(impl_->file_size);
        if (!SetFilePointerEx(impl_->file, size, nullptr, FILE_BEGIN) ||
            !SetEndOfFile(impl_->file)) {
            const DWORD err = GetLastError();
            close();
            destroy_file(path);
            throw Error(ErrorCode::Io, windows_error(err, "SetEndOfFile"), path);
        }

        detail::StoreHeader h{};
        h.magic = detail::kStoreMagic;
        h.version = detail::kStoreVersion;
        h.page_size = page_size;
        h.capacity_bytes = capacity_bytes;
        h.extent_count = impl_->extent_count;
        h.generation = 1;
        h.seed = 0xC0FFEE;
        detail::seal_store_header(h);
        DWORD written = 0;
        if (!sync_write(impl_->file, &h, sizeof(h), written) || written != sizeof(h)) {
            const DWORD err = GetLastError();
            close();
            destroy_file(path);
            throw Error(ErrorCode::Io, windows_error(err, "WriteFile(header)"), path);
        }
        if (!FlushFileBuffers(impl_->file)) {
            const DWORD err = GetLastError();
            close();
            destroy_file(path);
            throw Error(ErrorCode::Io, windows_error(err, "FlushFileBuffers"), path);
        }
    }

    impl_->iocp = CreateIoCompletionPort(impl_->file, nullptr, 0, 0);
    if (impl_->iocp == nullptr) {
        close();
        throw Error(ErrorCode::Io, windows_error(GetLastError(), "CreateIoCompletionPort"));
    }

    impl_->stopping = false;
    const unsigned worker_count = 4;
    try {
        for (unsigned i = 0; i < worker_count; ++i) {
            impl_->workers.emplace_back(iocp_worker, impl_.get());
        }
    } catch (...) {
        close();
        throw;
    }

    opened_ = true;
    StorageBackend::Info info;
    info.capacity_bytes = capacity_bytes;
    info.page_size = page_size;
    info.extent_count = impl_->extent_count;
    info.path = path;
    return info;
}

bool NvmeBackend::allocate_extent(uint64_t& offset_out) {
    std::lock_guard lifecycle_lock(lifecycle_mu_);
    if (!opened_) throw Error(ErrorCode::State, "store not open");
    std::lock_guard lock(impl_->mu);
    if (impl_->stopping) throw Error(ErrorCode::State, "store is closing");
    if (impl_->free_list.empty()) return false;
    const uint64_t idx = impl_->free_list.back();
    impl_->free_list.pop_back();
    impl_->used[idx] = true;
    ++impl_->used_extents;
    offset_out = (idx + 1) * impl_->page_size;  // extent 0 starts after the header block
    return true;
}

void NvmeBackend::free_extent(uint64_t offset) {
    free_extents(offset, 1);
}

bool NvmeBackend::allocate_extents(uint64_t extent_count, uint64_t& offset_out) {
    std::lock_guard lifecycle_lock(lifecycle_mu_);
    if (!opened_) throw Error(ErrorCode::State, "store not open");
    if (extent_count == 0) {
        throw Error(ErrorCode::InvalidArgument, "extent count must be nonzero");
    }
    if (extent_count == 1) return allocate_extent(offset_out);

    std::lock_guard lock(impl_->mu);
    if (impl_->stopping) throw Error(ErrorCode::State, "store is closing");
    if (extent_count > impl_->extent_count - impl_->used_extents.load()) return false;

    uint64_t run = 0;
    uint64_t first = 0;
    bool found = false;
    for (uint64_t cursor = impl_->extent_count; cursor != 0; --cursor) {
        const uint64_t idx = cursor - 1;
        if (!impl_->used[static_cast<std::size_t>(idx)]) {
            ++run;
            if (run == extent_count) {
                first = idx;
                found = true;
                break;
            }
        } else {
            run = 0;
        }
    }
    if (!found) return false;

    for (uint64_t i = 0; i < extent_count; ++i) {
        impl_->used[static_cast<std::size_t>(first + i)] = true;
    }
    const uint64_t end = first + extent_count;
    impl_->free_list.erase(
        std::remove_if(impl_->free_list.begin(), impl_->free_list.end(),
                       [first, end](uint64_t idx) { return idx >= first && idx < end; }),
        impl_->free_list.end());
    impl_->used_extents += extent_count;
    offset_out = (first + 1) * impl_->page_size;
    return true;
}

void NvmeBackend::free_extents(uint64_t offset, uint64_t extent_count) {
    std::lock_guard lifecycle_lock(lifecycle_mu_);
    if (!opened_) throw Error(ErrorCode::State, "store not open");
    if (extent_count == 0 || offset < impl_->page_size ||
        offset % impl_->page_size != 0) {
        throw Error(ErrorCode::InvalidArgument, "invalid extent range",
                    std::to_string(offset));
    }
    const uint64_t first = offset / impl_->page_size - 1;
    if (extent_count > impl_->extent_count ||
        first > impl_->extent_count - extent_count) {
        throw Error(ErrorCode::InvalidArgument, "invalid extent range",
                    std::to_string(offset));
    }

    std::lock_guard lock(impl_->mu);
    if (impl_->stopping) throw Error(ErrorCode::State, "store is closing");
    for (uint64_t i = 0; i < extent_count; ++i) {
        const std::size_t idx = static_cast<std::size_t>(first + i);
        if (!impl_->used[idx]) {
            throw Error(ErrorCode::State, "double free of store extent range",
                        std::to_string(offset));
        }
        if (impl_->active_io[idx] != 0) {
            throw Error(ErrorCode::State, "cannot free an extent with active I/O",
                        std::to_string(offset));
        }
    }
    for (uint64_t i = 0; i < extent_count; ++i) {
        const uint64_t idx = first + i;
        impl_->used[static_cast<std::size_t>(idx)] = false;
        impl_->free_list.push_back(idx);
    }
    impl_->used_extents -= extent_count;
}

uint64_t NvmeBackend::extents_used() const noexcept {
    return opened_ ? impl_->used_extents.load() : 0;
}

AsyncOpPtr NvmeBackend::read_async(uint64_t offset, void* dst, std::size_t bytes) {
    std::lock_guard lifecycle_lock(lifecycle_mu_);
    if (!opened_) throw Error(ErrorCode::State, "store not open");
    validate_io_request(impl_->page_size, impl_->file_size, offset, dst, bytes,
                        static_cast<uint64_t>(std::numeric_limits<DWORD>::max()));

    auto op = std::make_shared<AsyncOpImpl>();
    auto* io = new IocpOverlapped();
    io->impl = impl_.get();
    io->op = op;
    io->dst = static_cast<uint8_t*>(dst);
    io->bytes = bytes;
    io->first_extent = offset / impl_->page_size - 1;
    io->extent_count = static_cast<uint64_t>(bytes) / impl_->page_size;
    io->is_read = true;
    io->ov.Offset = static_cast<DWORD>(offset & 0xFFFFFFFFull);
    io->ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
    io->ov.hEvent = nullptr;  // completion goes to the IOCP

    {
        std::lock_guard lock(impl_->mu);
        if (impl_->stopping || impl_->file == INVALID_HANDLE_VALUE) {
            delete io;
            throw Error(ErrorCode::State, "store is closing");
        }
        for (uint64_t i = 0; i < io->extent_count; ++i) {
            const std::size_t idx = static_cast<std::size_t>(io->first_extent + i);
            if (!impl_->used[idx]) {
                delete io;
                throw Error(ErrorCode::State, "store read targets an unallocated extent");
            }
        }
        impl_->pending.emplace(&io->ov, op);
        for (uint64_t i = 0; i < io->extent_count; ++i) {
            ++impl_->active_io[static_cast<std::size_t>(io->first_extent + i)];
        }
    }

    DWORD unused = 0;
    const BOOL ok = ReadFile(impl_->file, dst, static_cast<DWORD>(bytes), &unused, &io->ov);
    if (!ok) {
        const DWORD err = GetLastError();
        if (err != ERROR_IO_PENDING) {
            retire_iocp_operation(impl_.get(), io);
            delete io;
            throw Error(ErrorCode::Io, windows_error(err, "ReadFile"));
        }
    }
    return op;
}

AsyncOpPtr NvmeBackend::write_async(uint64_t offset, const void* src, std::size_t bytes) {
    std::lock_guard lifecycle_lock(lifecycle_mu_);
    if (!opened_) throw Error(ErrorCode::State, "store not open");
    validate_io_request(impl_->page_size, impl_->file_size, offset, src, bytes,
                        static_cast<uint64_t>(std::numeric_limits<DWORD>::max()));

    auto op = std::make_shared<AsyncOpImpl>();
    auto* io = new IocpOverlapped();
    io->impl = impl_.get();
    io->op = op;
    io->src = static_cast<const uint8_t*>(src);
    io->bytes = bytes;
    io->first_extent = offset / impl_->page_size - 1;
    io->extent_count = static_cast<uint64_t>(bytes) / impl_->page_size;
    io->is_read = false;
    io->ov.Offset = static_cast<DWORD>(offset & 0xFFFFFFFFull);
    io->ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
    io->ov.hEvent = nullptr;

    {
        std::lock_guard lock(impl_->mu);
        if (impl_->stopping || impl_->file == INVALID_HANDLE_VALUE) {
            delete io;
            throw Error(ErrorCode::State, "store is closing");
        }
        for (uint64_t i = 0; i < io->extent_count; ++i) {
            const std::size_t idx = static_cast<std::size_t>(io->first_extent + i);
            if (!impl_->used[idx]) {
                delete io;
                throw Error(ErrorCode::State, "store write targets an unallocated extent");
            }
        }
        impl_->pending.emplace(&io->ov, op);
        for (uint64_t i = 0; i < io->extent_count; ++i) {
            ++impl_->active_io[static_cast<std::size_t>(io->first_extent + i)];
        }
    }

    DWORD unused = 0;
    const BOOL ok = WriteFile(impl_->file, src, static_cast<DWORD>(bytes), &unused, &io->ov);
    if (!ok) {
        const DWORD err = GetLastError();
        if (err != ERROR_IO_PENDING) {
            retire_iocp_operation(impl_.get(), io);
            delete io;
            throw Error(ErrorCode::Io, windows_error(err, "WriteFile"));
        }
    }
    return op;
}

void NvmeBackend::flush() {
    std::lock_guard lifecycle_lock(lifecycle_mu_);
    if (!opened_) return;
    std::vector<std::shared_ptr<AsyncOpImpl>> pending;
    {
        std::lock_guard lock(impl_->mu);
        pending.reserve(impl_->pending.size());
        for (const auto& [ov, weak] : impl_->pending) {
            (void)ov;
            if (auto op = weak.lock()) pending.push_back(std::move(op));
        }
    }
    for (const auto& op : pending) op->wait();
    if (!FlushFileBuffers(impl_->file)) {
        throw Error(ErrorCode::Io, windows_error(GetLastError(), "FlushFileBuffers"));
    }
}

void NvmeBackend::cancel_all() {
    std::lock_guard lifecycle_lock(lifecycle_mu_);
    if (!opened_) return;
    std::vector<OVERLAPPED*> ovs;
    {
        std::lock_guard lock(impl_->mu);
        for (auto& [ov, weak] : impl_->pending) {
            (void)weak;
            ovs.push_back(ov);
        }
    }
    for (OVERLAPPED* ov : ovs) {
        CancelIoEx(impl_->file, ov);
    }
}

void NvmeBackend::close() {
    std::lock_guard lifecycle_lock(lifecycle_mu_);
    // close() is the authoritative handle-release path even when open()
    // failed midway: a failed open() leaves impl_->file live, and any
    // subsequent open()/close()/destructor must release it or the file
    // stays pinned on disk (silently breaking deletion).
    if (!opened_ && impl_->file == INVALID_HANDLE_VALUE && impl_->iocp == nullptr) {
        return;
    }
    impl_->stop_workers();
    {
        std::lock_guard lock(impl_->mu);
        impl_->used.clear();
        impl_->active_io.clear();
        impl_->free_list.clear();
        impl_->used_extents = 0;
        impl_->stopping = false;  // allow reuse of the Impl by a future open()
    }
    opened_ = false;
}

void NvmeBackend::destroy_file(const std::string& path) {
    if (path.empty() || path.find('\0') != std::string::npos) {
        throw Error(ErrorCode::InvalidArgument, "store path is empty or contains NUL");
    }
    const std::wstring wide_path = utf8_path_to_wide(path);
    // Deletion can transiently fail with a sharing violation (antivirus
    // scan, lazy handle teardown); bounded retry keeps shutdown honest
    // without blind polling.
    for (int attempt = 0; attempt < 3; ++attempt) {
        if (DeleteFileW(wide_path.c_str()) || GetLastError() == ERROR_FILE_NOT_FOUND) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    const DWORD err = GetLastError();
    throw Error(ErrorCode::Io, windows_error(err, "DeleteFile"), path);
}

#else  // POSIX (Linux)

// Simple threaded pread/pwrite backend with the same AsyncOp interface.
// Synchronous kernel calls dispatched on a bounded worker pool.

namespace {
struct PosixOpContext;
}

struct NvmeBackend::Impl {
    int fd = -1;
    uint64_t page_size = 0;
    uint64_t capacity_bytes = 0;
    uint64_t file_size = 0;  // capacity_bytes + one header page
    uint64_t extent_count = 0;
    std::atomic<uint64_t> used_extents{0};
    std::mutex mu;
    std::vector<bool> used;
    std::vector<uint32_t> active_io;
    std::vector<uint64_t> free_list;
    std::vector<std::thread> workers;
    std::deque<std::shared_ptr<PosixOpContext>> queue;
    std::unordered_map<AsyncOpImpl*, std::weak_ptr<AsyncOpImpl>> pending;
    std::condition_variable cv;
    bool stopping = false;
    std::string path;

    ~Impl();
    void worker_loop();
    void run(const std::shared_ptr<PosixOpContext>& ctx);
};

namespace {

struct PosixOpContext {
    NvmeBackend::Impl* impl = nullptr;
    std::shared_ptr<AsyncOpImpl> op;
    int fd = -1;
    uint64_t offset = 0;
    void* buf = nullptr;
    std::size_t bytes = 0;
    uint64_t first_extent = 0;
    uint64_t extent_count = 0;
    bool is_read = false;
};

void retire_posix_operation(const std::shared_ptr<PosixOpContext>& ctx) {
    std::lock_guard lock(ctx->impl->mu);
    ctx->impl->pending.erase(ctx->op.get());
    for (uint64_t i = 0; i < ctx->extent_count; ++i) {
        const std::size_t idx = static_cast<std::size_t>(ctx->first_extent + i);
        if (ctx->impl->active_io[idx] != 0) --ctx->impl->active_io[idx];
    }
}

}  // namespace

NvmeBackend::Impl::~Impl() {
        {
            std::lock_guard lock(mu);
            stopping = true;
        }
        cv.notify_all();
        for (auto& t : workers) {
            if (t.joinable()) t.join();
        }
        if (fd >= 0) ::close(fd);
}

void NvmeBackend::Impl::worker_loop() {
        while (true) {
            std::shared_ptr<PosixOpContext> ctx;
            {
                std::unique_lock lock(mu);
                cv.wait(lock, [this] { return stopping || !queue.empty(); });
                if (stopping && queue.empty()) return;
                ctx = std::move(queue.front());
                queue.pop_front();
            }
            run(ctx);
        }
}

void NvmeBackend::Impl::run(const std::shared_ptr<PosixOpContext>& ctx) {
        std::size_t total = 0;
        while (total < ctx->bytes) {
            ssize_t r = ctx->is_read
                            ? ::pread(ctx->fd, static_cast<char*>(ctx->buf) + total,
                                      ctx->bytes - total,
                                      static_cast<off_t>(ctx->offset + total))
                            : ::pwrite(ctx->fd, static_cast<const char*>(ctx->buf) + total,
                                       ctx->bytes - total,
                                       static_cast<off_t>(ctx->offset + total));
            if (r < 0) {
                if (errno == EINTR) continue;
                const int err = errno;
                retire_posix_operation(ctx);
                ctx->op->fail(Error(ErrorCode::Io,
                                    std::string(ctx->is_read ? "pread" : "pwrite") +
                                        " failed: " + std::strerror(err)));
                return;
            }
            if (r == 0) {
                retire_posix_operation(ctx);
                ctx->op->fail(Error(
                    ErrorCode::Io,
                    ctx->is_read ? "unexpected EOF in store read"
                                 : "zero-byte progress in store write"));
                return;
            }
            total += static_cast<std::size_t>(r);
        }
        retire_posix_operation(ctx);
        ctx->op->complete(total);
}

NvmeBackend::NvmeBackend() : impl_(std::make_unique<Impl>()) {}

NvmeBackend::~NvmeBackend() {
    try {
        close();
    } catch (...) {
        // Destructors are noexcept; close() from explicit callers is the
        // error-propagation path.
    }
}

StorageBackend::Info NvmeBackend::open(const std::string& path,
                                       uint64_t capacity_bytes,
                                       uint64_t page_size) {
    std::lock_guard lifecycle_lock(lifecycle_mu_);
    close();  // release any live fd from a previous (possibly failed) open
    validate_store_geometry(
        path, capacity_bytes, page_size,
        static_cast<uint64_t>(std::numeric_limits<off_t>::max()),
        static_cast<uint64_t>(std::numeric_limits<ssize_t>::max()));
    impl_->path = path;
    impl_->page_size = page_size;
    impl_->capacity_bytes = capacity_bytes;
    impl_->file_size = capacity_bytes + page_size;  // header page + payload
    impl_->extent_count = capacity_bytes / page_size;
    try {
        impl_->used.assign(static_cast<std::size_t>(impl_->extent_count), false);
        impl_->active_io.assign(static_cast<std::size_t>(impl_->extent_count), 0);
        impl_->free_list.clear();
        impl_->free_list.reserve(static_cast<std::size_t>(impl_->extent_count));
        for (uint64_t idx = 0; idx < impl_->extent_count; ++idx) {
            impl_->free_list.push_back(idx);  // every extent starts free
        }
    } catch (const std::bad_alloc&) {
        throw Error(ErrorCode::InvalidArgument,
                    "store geometry requires too much allocation metadata");
    } catch (const std::length_error&) {
        throw Error(ErrorCode::InvalidArgument,
                    "store extent count exceeds the allocation metadata limit");
    }
    impl_->used_extents = 0;

    bool created = true;
    impl_->fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (impl_->fd < 0) {
        const int create_error = errno;
        if (create_error != EEXIST) {
            throw Error(ErrorCode::Io,
                        std::string("create store failed: ") +
                            std::strerror(create_error),
                        path);
        }
        created = false;
        impl_->fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
    }
    if (impl_->fd < 0) {
        throw Error(ErrorCode::Io,
                    std::string("open store failed: ") + std::strerror(errno), path);
    }
    if (::flock(impl_->fd, LOCK_EX | LOCK_NB) != 0) {
        const int err = errno;
        ::close(impl_->fd);
        impl_->fd = -1;
        throw Error(ErrorCode::Io,
                    std::string("store is already open or cannot be locked: ") +
                        std::strerror(err),
                    path);
    }

    struct stat st {};
    if (::fstat(impl_->fd, &st) != 0) {
        const int err = errno;
        close();
        if (created) destroy_file(path);
        throw Error(ErrorCode::Io,
                    std::string("fstat store failed: ") + std::strerror(err), path);
    }
    if (!S_ISREG(st.st_mode)) {
        close();
        throw Error(ErrorCode::InvalidArgument, "store path is not a regular file", path);
    }

    if (!created) {
        detail::StoreHeader h{};
        std::size_t total = 0;
        while (total < sizeof(h)) {
            const ssize_t got = ::pread(impl_->fd,
                                        reinterpret_cast<char*>(&h) + total,
                                        sizeof(h) - total,
                                        static_cast<off_t>(total));
            if (got < 0 && errno == EINTR) continue;
            if (got < 0) {
                const int err = errno;
                close();
                throw Error(ErrorCode::Io,
                            std::string("pread header failed: ") + std::strerror(err),
                            path);
            }
            if (got == 0) break;
            total += static_cast<std::size_t>(got);
        }
        if (total != sizeof(h)) {
            close();
            throw Error(ErrorCode::StoreCorrupt,
                        "existing store has truncated header", path);
        }
        const std::string reason =
            detail::header_mismatch_reason(h, page_size, capacity_bytes);
        if (!reason.empty()) {
            close();
            throw Error(ErrorCode::StoreCorrupt,
                        "existing store rejected: " + reason, path);
        }
        if (st.st_size < 0 ||
            static_cast<uint64_t>(st.st_size) != impl_->file_size) {
            const int64_t actual_size = static_cast<int64_t>(st.st_size);
            close();
            throw Error(ErrorCode::StoreCorrupt,
                        "existing store file size does not match its header",
                        "expected " + std::to_string(impl_->file_size) +
                            " bytes, found " + std::to_string(actual_size));
        }
    } else {
        if (::ftruncate(impl_->fd, static_cast<off_t>(impl_->file_size)) != 0) {
            const int err = errno;
            close();
            destroy_file(path);
            throw Error(ErrorCode::Io,
                        std::string("ftruncate failed: ") + std::strerror(err), path);
        }

        detail::StoreHeader h{};
        h.magic = detail::kStoreMagic;
        h.version = detail::kStoreVersion;
        h.page_size = page_size;
        h.capacity_bytes = capacity_bytes;
        h.extent_count = impl_->extent_count;
        h.generation = 1;
        h.seed = 0xC0FFEE;
        detail::seal_store_header(h);
        std::size_t total = 0;
        while (total < sizeof(h)) {
            const ssize_t written = ::pwrite(impl_->fd,
                                             reinterpret_cast<const char*>(&h) + total,
                                             sizeof(h) - total,
                                             static_cast<off_t>(total));
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) {
                const int err = errno;
                close();
                destroy_file(path);
                throw Error(ErrorCode::Io,
                            std::string("pwrite header failed: ") +
                                (written < 0 ? std::strerror(err) : "zero-byte write"),
                            path);
            }
            total += static_cast<std::size_t>(written);
        }
        if (::fsync(impl_->fd) != 0) {
            const int err = errno;
            close();
            destroy_file(path);
            throw Error(ErrorCode::Io,
                        std::string("fsync header failed: ") + std::strerror(err), path);
        }
    }

    {
        std::lock_guard lock(impl_->mu);
        impl_->stopping = false;
    }
    const unsigned worker_count = 4;
    try {
        for (unsigned i = 0; i < worker_count; ++i) {
            impl_->workers.emplace_back([this] { impl_->worker_loop(); });
        }
    } catch (...) {
        close();
        throw;
    }

    opened_ = true;
    StorageBackend::Info info;
    info.capacity_bytes = capacity_bytes;
    info.page_size = page_size;
    info.extent_count = impl_->extent_count;
    info.path = path;
    return info;
}

bool NvmeBackend::allocate_extent(uint64_t& offset_out) {
    std::lock_guard lifecycle_lock(lifecycle_mu_);
    if (!opened_) throw Error(ErrorCode::State, "store not open");
    std::lock_guard lock(impl_->mu);
    if (impl_->stopping) throw Error(ErrorCode::State, "store is closing");
    if (impl_->free_list.empty()) return false;
    const uint64_t idx = impl_->free_list.back();
    impl_->free_list.pop_back();
    impl_->used[idx] = true;
    ++impl_->used_extents;
    offset_out = (idx + 1) * impl_->page_size;
    return true;
}

void NvmeBackend::free_extent(uint64_t offset) {
    free_extents(offset, 1);
}

bool NvmeBackend::allocate_extents(uint64_t extent_count, uint64_t& offset_out) {
    std::lock_guard lifecycle_lock(lifecycle_mu_);
    if (!opened_) throw Error(ErrorCode::State, "store not open");
    if (extent_count == 0) {
        throw Error(ErrorCode::InvalidArgument, "extent count must be nonzero");
    }
    if (extent_count == 1) return allocate_extent(offset_out);

    std::lock_guard lock(impl_->mu);
    if (impl_->stopping) throw Error(ErrorCode::State, "store is closing");
    if (extent_count > impl_->extent_count - impl_->used_extents.load()) return false;

    uint64_t run = 0;
    uint64_t first = 0;
    bool found = false;
    for (uint64_t cursor = impl_->extent_count; cursor != 0; --cursor) {
        const uint64_t idx = cursor - 1;
        if (!impl_->used[static_cast<std::size_t>(idx)]) {
            ++run;
            if (run == extent_count) {
                first = idx;
                found = true;
                break;
            }
        } else {
            run = 0;
        }
    }
    if (!found) return false;

    for (uint64_t i = 0; i < extent_count; ++i) {
        impl_->used[static_cast<std::size_t>(first + i)] = true;
    }
    const uint64_t end = first + extent_count;
    impl_->free_list.erase(
        std::remove_if(impl_->free_list.begin(), impl_->free_list.end(),
                       [first, end](uint64_t idx) { return idx >= first && idx < end; }),
        impl_->free_list.end());
    impl_->used_extents += extent_count;
    offset_out = (first + 1) * impl_->page_size;
    return true;
}

void NvmeBackend::free_extents(uint64_t offset, uint64_t extent_count) {
    std::lock_guard lifecycle_lock(lifecycle_mu_);
    if (!opened_) throw Error(ErrorCode::State, "store not open");
    if (extent_count == 0 || offset < impl_->page_size ||
        offset % impl_->page_size != 0) {
        throw Error(ErrorCode::InvalidArgument, "invalid extent range",
                    std::to_string(offset));
    }
    const uint64_t first = offset / impl_->page_size - 1;
    if (extent_count > impl_->extent_count ||
        first > impl_->extent_count - extent_count) {
        throw Error(ErrorCode::InvalidArgument, "invalid extent range",
                    std::to_string(offset));
    }

    std::lock_guard lock(impl_->mu);
    if (impl_->stopping) throw Error(ErrorCode::State, "store is closing");
    for (uint64_t i = 0; i < extent_count; ++i) {
        const std::size_t idx = static_cast<std::size_t>(first + i);
        if (!impl_->used[idx]) {
            throw Error(ErrorCode::State, "double free of store extent range",
                        std::to_string(offset));
        }
        if (impl_->active_io[idx] != 0) {
            throw Error(ErrorCode::State, "cannot free an extent with active I/O",
                        std::to_string(offset));
        }
    }
    for (uint64_t i = 0; i < extent_count; ++i) {
        const uint64_t idx = first + i;
        impl_->used[static_cast<std::size_t>(idx)] = false;
        impl_->free_list.push_back(idx);
    }
    impl_->used_extents -= extent_count;
}

uint64_t NvmeBackend::extents_used() const noexcept {
    return opened_ ? impl_->used_extents.load() : 0;
}

AsyncOpPtr NvmeBackend::read_async(uint64_t offset, void* dst, std::size_t bytes) {
    std::lock_guard lifecycle_lock(lifecycle_mu_);
    if (!opened_) throw Error(ErrorCode::State, "store not open");
    validate_io_request(
        impl_->page_size, impl_->file_size, offset, dst, bytes,
        static_cast<uint64_t>(std::numeric_limits<ssize_t>::max()));
    auto ctx = std::make_shared<PosixOpContext>();
    ctx->impl = impl_.get();
    ctx->op = std::make_shared<AsyncOpImpl>();
    ctx->fd = impl_->fd;
    ctx->offset = offset;
    ctx->buf = dst;
    ctx->bytes = bytes;
    ctx->first_extent = offset / impl_->page_size - 1;
    ctx->extent_count = static_cast<uint64_t>(bytes) / impl_->page_size;
    ctx->is_read = true;
    {
        std::lock_guard lock(impl_->mu);
        if (impl_->stopping || impl_->fd < 0) {
            throw Error(ErrorCode::State, "store is closing");
        }
        for (uint64_t i = 0; i < ctx->extent_count; ++i) {
            const std::size_t idx = static_cast<std::size_t>(ctx->first_extent + i);
            if (!impl_->used[idx]) {
                throw Error(ErrorCode::State, "store read targets an unallocated extent");
            }
        }
        impl_->pending.emplace(ctx->op.get(), ctx->op);
        try {
            impl_->queue.push_back(ctx);
        } catch (...) {
            impl_->pending.erase(ctx->op.get());
            throw;
        }
        for (uint64_t i = 0; i < ctx->extent_count; ++i) {
            ++impl_->active_io[static_cast<std::size_t>(ctx->first_extent + i)];
        }
    }
    impl_->cv.notify_one();
    return ctx->op;
}

AsyncOpPtr NvmeBackend::write_async(uint64_t offset, const void* src, std::size_t bytes) {
    std::lock_guard lifecycle_lock(lifecycle_mu_);
    if (!opened_) throw Error(ErrorCode::State, "store not open");
    validate_io_request(
        impl_->page_size, impl_->file_size, offset, src, bytes,
        static_cast<uint64_t>(std::numeric_limits<ssize_t>::max()));
    auto ctx = std::make_shared<PosixOpContext>();
    ctx->impl = impl_.get();
    ctx->op = std::make_shared<AsyncOpImpl>();
    ctx->fd = impl_->fd;
    ctx->offset = offset;
    ctx->buf = const_cast<void*>(src);
    ctx->bytes = bytes;
    ctx->first_extent = offset / impl_->page_size - 1;
    ctx->extent_count = static_cast<uint64_t>(bytes) / impl_->page_size;
    ctx->is_read = false;
    {
        std::lock_guard lock(impl_->mu);
        if (impl_->stopping || impl_->fd < 0) {
            throw Error(ErrorCode::State, "store is closing");
        }
        for (uint64_t i = 0; i < ctx->extent_count; ++i) {
            const std::size_t idx = static_cast<std::size_t>(ctx->first_extent + i);
            if (!impl_->used[idx]) {
                throw Error(ErrorCode::State, "store write targets an unallocated extent");
            }
        }
        impl_->pending.emplace(ctx->op.get(), ctx->op);
        try {
            impl_->queue.push_back(ctx);
        } catch (...) {
            impl_->pending.erase(ctx->op.get());
            throw;
        }
        for (uint64_t i = 0; i < ctx->extent_count; ++i) {
            ++impl_->active_io[static_cast<std::size_t>(ctx->first_extent + i)];
        }
    }
    impl_->cv.notify_one();
    return ctx->op;
}

void NvmeBackend::flush() {
    std::lock_guard lifecycle_lock(lifecycle_mu_);
    if (!opened_) return;
    std::vector<std::shared_ptr<AsyncOpImpl>> pending;
    {
        std::lock_guard lock(impl_->mu);
        pending.reserve(impl_->pending.size());
        for (const auto& [raw, weak] : impl_->pending) {
            (void)raw;
            if (auto op = weak.lock()) pending.push_back(std::move(op));
        }
    }
    for (const auto& op : pending) op->wait();
    if (::fsync(impl_->fd) != 0) {
        throw Error(ErrorCode::Io,
                    std::string("fsync failed: ") + std::strerror(errno), impl_->path);
    }
}

void NvmeBackend::cancel_all() {
    // POSIX path has no per-op cancellation; queued ops are dropped on
    // close. In-flight ops complete. Shutdown drains the queue.
    std::lock_guard lifecycle_lock(lifecycle_mu_);
    if (!opened_) return;
    std::lock_guard lock(impl_->mu);
    while (!impl_->queue.empty()) {
        auto ctx = std::move(impl_->queue.front());
        impl_->queue.pop_front();
        impl_->pending.erase(ctx->op.get());
        for (uint64_t i = 0; i < ctx->extent_count; ++i) {
            const std::size_t idx = static_cast<std::size_t>(ctx->first_extent + i);
            if (impl_->active_io[idx] != 0) --impl_->active_io[idx];
        }
        ctx->op->cancel();
    }
}

void NvmeBackend::close() {
    std::lock_guard lifecycle_lock(lifecycle_mu_);
    // close() is the authoritative fd-release path even when open() failed
    // midway: a failed open() leaves impl_->fd live, and any subsequent
    // open()/close()/destructor must release it or the store file stays
    // pinned on disk.
    if (!opened_ && impl_->fd < 0) return;
    {
        std::lock_guard lock(impl_->mu);
        impl_->stopping = true;
        while (!impl_->queue.empty()) {
            auto ctx = std::move(impl_->queue.front());
            impl_->queue.pop_front();
            impl_->pending.erase(ctx->op.get());
            for (uint64_t i = 0; i < ctx->extent_count; ++i) {
                const std::size_t idx = static_cast<std::size_t>(ctx->first_extent + i);
                if (impl_->active_io[idx] != 0) --impl_->active_io[idx];
            }
            ctx->op->cancel();
        }
    }
    impl_->cv.notify_all();
    for (auto& t : impl_->workers) {
        if (t.joinable()) t.join();
    }
    impl_->workers.clear();
    if (impl_->fd >= 0) {
        ::close(impl_->fd);
        impl_->fd = -1;
    }
    {
        std::lock_guard lock(impl_->mu);
        impl_->used.clear();
        impl_->active_io.clear();
        impl_->free_list.clear();
        impl_->pending.clear();
        impl_->used_extents = 0;
        impl_->stopping = false;
    }
    opened_ = false;
}

void NvmeBackend::destroy_file(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        if (errno == ENOENT) return;
        throw Error(ErrorCode::Io,
                    std::string("open store for deletion failed: ") +
                        std::strerror(errno),
                    path);
    }
    if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
        const int err = errno;
        ::close(fd);
        throw Error(ErrorCode::Io,
                    std::string("cannot delete an open store: ") +
                        std::strerror(err),
                    path);
    }
    if (::unlink(path.c_str()) != 0) {
        const int err = errno;
        ::close(fd);
        throw Error(ErrorCode::Io,
                    std::string("unlink store failed: ") + std::strerror(err), path);
    }
    ::close(fd);
}

#endif

}  // namespace flashtier
