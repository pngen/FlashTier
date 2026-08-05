#include "flashtier/backends/nvme_backend.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <optional>
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
#endif

namespace flashtier {

namespace {

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
            bytes_done_ = bytes;
            status_ = Status::Complete;
        }
        cv_.notify_all();
    }

    void fail(Error err) {
        {
            std::lock_guard lock(mu_);
            error_.emplace(std::move(err));
            status_ = Status::Failed;
        }
        cv_.notify_all();
    }

    void cancel() {
        {
            std::lock_guard lock(mu_);
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
    // Throws ErrorCode::Io when a packet cannot be posted. Safe to call
    // from close() (propagation possible) and from the destructor.
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
            // One sentinel per worker: every worker must be woken exactly
            // once, otherwise it stays blocked in GetQueuedCompletionStatus
            // and the joins below hang forever.
            const std::size_t n = local.size();
            for (std::size_t i = 0; i < n; ++i) {
                if (!PostQueuedCompletionStatus(iocp, 0, 0, nullptr)) {
                    throw Error(ErrorCode::Io,
                                windows_error(GetLastError(),
                                              "PostQueuedCompletionStatus"));
                }
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
    bool is_read = false;
};

IocpOverlapped* container_of(OVERLAPPED* ov) {
    return reinterpret_cast<IocpOverlapped*>(reinterpret_cast<char*>(ov) -
                                             offsetof(IocpOverlapped, ov));
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
// cancellation completions drain first, then the per-worker sentinel
// (null OVERLAPPED) exits the loop. A bounded wait acts as a backstop so
// a lost sentinel can never hang the join.
void iocp_worker(NvmeBackend::Impl* impl) {
    while (true) {
        DWORD bytes = 0;
        ULONG_PTR key = 0;
        OVERLAPPED* ov = nullptr;
        if (!GetQueuedCompletionStatus(impl->iocp, &bytes, &key, &ov, 200)) {
            const DWORD err = GetLastError();
            if (err == WAIT_TIMEOUT) {
                if (impl->stopping) break;  // backstop for lost sentinels
                continue;
            }
            if (ov != nullptr) {
                IocpOverlapped* io = container_of(ov);
                {
                    std::lock_guard lock(impl->mu);
                    impl->pending.erase(&io->ov);
                }
                if (err == ERROR_OPERATION_ABORTED) {
                    io->op->cancel();
                } else {
                    io->op->fail(Error(ErrorCode::Io,
                                       windows_error(err, "overlapped I/O")));
                }
                delete io;
                continue;
            }
            if (impl->stopping) break;
            continue;
        }
        if (ov == nullptr) {
            if (impl->stopping) break;  // shutdown sentinel
            continue;
        }
        IocpOverlapped* io = container_of(ov);
        {
            std::lock_guard lock(impl->mu);
            impl->pending.erase(&io->ov);
        }
        if (io->op->status() == AsyncOp::Status::Cancelled) {
            // Operation was cancelled; do not touch user buffers.
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
    close();  // release live handles from a previous (possibly failed) open
    if (capacity_bytes == 0 || page_size == 0 || capacity_bytes % page_size != 0) {
        throw Error(ErrorCode::InvalidArgument,
                    "store capacity must be a nonzero multiple of page size");
    }
    if (capacity_bytes / page_size < 1) {
        throw Error(ErrorCode::InvalidArgument, "store must hold at least one extent");
    }

    impl_->path = path;
    impl_->page_size = page_size;
    impl_->capacity_bytes = capacity_bytes;
    impl_->file_size = capacity_bytes + page_size;  // header page + payload
    impl_->extent_count = capacity_bytes / page_size;
    impl_->used.assign(impl_->extent_count, false);
    impl_->free_list.clear();
    for (uint64_t idx = 0; idx < impl_->extent_count; ++idx) {
        impl_->free_list.push_back(idx);  // every extent starts free
    }
    impl_->used_extents = 0;

    const bool exists = GetFileAttributesA(path.c_str()) != INVALID_FILE_ATTRIBUTES;

    impl_->file = CreateFileA(path.c_str(),
                              GENERIC_READ | GENERIC_WRITE,
                              FILE_SHARE_READ | FILE_SHARE_WRITE,
                              nullptr, OPEN_ALWAYS,
                              FILE_FLAG_OVERLAPPED,
                              nullptr);
    if (impl_->file == INVALID_HANDLE_VALUE) {
        throw Error(ErrorCode::Io, windows_error(GetLastError(), "CreateFile"));
    }

    // Preallocate the backing file (sparse is fine; capacity is reserved).
    LARGE_INTEGER size;
    size.QuadPart = static_cast<LONGLONG>(impl_->file_size);
    if (!SetFilePointerEx(impl_->file, size, nullptr, FILE_BEGIN) ||
        !SetEndOfFile(impl_->file)) {
        throw Error(ErrorCode::Io, windows_error(GetLastError(), "SetEndOfFile"));
    }

    // Read/validate or write the header.
    detail::StoreHeader h;
    detail::StoreHeader existing{};
    DWORD got = 0;
    const bool read_ok =
        sync_read(impl_->file, &existing, sizeof(existing), got) && got == sizeof(existing);

    if (exists && read_ok) {
        const std::string reason =
            detail::header_mismatch_reason(existing, page_size, capacity_bytes);
        if (!reason.empty()) {
            close();
            throw Error(ErrorCode::StoreCorrupt,
                        "existing store rejected: " + reason, path);
        }
        h = existing;
        // The free map is rebuilt from the runtime page table on each run;
        // the header carries identity, not extent usage.
    } else {
        h.magic = detail::kStoreMagic;
        h.version = detail::kStoreVersion;
        h.page_size = page_size;
        h.capacity_bytes = capacity_bytes;
        h.extent_count = impl_->extent_count;
        h.generation = 1;
        h.seed = 0xC0FFEE;
        DWORD written = 0;
        if (!sync_write(impl_->file, &h, sizeof(h), written) || written != sizeof(h)) {
            close();
            throw Error(ErrorCode::Io, windows_error(GetLastError(), "WriteFile(header)"));
        }
        if (!FlushFileBuffers(impl_->file)) {
            close();
            throw Error(ErrorCode::Io, windows_error(GetLastError(), "FlushFileBuffers"));
        }
    }

    impl_->iocp = CreateIoCompletionPort(impl_->file, nullptr, 0, 0);
    if (impl_->iocp == nullptr) {
        close();
        throw Error(ErrorCode::Io, windows_error(GetLastError(), "CreateIoCompletionPort"));
    }

    impl_->stopping = false;
    const unsigned worker_count = 4;
    for (unsigned i = 0; i < worker_count; ++i) {
        impl_->workers.emplace_back(iocp_worker, impl_.get());
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
        if (!opened_) throw Error(ErrorCode::State, "store not open");
    std::lock_guard lock(impl_->mu);
    if (impl_->free_list.empty()) return false;
    const uint64_t idx = impl_->free_list.back();
    impl_->free_list.pop_back();
    impl_->used[idx] = true;
    ++impl_->used_extents;
    offset_out = (idx + 1) * impl_->page_size;  // extent 0 starts after the header block
    return true;
}

void NvmeBackend::free_extent(uint64_t offset) {
    if (!opened_) throw Error(ErrorCode::State, "store not open");
    const uint64_t idx = offset / impl_->page_size - 1;
    if (offset % impl_->page_size != 0 || offset < impl_->page_size ||
        offset >= impl_->file_size || idx >= impl_->extent_count) {
        throw Error(ErrorCode::InvalidArgument, "invalid extent offset",
                    std::to_string(offset));
    }
    std::lock_guard lock(impl_->mu);
    if (!impl_->used[idx]) {
        throw Error(ErrorCode::State, "double free of store extent",
                    std::to_string(offset));
    }
    impl_->used[idx] = false;
    --impl_->used_extents;
    impl_->free_list.push_back(idx);
}

uint64_t NvmeBackend::extents_used() const noexcept {
    return opened_ ? impl_->used_extents.load() : 0;
}

AsyncOpPtr NvmeBackend::read_async(uint64_t offset, void* dst, std::size_t bytes) {
    if (!opened_) throw Error(ErrorCode::State, "store not open");
    if (bytes % impl_->page_size != 0) {
        throw Error(ErrorCode::InvalidArgument, "store I/O must be page-aligned in size");
    }
    if (offset % impl_->page_size != 0 || offset + bytes > impl_->file_size) {
        throw Error(ErrorCode::InvalidArgument, "store I/O out of range or misaligned",
                    "offset " + std::to_string(offset) + " bytes " + std::to_string(bytes));
    }

    auto op = std::make_shared<AsyncOpImpl>();
    auto* io = new IocpOverlapped();
    io->impl = impl_.get();
    io->op = op;
    io->dst = static_cast<uint8_t*>(dst);
    io->bytes = bytes;
    io->is_read = true;
    io->ov.Offset = static_cast<DWORD>(offset & 0xFFFFFFFFull);
    io->ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
    io->ov.hEvent = nullptr;  // completion goes to the IOCP

    {
        std::lock_guard lock(impl_->mu);
        impl_->pending[&io->ov] = op;
    }

    DWORD unused = 0;
    const BOOL ok = ReadFile(impl_->file, dst, static_cast<DWORD>(bytes), &unused, &io->ov);
    if (!ok) {
        const DWORD err = GetLastError();
        if (err != ERROR_IO_PENDING) {
            std::lock_guard lock(impl_->mu);
            impl_->pending.erase(&io->ov);
            delete io;
            throw Error(ErrorCode::Io, windows_error(err, "ReadFile"));
        }
    }
    return op;
}

AsyncOpPtr NvmeBackend::write_async(uint64_t offset, const void* src, std::size_t bytes) {
    if (!opened_) throw Error(ErrorCode::State, "store not open");
    if (bytes % impl_->page_size != 0) {
        throw Error(ErrorCode::InvalidArgument, "store I/O must be page-aligned in size");
    }
    if (offset % impl_->page_size != 0 || offset + bytes > impl_->file_size) {
        throw Error(ErrorCode::InvalidArgument, "store I/O out of range or misaligned");
    }

    auto op = std::make_shared<AsyncOpImpl>();
    auto* io = new IocpOverlapped();
    io->impl = impl_.get();
    io->op = op;
    io->src = static_cast<const uint8_t*>(src);
    io->bytes = bytes;
    io->is_read = false;
    io->ov.Offset = static_cast<DWORD>(offset & 0xFFFFFFFFull);
    io->ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
    io->ov.hEvent = nullptr;

    {
        std::lock_guard lock(impl_->mu);
        impl_->pending[&io->ov] = op;
    }

    DWORD unused = 0;
    const BOOL ok = WriteFile(impl_->file, src, static_cast<DWORD>(bytes), &unused, &io->ov);
    if (!ok) {
        const DWORD err = GetLastError();
        if (err != ERROR_IO_PENDING) {
            std::lock_guard lock(impl_->mu);
            impl_->pending.erase(&io->ov);
            delete io;
            throw Error(ErrorCode::Io, windows_error(err, "WriteFile"));
        }
    }
    return op;
}

void NvmeBackend::flush() {
    if (!opened_) return;
    if (!FlushFileBuffers(impl_->file)) {
        throw Error(ErrorCode::Io, windows_error(GetLastError(), "FlushFileBuffers"));
    }
}

void NvmeBackend::cancel_all() {
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
    // close() is the authoritative handle-release path even when open()
    // failed midway: a failed open() leaves impl_->file live, and any
    // subsequent open()/close()/destructor must release it or the file
    // stays pinned on disk (silently breaking DeleteFileA).
    if (!opened_ && impl_->file == INVALID_HANDLE_VALUE && impl_->iocp == nullptr) {
        return;
    }
    impl_->stop_workers();
    {
        std::lock_guard lock(impl_->mu);
        impl_->used.clear();
        impl_->free_list.clear();
        impl_->used_extents = 0;
        impl_->stopping = false;  // allow reuse of the Impl by a future open()
    }
    opened_ = false;
}

void NvmeBackend::destroy_file(const std::string& path) {
    DeleteFileA(path.c_str());
}

#else  // POSIX (Linux)

// Simple threaded pread/pwrite backend with the same AsyncOp interface.
// Synchronous kernel calls dispatched on a bounded worker pool.

#include <deque>

namespace {

struct PosixImpl;

struct PosixOpContext {
    PosixImpl* impl = nullptr;
    std::shared_ptr<AsyncOpImpl> op;
    int fd = -1;
    uint64_t offset = 0;
    void* buf = nullptr;
    std::size_t bytes = 0;
    bool is_read = false;
};

struct PosixImpl {
    int fd = -1;
    uint64_t page_size = 0;
    uint64_t capacity_bytes = 0;
    uint64_t file_size = 0;  // capacity_bytes + one header page
    uint64_t extent_count = 0;
    std::atomic<uint64_t> used_extents{0};
    std::mutex mu;
    std::vector<bool> used;
    std::vector<uint64_t> free_list;
    std::vector<std::thread> workers;
    std::deque<std::shared_ptr<PosixOpContext>> queue;
    std::condition_variable cv;
    bool stopping = false;
    std::string path;

    ~PosixImpl() {
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

    void worker_loop() {
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

    void run(const std::shared_ptr<PosixOpContext>& ctx) {
        ssize_t done = 0;
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
                ctx->op->fail(Error(ErrorCode::Io,
                                    std::string(ctx->is_read ? "pread" : "pwrite") +
                                        " failed: " + std::strerror(errno)));
                return;
            }
            if (r == 0) {
                ctx->op->fail(Error(ErrorCode::Io, "unexpected EOF in store read"));
                return;
            }
            total += static_cast<std::size_t>(r);
        }
        ctx->op->complete(total);
    }
};

}  // namespace

NvmeBackend::NvmeBackend() : impl_(std::make_unique<PosixImpl>()) {}

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
    close();  // release any live fd from a previous (possibly failed) open
    if (capacity_bytes == 0 || page_size == 0 || capacity_bytes % page_size != 0) {
        throw Error(ErrorCode::InvalidArgument,
                    "store capacity must be a nonzero multiple of page size");
    }
    impl_->path = path;
    impl_->page_size = page_size;
    impl_->capacity_bytes = capacity_bytes;
    impl_->file_size = capacity_bytes + page_size;  // header page + payload
    impl_->extent_count = capacity_bytes / page_size;
    impl_->used.assign(impl_->extent_count, false);
    impl_->free_list.clear();
    for (uint64_t idx = 0; idx < impl_->extent_count; ++idx) {
        impl_->free_list.push_back(idx);  // every extent starts free
    }
    impl_->used_extents = 0;

    const bool exists = ::access(path.c_str(), F_OK) == 0;
    impl_->fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (impl_->fd < 0) {
        throw Error(ErrorCode::Io,
                    std::string("open store failed: ") + std::strerror(errno), path);
    }

    if (::ftruncate(impl_->fd, static_cast<off_t>(impl_->file_size)) != 0) {
        throw Error(ErrorCode::Io,
                    std::string("ftruncate failed: ") + std::strerror(errno), path);
    }

    detail::StoreHeader h;
    if (exists) {
        ssize_t got = ::pread(impl_->fd, &h, sizeof(h), 0);
        if (got < 0) {
            throw Error(ErrorCode::Io,
                        std::string("pread header failed: ") + std::strerror(errno), path);
        }
        if (got == static_cast<ssize_t>(sizeof(h))) {
            const std::string reason =
                detail::header_mismatch_reason(h, page_size, capacity_bytes);
            if (!reason.empty()) {
                close();
                throw Error(ErrorCode::StoreCorrupt,
                            "existing store rejected: " + reason, path);
            }
        } else {
            close();
            throw Error(ErrorCode::StoreCorrupt,
                        "existing store has truncated header", path);
        }
    } else {
        h.magic = detail::kStoreMagic;
        h.version = detail::kStoreVersion;
        h.page_size = page_size;
        h.capacity_bytes = capacity_bytes;
        h.extent_count = impl_->extent_count;
        h.generation = 1;
        h.seed = 0xC0FFEE;
        ssize_t w = ::pwrite(impl_->fd, &h, sizeof(h), 0);
        if (w != static_cast<ssize_t>(sizeof(h))) {
            throw Error(ErrorCode::Io,
                        std::string("pwrite header failed: ") + std::strerror(errno), path);
        }
        ::fsync(impl_->fd);
    }

    {
        std::lock_guard lock(impl_->mu);
        impl_->stopping = false;
        const unsigned worker_count = 4;
        for (unsigned i = 0; i < worker_count; ++i) {
            impl_->workers.emplace_back([this] { impl_->worker_loop(); });
        }
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
        if (!opened_) throw Error(ErrorCode::State, "store not open");
    std::lock_guard lock(impl_->mu);
    if (impl_->free_list.empty()) return false;
    const uint64_t idx = impl_->free_list.back();
    impl_->free_list.pop_back();
    impl_->used[idx] = true;
    ++impl_->used_extents;
    offset_out = (idx + 1) * impl_->page_size;
    return true;
}

void NvmeBackend::free_extent(uint64_t offset) {
    if (!opened_) throw Error(ErrorCode::State, "store not open");
    const uint64_t idx = offset / impl_->page_size - 1;
    if (offset % impl_->page_size != 0 || offset < impl_->page_size ||
        offset >= impl_->file_size || idx >= impl_->extent_count) {
        throw Error(ErrorCode::InvalidArgument, "invalid extent offset",
                    std::to_string(offset));
    }
    std::lock_guard lock(impl_->mu);
    if (!impl_->used[idx]) {
        throw Error(ErrorCode::State, "double free of store extent",
                    std::to_string(offset));
    }
    impl_->used[idx] = false;
    --impl_->used_extents;
    impl_->free_list.push_back(idx);
}

uint64_t NvmeBackend::extents_used() const noexcept {
    return opened_ ? impl_->used_extents.load() : 0;
}

AsyncOpPtr NvmeBackend::read_async(uint64_t offset, void* dst, std::size_t bytes) {
    if (!opened_) throw Error(ErrorCode::State, "store not open");
    if (bytes % impl_->page_size != 0 || offset % impl_->page_size != 0 ||
        offset + bytes > impl_->file_size) {
        throw Error(ErrorCode::InvalidArgument, "store I/O out of range or misaligned");
    }
    auto ctx = std::make_shared<PosixOpContext>();
    ctx->impl = impl_.get();
    ctx->op = std::make_shared<AsyncOpImpl>();
    ctx->fd = impl_->fd;
    ctx->offset = offset;
    ctx->buf = dst;
    ctx->bytes = bytes;
    ctx->is_read = true;
    {
        std::lock_guard lock(impl_->mu);
        impl_->queue.push_back(ctx);
    }
    impl_->cv.notify_one();
    return ctx->op;
}

AsyncOpPtr NvmeBackend::write_async(uint64_t offset, const void* src, std::size_t bytes) {
    if (!opened_) throw Error(ErrorCode::State, "store not open");
    if (bytes % impl_->page_size != 0 || offset % impl_->page_size != 0 ||
        offset + bytes > impl_->file_size) {
        throw Error(ErrorCode::InvalidArgument, "store I/O out of range or misaligned");
    }
    auto ctx = std::make_shared<PosixOpContext>();
    ctx->impl = impl_.get();
    ctx->op = std::make_shared<AsyncOpImpl>();
    ctx->fd = impl_->fd;
    ctx->offset = offset;
    ctx->buf = const_cast<void*>(src);
    ctx->bytes = bytes;
    ctx->is_read = false;
    {
        std::lock_guard lock(impl_->mu);
        impl_->queue.push_back(ctx);
    }
    impl_->cv.notify_one();
    return ctx->op;
}

void NvmeBackend::flush() {
    if (!opened_) return;
    if (::fsync(impl_->fd) != 0) {
        throw Error(ErrorCode::Io,
                    std::string("fsync failed: ") + std::strerror(errno), impl_->path);
    }
}

void NvmeBackend::cancel_all() {
    // POSIX path has no per-op cancellation; queued ops are dropped on
    // close. In-flight ops complete. Shutdown drains the queue.
    if (!opened_) return;
    std::lock_guard lock(impl_->mu);
    while (!impl_->queue.empty()) {
        auto ctx = std::move(impl_->queue.front());
        impl_->queue.pop_front();
        ctx->op->cancel();
    }
}

void NvmeBackend::close() {
    // close() is the authoritative fd-release path even when open() failed
    // midway: a failed open() leaves impl_->fd live, and any subsequent
    // open()/close()/destructor must release it or the store file stays
    // pinned on disk.
    if (!opened_ && impl_->fd < 0) return;
    cancel_all();
    {
        std::lock_guard lock(impl_->mu);
        impl_->stopping = true;
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
        impl_->free_list.clear();
        impl_->used_extents = 0;
        impl_->stopping = false;
    }
    opened_ = false;
}

void NvmeBackend::destroy_file(const std::string& path) {
    ::unlink(path.c_str());
}

#endif

}  // namespace flashtier
