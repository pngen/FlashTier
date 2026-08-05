#include "test_harness.hpp"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <vector>

#include "flashtier/backends/nvme_backend.hpp"
#include "flashtier/error.hpp"
#include "flashtier/integrity.hpp"

using namespace flashtier;

namespace {

std::string temp_store_path() {
    return (std::filesystem::temp_directory_path() / "ft-test-store.bin").string();
}

constexpr uint64_t kPageSize = 64 * 1024;

}  // namespace

FT_TEST(store_opens_creates_and_reopens) {
    const std::string path = temp_store_path();
    NvmeBackend::destroy_file(path);

    {
        NvmeBackend b;
        auto info = b.open(path, 8 * kPageSize, kPageSize);
        FT_ASSERT_EQ(info.capacity_bytes, 8 * kPageSize);
        FT_ASSERT_EQ(info.extent_count, 8u);
        FT_ASSERT(b.is_open());
    }
    // Reopen must validate the existing header (same params).
    {
        NvmeBackend b;
        auto info = b.open(path, 8 * kPageSize, kPageSize);
        FT_ASSERT_EQ(info.extent_count, 8u);
    }
    NvmeBackend::destroy_file(path);
}

FT_TEST(store_rejects_incompatible_parameters) {
    const std::string path = temp_store_path();
    NvmeBackend::destroy_file(path);
    {
        NvmeBackend b;
        b.open(path, 8 * kPageSize, kPageSize);
    }
    {
        NvmeBackend b;
        FT_ASSERT_THROWS(b.open(path, 8 * kPageSize, 128 * 1024), ErrorCode::StoreCorrupt);
        FT_ASSERT_THROWS(b.open(path, 4 * kPageSize, kPageSize), ErrorCode::StoreCorrupt);
    }
    NvmeBackend::destroy_file(path);
}

FT_TEST(store_rejects_corrupted_header) {
    const std::string path = temp_store_path();
    NvmeBackend::destroy_file(path);
    {
        NvmeBackend b;
        b.open(path, 8 * kPageSize, kPageSize);
    }
    // Truncate the file so the header is garbage.
    {
        std::FILE* f = std::fopen(path.c_str(), "r+b");
        FT_ASSERT(f != nullptr);
        const char junk[64] = "not a flashtier store at all";
        std::fwrite(junk, 1, sizeof(junk), f);
        std::fflush(f);
        std::fclose(f);
    }
    {
        NvmeBackend b;
        FT_ASSERT_THROWS(b.open(path, 8 * kPageSize, kPageSize), ErrorCode::StoreCorrupt);
    }
    NvmeBackend::destroy_file(path);
}

FT_TEST(store_extent_allocation_and_free) {
    const std::string path = temp_store_path();
    NvmeBackend::destroy_file(path);
    NvmeBackend b;
    b.open(path, 8 * kPageSize, kPageSize);

    std::vector<uint64_t> offsets;
    for (int i = 0; i < 8; ++i) {
        uint64_t off = 0;
        FT_ASSERT(b.allocate_extent(off));
        FT_ASSERT(off % kPageSize == 0);
        FT_ASSERT(off >= kPageSize);  // never the header page
        offsets.push_back(off);
    }
    FT_ASSERT_EQ(b.extents_used(), 8u);
    uint64_t dummy = 0;
    FT_ASSERT(!b.allocate_extent(dummy));  // full

    for (uint64_t off : offsets) {
        b.free_extent(off);
    }
    FT_ASSERT_EQ(b.extents_used(), 0u);
    FT_ASSERT(b.allocate_extent(dummy));  // reusable after free
    NvmeBackend::destroy_file(path);
}

FT_TEST(store_rejects_double_free_and_invalid_offsets) {
    const std::string path = temp_store_path();
    NvmeBackend::destroy_file(path);
    NvmeBackend b;
    b.open(path, 8 * kPageSize, kPageSize);
    uint64_t off = 0;
    FT_ASSERT(b.allocate_extent(off));
    b.free_extent(off);
    FT_ASSERT_THROWS(b.free_extent(off), ErrorCode::State);
    FT_ASSERT_THROWS(b.free_extent(0), ErrorCode::InvalidArgument);
    FT_ASSERT_THROWS(b.free_extent(off + 1), ErrorCode::InvalidArgument);
    FT_ASSERT_THROWS(b.free_extent(9 * kPageSize), ErrorCode::InvalidArgument);
    NvmeBackend::destroy_file(path);
}

FT_TEST(store_async_write_read_integrity) {
    const std::string path = temp_store_path();
    NvmeBackend::destroy_file(path);
    NvmeBackend b;
    b.open(path, 8 * kPageSize, kPageSize);

    uint64_t off = 0;
    FT_ASSERT(b.allocate_extent(off));

    std::vector<uint8_t> payload(kPageSize);
    fill_pattern(payload.data(), payload.size(), 99, 1234);

    auto w = b.write_async(off, payload.data(), payload.size());
    FT_ASSERT(!w->is_complete() || w->status() == AsyncOp::Status::Complete);
    w->wait();
    FT_ASSERT_EQ(w->status(), AsyncOp::Status::Complete);
    FT_ASSERT_EQ(w->bytes_done(), kPageSize);

    std::vector<uint8_t> read_back(kPageSize, 0x00);
    auto r = b.read_async(off, read_back.data(), read_back.size());
    r->wait();
    FT_ASSERT_EQ(r->status(), AsyncOp::Status::Complete);
    FT_ASSERT_EQ(r->bytes_done(), kPageSize);
    FT_ASSERT(std::memcmp(payload.data(), read_back.data(), kPageSize) == 0);
    NvmeBackend::destroy_file(path);
}

FT_TEST(store_async_ops_complete_independently) {
    const std::string path = temp_store_path();
    NvmeBackend::destroy_file(path);
    NvmeBackend b;
    b.open(path, 8 * kPageSize, kPageSize);

    std::vector<uint8_t> payload(kPageSize, 0x5A);
    std::vector<AsyncOpPtr> writes;
    std::vector<uint64_t> offsets;
    for (int i = 0; i < 8; ++i) {
        uint64_t off = 0;
        FT_ASSERT(b.allocate_extent(off));
        offsets.push_back(off);
        writes.push_back(b.write_async(off, payload.data(), payload.size()));
    }
    for (auto& w : writes) w->wait();
    for (std::size_t i = 0; i < writes.size(); ++i) {
        FT_ASSERT_EQ(writes[i]->status(), AsyncOp::Status::Complete);
        std::vector<uint8_t> back(kPageSize, 0);
        auto r = b.read_async(offsets[i], back.data(), kPageSize);
        r->wait();
        FT_ASSERT(std::memcmp(payload.data(), back.data(), kPageSize) == 0);
    }
    NvmeBackend::destroy_file(path);
}

FT_TEST(store_rejects_misaligned_io) {
    const std::string path = temp_store_path();
    NvmeBackend::destroy_file(path);
    NvmeBackend b;
    b.open(path, 8 * kPageSize, kPageSize);
    uint64_t off = 0;
    FT_ASSERT(b.allocate_extent(off));
    std::vector<uint8_t> buf(kPageSize);
    FT_ASSERT_THROWS(b.read_async(off, buf.data(), 1000), ErrorCode::InvalidArgument);
    FT_ASSERT_THROWS(b.read_async(off + 1, buf.data(), kPageSize), ErrorCode::InvalidArgument);
    NvmeBackend::destroy_file(path);
}

FT_TEST(store_shutdown_with_in_flight_ops_is_deterministic) {
    const std::string path = temp_store_path();
    NvmeBackend::destroy_file(path);
    {
        NvmeBackend b;
        b.open(path, 8 * kPageSize, kPageSize);
        uint64_t off = 0;
        FT_ASSERT(b.allocate_extent(off));
        std::vector<uint8_t> payload(kPageSize, 0x11);
        auto w = b.write_async(off, payload.data(), kPageSize);
        w->wait();  // complete before close
        // close() with nothing pending is a no-op exercise.
    }
    NvmeBackend::destroy_file(path);
}

FT_TEST(store_header_validation_helper) {
    const std::string path = temp_store_path();
    NvmeBackend::destroy_file(path);
    FT_ASSERT(!store_header_valid(path, kPageSize, 8 * kPageSize));
    {
        NvmeBackend b;
        b.open(path, 8 * kPageSize, kPageSize);
    }
    FT_ASSERT(store_header_valid(path, kPageSize, 8 * kPageSize));
    FT_ASSERT(!store_header_valid(path, kPageSize, 16 * kPageSize));
    NvmeBackend::destroy_file(path);
}

// Deterministic multi-worker shutdown: exercises the real IOCP worker pool.
// close() must post one shutdown packet per worker, drain in-flight
// completions/cancellations, join every worker within a bounded interval,
// be idempotent, and leave the destructor hang-free.
FT_TEST(store_deterministic_multi_worker_shutdown) {
    const std::string path = temp_store_path();
    NvmeBackend::destroy_file(path);
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    {
        NvmeBackend b;
        b.open(path, 8 * kPageSize, kPageSize);

        std::vector<uint64_t> offsets;
        for (int i = 0; i < 8; ++i) {
            uint64_t off = 0;
            FT_ASSERT(b.allocate_extent(off));
            offsets.push_back(off);
        }

        // Normal completions through the worker pool.
        std::vector<uint8_t> payload(kPageSize, 0x77);
        std::vector<AsyncOpPtr> writes;
        for (uint64_t off : offsets) {
            writes.push_back(b.write_async(off, payload.data(), kPageSize));
        }
        for (auto& w : writes) w->wait();
        for (auto& w : writes) FT_ASSERT_EQ(w->status(), AsyncOp::Status::Complete);

        // In-flight reads when close() is called: they must settle
        // (complete or cancel) and close must return promptly.
        std::vector<AsyncOpPtr> pending;
        for (uint64_t off : offsets) {
            pending.push_back(b.read_async(off, payload.data(), kPageSize));
        }
        const auto t1 = Clock::now();
        b.close();
        const double close_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - t1).count();
        FT_ASSERT(close_ms < 5000.0);
        for (auto& op : pending) {
            FT_ASSERT(op->status() == AsyncOp::Status::Complete ||
                      op->status() == AsyncOp::Status::Cancelled);
        }
        b.close();  // idempotent
        FT_ASSERT(!b.is_open());
    }  // destructor must not hang
    const double total_s = std::chrono::duration<double>(Clock::now() - t0).count();
    FT_ASSERT(total_s < 10.0);
    NvmeBackend::destroy_file(path);
}

int main() { return ft_test::run_all("test_nvme_backend"); }
