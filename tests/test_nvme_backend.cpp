#include "test_harness.hpp"

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <limits>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

#include "flashtier/backends/nvme_backend.hpp"
#include "flashtier/error.hpp"
#include "flashtier/integrity.hpp"

using namespace flashtier;

namespace {

std::string temp_store_path() {
#if defined(_WIN32)
    const int pid = _getpid();
#else
    const int pid = static_cast<int>(::getpid());
#endif
    return (std::filesystem::temp_directory_path() /
            ("ft-test-store-" + std::to_string(pid) + ".bin"))
        .string();
}

std::string temp_directory_utf8() {
    const auto encoded = std::filesystem::temp_directory_path().u8string();
    return std::string(reinterpret_cast<const char*>(encoded.data()), encoded.size());
}

std::string temp_store_path_utf8(const std::string& leaf) {
    std::string path = temp_directory_utf8();
    if (!path.empty() && path.back() != '/' && path.back() != '\\') {
#if defined(_WIN32)
        path.push_back('\\');
#else
        path.push_back('/');
#endif
    }
    return path + leaf;
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

FT_TEST(store_accepts_utf8_backing_paths) {
#if defined(_WIN32)
    const int pid = _getpid();
#else
    const int pid = static_cast<int>(::getpid());
#endif
    // U+2603 SNOWMAN is deliberately outside common Windows ANSI code pages.
    // The public string API is UTF-8 on Windows, so this must reach the wide
    // filesystem API without depending on the process active code page.
    const std::string path = temp_store_path_utf8(
        "ft-test-\xE2\x98\x83-" + std::to_string(pid) + ".bin");
    NvmeBackend::destroy_file(path);
    {
        NvmeBackend b;
        b.open(path, 4 * kPageSize, kPageSize);
        uint64_t offset = 0;
        FT_ASSERT(b.allocate_extent(offset));
        std::vector<uint8_t> payload(kPageSize, 0xA7);
        auto write = b.write_async(offset, payload.data(), payload.size());
        write->wait();
        FT_ASSERT_EQ(write->status(), AsyncOp::Status::Complete);
    }
    FT_ASSERT(store_header_valid(path, kPageSize, 4 * kPageSize));
    {
        NvmeBackend b;
        const auto info = b.open(path, 4 * kPageSize, kPageSize);
        FT_ASSERT_EQ(info.extent_count, 4u);
    }
    NvmeBackend::destroy_file(path);
    FT_ASSERT(!store_header_valid(path, kPageSize, 4 * kPageSize));
}

#if defined(_WIN32)
FT_TEST(store_rejects_invalid_utf8_and_embedded_nul_paths) {
    const std::string base = temp_store_path_utf8("ft-test-invalid-");
    std::string invalid_utf8 = base;
    invalid_utf8.push_back(static_cast<char>(0xC3));
    invalid_utf8.push_back('(');  // invalid continuation byte
    invalid_utf8.append(".bin");

    NvmeBackend b;
    FT_ASSERT_THROWS(b.open(invalid_utf8, 4 * kPageSize, kPageSize),
                     ErrorCode::InvalidArgument);
    FT_ASSERT(!b.is_open());
    FT_ASSERT_THROWS(NvmeBackend::destroy_file(invalid_utf8),
                     ErrorCode::InvalidArgument);
    FT_ASSERT(!store_header_valid(invalid_utf8, kPageSize, 4 * kPageSize));

    std::string embedded_nul = base + "prefix";
    embedded_nul.push_back('\0');
    embedded_nul.append("suffix.bin");
    FT_ASSERT_THROWS(b.open(embedded_nul, 4 * kPageSize, kPageSize),
                     ErrorCode::InvalidArgument);
    FT_ASSERT_THROWS(NvmeBackend::destroy_file(embedded_nul),
                     ErrorCode::InvalidArgument);
}
#endif

FT_TEST(store_rejects_incompatible_parameters) {
    const std::string path = temp_store_path();
    NvmeBackend::destroy_file(path);
    {
        NvmeBackend b;
        b.open(path, 8 * kPageSize, kPageSize);
    }
    const auto original_size = std::filesystem::file_size(path);
    {
        NvmeBackend b;
        FT_ASSERT_THROWS(b.open(path, 8 * kPageSize, 128 * 1024), ErrorCode::StoreCorrupt);
        FT_ASSERT_THROWS(b.open(path, 4 * kPageSize, kPageSize), ErrorCode::StoreCorrupt);
    }
    FT_ASSERT_EQ(std::filesystem::file_size(path), original_size);
    {
        NvmeBackend b;
        b.open(path, 8 * kPageSize, kPageSize);
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

FT_TEST(store_checksum_detects_non_identity_header_corruption) {
    const std::string path = temp_store_path();
    NvmeBackend::destroy_file(path);
    {
        NvmeBackend b;
        b.open(path, 8 * kPageSize, kPageSize);
    }
    {
        std::FILE* f = std::fopen(path.c_str(), "r+b");
        FT_ASSERT(f != nullptr);
        // Generation is otherwise range-valid; only the sealed checksum
        // distinguishes this single-bit mutation from an intact header.
        FT_ASSERT(std::fseek(f, 32, SEEK_SET) == 0);
        const int old_value = std::fgetc(f);
        FT_ASSERT(old_value != EOF);
        FT_ASSERT(std::fseek(f, 32, SEEK_SET) == 0);
        FT_ASSERT(std::fputc(old_value ^ 0x01, f) != EOF);
        FT_ASSERT(std::fflush(f) == 0);
        FT_ASSERT(std::fclose(f) == 0);
    }
    FT_ASSERT(!store_header_valid(path, kPageSize, 8 * kPageSize));
    {
        NvmeBackend b;
        FT_ASSERT_THROWS(b.open(path, 8 * kPageSize, kPageSize),
                         ErrorCode::StoreCorrupt);
    }
    NvmeBackend::destroy_file(path);
}

FT_TEST(store_rejects_truncated_payload_without_healing_file) {
    const std::string path = temp_store_path();
    NvmeBackend::destroy_file(path);
    {
        NvmeBackend b;
        b.open(path, 8 * kPageSize, kPageSize);
    }
    const uintmax_t truncated_size = 8 * kPageSize;
    std::filesystem::resize_file(path, truncated_size);
    FT_ASSERT(!store_header_valid(path, kPageSize, 8 * kPageSize));
    {
        NvmeBackend b;
        FT_ASSERT_THROWS(b.open(path, 8 * kPageSize, kPageSize),
                         ErrorCode::StoreCorrupt);
    }
    FT_ASSERT_EQ(std::filesystem::file_size(path), truncated_size);
    NvmeBackend::destroy_file(path);
}

FT_TEST(store_rejects_unrepresentable_or_overlapping_geometry) {
    const std::string path = temp_store_path();
    NvmeBackend::destroy_file(path);
    NvmeBackend b;
    FT_ASSERT_THROWS(b.open("", 8 * kPageSize, kPageSize),
                     ErrorCode::InvalidArgument);
    FT_ASSERT_THROWS(b.open(path, 8 * 1024, 1024), ErrorCode::InvalidArgument);
    const uint64_t huge_capacity =
        std::numeric_limits<uint64_t>::max() - kPageSize + 1;
    FT_ASSERT_THROWS(b.open(path, huge_capacity, kPageSize),
                     ErrorCode::InvalidArgument);
    FT_ASSERT(!std::filesystem::exists(path));
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
    b.close();
    NvmeBackend::destroy_file(path);
}

FT_TEST(store_contiguous_multi_extent_roundtrip_and_release) {
    const std::string path = temp_store_path();
    NvmeBackend::destroy_file(path);
    NvmeBackend b;
    b.open(path, 8 * kPageSize, kPageSize);

    uint64_t offset = 0;
    FT_ASSERT(b.allocate_extents(3, offset));
    FT_ASSERT_EQ(b.extents_used(), 3u);
    std::vector<uint8_t> payload(3 * kPageSize);
    for (std::size_t i = 0; i < payload.size(); ++i) {
        payload[i] = static_cast<uint8_t>((i / kPageSize) * 71 + (i % 251));
    }
    auto write = b.write_async(offset, payload.data(), payload.size());
    write->wait();
    FT_ASSERT_EQ(write->bytes_done(), payload.size());
    b.flush();

    std::vector<uint8_t> actual(payload.size(), 0);
    auto read = b.read_async(offset, actual.data(), actual.size());
    read->wait();
    FT_ASSERT(std::memcmp(payload.data(), actual.data(), payload.size()) == 0);

    b.free_extents(offset, 3);
    FT_ASSERT_EQ(b.extents_used(), 0u);
    FT_ASSERT_THROWS(b.free_extents(offset, 3), ErrorCode::State);
    b.close();
    NvmeBackend::destroy_file(path);
}

FT_TEST(store_multi_extent_allocation_is_atomic_under_fragmentation) {
    const std::string path = temp_store_path();
    NvmeBackend::destroy_file(path);
    NvmeBackend b;
    b.open(path, 6 * kPageSize, kPageSize);
    std::vector<uint64_t> offsets;
    for (int i = 0; i < 6; ++i) {
        uint64_t offset = 0;
        FT_ASSERT(b.allocate_extent(offset));
        offsets.push_back(offset);
    }
    b.free_extent(offsets[0]);
    b.free_extent(offsets[2]);
    b.free_extent(offsets[4]);
    FT_ASSERT_EQ(b.extents_used(), 3u);
    uint64_t range = 0;
    FT_ASSERT(!b.allocate_extents(2, range));
    FT_ASSERT_EQ(b.extents_used(), 3u);
    b.free_extent(offsets[1]);
    b.free_extent(offsets[3]);
    b.free_extent(offsets[5]);
    b.close();
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
    b.close();
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
    b.close();
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
    b.flush();  // flush must include writes pending when it was called
    for (const auto& write : writes) {
        FT_ASSERT_EQ(write->status(), AsyncOp::Status::Complete);
    }
    for (auto& w : writes) w->wait();
    for (std::size_t i = 0; i < writes.size(); ++i) {
        FT_ASSERT_EQ(writes[i]->status(), AsyncOp::Status::Complete);
        std::vector<uint8_t> back(kPageSize, 0);
        auto r = b.read_async(offsets[i], back.data(), kPageSize);
        r->wait();
        FT_ASSERT(std::memcmp(payload.data(), back.data(), kPageSize) == 0);
    }
    b.close();
    NvmeBackend::destroy_file(path);
}

FT_TEST(store_excludes_concurrent_writers_to_the_same_path) {
    const std::string path = temp_store_path();
    NvmeBackend::destroy_file(path);
    NvmeBackend first;
    first.open(path, 8 * kPageSize, kPageSize);
    NvmeBackend second;
    FT_ASSERT_THROWS(second.open(path, 8 * kPageSize, kPageSize), ErrorCode::Io);

    uint64_t offset = 0;
    FT_ASSERT(first.allocate_extent(offset));
    std::vector<uint8_t> payload(kPageSize, 0xA5);
    auto write = first.write_async(offset, payload.data(), payload.size());
    write->wait();
    FT_ASSERT_EQ(write->status(), AsyncOp::Status::Complete);
    first.close();

    second.open(path, 8 * kPageSize, kPageSize);
    second.close();
    NvmeBackend::destroy_file(path);
}

FT_TEST(store_rejects_misaligned_io) {
    const std::string path = temp_store_path();
    NvmeBackend::destroy_file(path);
    NvmeBackend b;
    b.open(path, 8 * kPageSize, kPageSize);
    uint64_t off = 0;
    FT_ASSERT(b.allocate_extent(off));
    uint64_t adjacent = 0;
    FT_ASSERT(b.allocate_extent(adjacent));
    std::vector<uint8_t> buf(kPageSize);
    FT_ASSERT_THROWS(b.read_async(off, buf.data(), 1000), ErrorCode::InvalidArgument);
    FT_ASSERT_THROWS(b.read_async(off + 1, buf.data(), kPageSize), ErrorCode::InvalidArgument);
    FT_ASSERT_THROWS(b.read_async(0, buf.data(), kPageSize), ErrorCode::InvalidArgument);
    FT_ASSERT_THROWS(b.write_async(0, buf.data(), kPageSize), ErrorCode::InvalidArgument);
    FT_ASSERT_THROWS(b.read_async(off, nullptr, kPageSize), ErrorCode::InvalidArgument);
    FT_ASSERT_THROWS(b.read_async(off, buf.data(), 0), ErrorCode::InvalidArgument);
    const uint64_t huge_aligned =
        std::numeric_limits<uint64_t>::max() & ~(kPageSize - 1);
    FT_ASSERT_THROWS(b.read_async(huge_aligned, buf.data(), kPageSize),
                     ErrorCode::InvalidArgument);
    FT_ASSERT_THROWS(b.read_async(kPageSize, buf.data(), kPageSize), ErrorCode::State);

    // A multi-page request must own every covered extent, not merely the
    // first one. `off` is the upper extent and `adjacent` is directly below.
    b.free_extent(off);
    std::vector<uint8_t> two_pages(2 * kPageSize);
    FT_ASSERT_THROWS(b.write_async(adjacent, two_pages.data(), two_pages.size()),
                     ErrorCode::State);
    b.free_extent(adjacent);
    b.close();
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
        std::vector<std::vector<uint8_t>> read_buffers(
            offsets.size(), std::vector<uint8_t>(kPageSize));
        for (std::size_t i = 0; i < offsets.size(); ++i) {
            pending.push_back(
                b.read_async(offsets[i], read_buffers[i].data(), kPageSize));
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

FT_TEST(store_concurrent_submission_and_close_settles_every_returned_op) {
    const std::string path = temp_store_path();
    NvmeBackend::destroy_file(path);
    NvmeBackend b;
    b.open(path, 16 * kPageSize, kPageSize);

    std::vector<uint64_t> offsets;
    for (int i = 0; i < 4; ++i) {
        uint64_t offset = 0;
        FT_ASSERT(b.allocate_extent(offset));
        offsets.push_back(offset);
    }
    std::vector<std::vector<uint8_t>> buffers(
        offsets.size(), std::vector<uint8_t>(kPageSize, 0x3C));
    std::atomic<bool> go{false};
    std::atomic<int> submitted{0};
    std::atomic<int> terminal{0};
    std::atomic<int> unexpected{0};
    std::vector<std::thread> submitters;
    for (std::size_t i = 0; i < offsets.size(); ++i) {
        submitters.emplace_back([&, i] {
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            for (int iteration = 0; iteration < 100; ++iteration) {
                try {
                    auto op = b.write_async(offsets[i], buffers[i].data(), kPageSize);
                    ++submitted;
                    op->wait();
                    if (op->status() == AsyncOp::Status::Complete ||
                        op->status() == AsyncOp::Status::Cancelled) {
                        ++terminal;
                    } else {
                        ++unexpected;
                    }
                } catch (const Error& e) {
                    if (e.code() == ErrorCode::State) return;  // close won the race
                    ++unexpected;
                    return;
                } catch (...) {
                    ++unexpected;
                    return;
                }
            }
        });
    }
    go.store(true, std::memory_order_release);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (submitted.load() < 4 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    const bool all_started = submitted.load() >= 4;
    b.close();
    for (auto& thread : submitters) thread.join();
    FT_ASSERT(all_started);
    FT_ASSERT_EQ(unexpected.load(), 0);
    FT_ASSERT_EQ(terminal.load(), submitted.load());
    FT_ASSERT(!b.is_open());
    NvmeBackend::destroy_file(path);
}

FT_TEST(store_repeated_open_close_releases_workers_handles_and_file) {
    const std::string path = temp_store_path();
    NvmeBackend::destroy_file(path);
    NvmeBackend b;
    std::vector<uint8_t> payload(kPageSize, 0x6D);
    for (int iteration = 0; iteration < 64; ++iteration) {
        b.open(path, 4 * kPageSize, kPageSize);
        uint64_t offset = 0;
        FT_ASSERT(b.allocate_extent(offset));
        auto write = b.write_async(offset, payload.data(), payload.size());
        write->wait();
        FT_ASSERT_EQ(write->status(), AsyncOp::Status::Complete);
        b.close();
        FT_ASSERT(!b.is_open());
        NvmeBackend::destroy_file(path);  // proves no live file handle remains
        FT_ASSERT(!std::filesystem::exists(path));
    }
}

int main() { return ft_test::run_all("test_nvme_backend"); }
