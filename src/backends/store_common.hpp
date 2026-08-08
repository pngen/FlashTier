#pragma once

// Internal store header shared by storage_backend.cpp and nvme_backend.cpp.
// Not installed; never part of the public API.

#include <cstdint>
#include <cstddef>
#include <limits>
#include <string>

#include "flashtier/error.hpp"

namespace flashtier {
namespace detail {

constexpr uint32_t kStoreMagic = 0x46544E56;  // "FTNV"
constexpr uint32_t kStoreVersion = 2;
constexpr uint64_t kHeaderBlockSize = 4096;

struct StoreHeader {
    uint32_t magic = kStoreMagic;
    uint32_t version = kStoreVersion;
    uint64_t page_size = 0;
    uint64_t capacity_bytes = 0;
    uint64_t extent_count = 0;
    uint64_t generation = 0;
    uint64_t seed = 0;
    uint64_t checksum = 0;
    uint8_t reserved[4096 - (4 + 4 + 8 * 6)] = {};
};

static_assert(sizeof(StoreHeader) == 4096, "StoreHeader must be one 4 KiB block");

inline uint64_t store_header_checksum(const StoreHeader& header) noexcept {
    StoreHeader copy = header;
    copy.checksum = 0;
    const auto* bytes = reinterpret_cast<const uint8_t*>(&copy);
    uint64_t hash = 14695981039346656037ull;
    for (std::size_t i = 0; i < sizeof(copy); ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

inline void seal_store_header(StoreHeader& header) noexcept {
    header.checksum = store_header_checksum(header);
}

inline std::string header_mismatch_reason(const StoreHeader& h, uint64_t page_size,
                                          uint64_t capacity_bytes) {
    if (h.magic != kStoreMagic) return "magic mismatch (not a FlashTier store)";
    if (h.version != kStoreVersion)
        return "unsupported store version " + std::to_string(h.version);
    if (h.checksum != store_header_checksum(h)) return "header checksum mismatch";
    if (page_size != 0 && h.page_size != page_size)
        return "page size mismatch (header " + std::to_string(h.page_size) +
               ", requested " + std::to_string(page_size) + ")";
    if (capacity_bytes != 0 && h.capacity_bytes != capacity_bytes)
        return "capacity mismatch (header " + std::to_string(h.capacity_bytes) +
               ", requested " + std::to_string(capacity_bytes) + ")";
    if (h.page_size < kHeaderBlockSize || h.capacity_bytes == 0 || h.extent_count == 0)
        return "truncated or zeroed header";
    if (h.capacity_bytes % h.page_size != 0)
        return "capacity is not a multiple of page size";
    if (h.extent_count != h.capacity_bytes / h.page_size)
        return "extent count inconsistent with capacity";
    if (h.capacity_bytes > std::numeric_limits<uint64_t>::max() - h.page_size)
        return "store file size overflows 64 bits";
    for (const uint8_t byte : h.reserved) {
        if (byte != 0) return "reserved header bytes are nonzero";
    }
    return {};
}

}  // namespace detail
}  // namespace flashtier
