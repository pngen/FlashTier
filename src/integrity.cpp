#include "flashtier/integrity.hpp"

#include <algorithm>

namespace flashtier {

uint64_t fnv1a64(const void* data, std::size_t n, uint64_t seed) noexcept {
    const auto* p = static_cast<const uint8_t*>(data);
    uint64_t h = seed;
    for (std::size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

namespace {

// splitmix64 finalizer; deterministic across platforms.
uint64_t splitmix64(uint64_t x) noexcept {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

}  // namespace

void fill_pattern(void* dst, std::size_t n, uint64_t seed, uint64_t page_id) noexcept {
    auto* bytes = static_cast<uint8_t*>(dst);
    const std::size_t blocks = n / 8;
    for (std::size_t b = 0; b < blocks; ++b) {
        const uint64_t v = splitmix64(seed ^ (page_id * 0x9E3779B97F4A7C15ull) ^ b);
        for (std::size_t j = 0; j < 8; ++j) {
            bytes[b * 8 + j] = static_cast<uint8_t>(v >> (j * 8));
        }
    }
    // Tail bytes (n % 8): derive from the last block.
    const std::size_t tail = n % 8;
    if (tail != 0) {
        const uint64_t v = splitmix64(seed ^ (page_id * 0x9E3779B97F4A7C15ull) ^ blocks);
        for (std::size_t j = 0; j < tail; ++j) {
            bytes[blocks * 8 + j] = static_cast<uint8_t>(v >> (j * 8));
        }
    }
}

std::optional<IntegrityMismatch> verify_pattern(
    const void* data, std::size_t n, uint64_t seed, uint64_t page_id,
    std::size_t check_limit) noexcept {
    const auto* bytes = static_cast<const uint8_t*>(data);
    const std::size_t limit = check_limit == 0 ? n : std::min(check_limit, n);

    const std::size_t blocks = n / 8;
    for (std::size_t b = 0; b < blocks && b * 8 < limit; ++b) {
        const uint64_t v = splitmix64(seed ^ (page_id * 0x9E3779B97F4A7C15ull) ^ b);
        for (std::size_t j = 0; j < 8 && b * 8 + j < limit; ++j) {
            const uint8_t expected = static_cast<uint8_t>(v >> (j * 8));
            if (bytes[b * 8 + j] != expected) {
                return IntegrityMismatch{b * 8 + j, expected, bytes[b * 8 + j]};
            }
        }
    }
    // Tail bytes (n % 8) derive from the last block, matching fill_pattern.
    const std::size_t tail = n % 8;
    if (tail != 0 && blocks * 8 < limit) {
        const uint64_t v = splitmix64(seed ^ (page_id * 0x9E3779B97F4A7C15ull) ^ blocks);
        for (std::size_t j = 0; j < tail && blocks * 8 + j < limit; ++j) {
            const uint8_t expected = static_cast<uint8_t>(v >> (j * 8));
            if (bytes[blocks * 8 + j] != expected) {
                return IntegrityMismatch{blocks * 8 + j, expected,
                                         bytes[blocks * 8 + j]};
            }
        }
    }
    return std::nullopt;
}

}  // namespace flashtier
