#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include "flashtier/error.hpp"

namespace flashtier {

// FNV-1a 64 checksum used for routine telemetry integrity.
uint64_t fnv1a64(const void* data, std::size_t n, uint64_t seed = 0xC0FFEE) noexcept;

// Deterministic content: each 8-byte block of a page is the little-endian
// encoding of splitmix64(seed ^ page_id ^ block_index). Any platform can
// regenerate the same expected bytes, so full content verification is always
// possible.
void fill_pattern(void* dst, std::size_t n, uint64_t seed, uint64_t page_id) noexcept;

struct IntegrityMismatch {
    std::size_t offset = 0;
    uint8_t expected = 0;
    uint8_t actual = 0;
};

// Verify deterministic content. Returns the first mismatch, if any.
// `check_limit` limits how many bytes are sampled (0 = verify everything).
std::optional<IntegrityMismatch> verify_pattern(
    const void* data, std::size_t n, uint64_t seed, uint64_t page_id,
    std::size_t check_limit = 0) noexcept;

}  // namespace flashtier
