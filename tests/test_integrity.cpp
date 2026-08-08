#include "test_harness.hpp"

#include <cstring>
#include <limits>
#include <vector>

#include "flashtier/error.hpp"
#include "flashtier/integrity.hpp"
#include "flashtier/workload.hpp"

using namespace flashtier;

FT_TEST(checksum_is_deterministic_and_order_sensitive) {
    const uint8_t a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const uint8_t b[] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 9};
    const uint64_t ca = fnv1a64(a, sizeof(a));
    const uint64_t cb = fnv1a64(b, sizeof(b));
    FT_ASSERT_EQ(ca, fnv1a64(a, sizeof(a)));
    FT_ASSERT(ca != cb);
    FT_ASSERT(ca != 0);
}

FT_TEST(fill_pattern_verifies_clean) {
    std::vector<uint8_t> buf(256 * 1024);
    fill_pattern(buf.data(), buf.size(), 42, 7);
    const auto mismatch = verify_pattern(buf.data(), buf.size(), 42, 7);
    FT_ASSERT(!mismatch.has_value());
    // Wrong page id must fail.
    FT_ASSERT(verify_pattern(buf.data(), buf.size(), 42, 8).has_value());
    // Wrong seed must fail.
    FT_ASSERT(verify_pattern(buf.data(), buf.size(), 43, 7).has_value());
}

FT_TEST(fill_pattern_detects_corruption) {
    std::vector<uint8_t> buf(64 * 1024);
    fill_pattern(buf.data(), buf.size(), 1, 3);
    buf[1000] ^= 0xFF;
    const auto mismatch = verify_pattern(buf.data(), buf.size(), 1, 3);
    FT_ASSERT(mismatch.has_value());
    FT_ASSERT_EQ(mismatch->offset, 1000u);
}

FT_TEST(fill_pattern_verifies_tail) {
    // Non-multiple-of-8 sizes still verify.
    std::vector<uint8_t> buf(4099);
    fill_pattern(buf.data(), buf.size(), 5, 9);
    FT_ASSERT(!verify_pattern(buf.data(), buf.size(), 5, 9).has_value());
    buf.back() ^= 0x01;
    FT_ASSERT(verify_pattern(buf.data(), buf.size(), 5, 9).has_value());
}

FT_TEST(fill_pattern_deterministic_across_calls) {
    std::vector<uint8_t> a(1024), b(1024);
    fill_pattern(a.data(), a.size(), 7, 11);
    fill_pattern(b.data(), b.size(), 7, 11);
    FT_ASSERT(std::memcmp(a.data(), b.data(), a.size()) == 0);
}

FT_TEST(fill_pattern_has_platform_independent_byte_order) {
    std::vector<uint8_t> buf(8);
    fill_pattern(buf.data(), buf.size(), 42, 7);
    const uint8_t expected[] = {0x89, 0xA8, 0x73, 0xDE,
                                0xC7, 0x05, 0xBD, 0xCB};
    FT_ASSERT(std::memcmp(buf.data(), expected, sizeof(expected)) == 0);
}

FT_TEST(verify_pattern_check_limit_is_an_exact_prefix) {
    std::vector<uint8_t> bytes(128);
    fill_pattern(bytes.data(), bytes.size(), 3, 4);
    bytes[100] ^= 0x01;
    FT_ASSERT(!verify_pattern(bytes.data(), bytes.size(), 3, 4, 100).has_value());
    const auto mismatch = verify_pattern(bytes.data(), bytes.size(), 3, 4, 101);
    FT_ASSERT(mismatch.has_value());
    FT_ASSERT_EQ(mismatch->offset, 100u);
}

FT_TEST(trace_generators_are_deterministic) {
    TraceGenerator g1(12345, 64, 1000, TracePattern::Skewed, 1.1);
    TraceGenerator g2(12345, 64, 1000, TracePattern::Skewed, 1.1);
    FT_ASSERT_EQ(g1.accesses().size(), g2.accesses().size());
    for (std::size_t i = 0; i < g1.accesses().size(); ++i) {
        FT_ASSERT_EQ(g1.accesses()[i].page_id, g2.accesses()[i].page_id);
    }
}

FT_TEST(trace_generators_respect_page_count) {
    for (TracePattern pat : {TracePattern::Sequential, TracePattern::RandomUniform,
                             TracePattern::Skewed, TracePattern::MoE}) {
        TraceGenerator g(1, 16, 500, pat, 1.0);
        for (const auto& a : g.accesses()) {
            FT_ASSERT(a.page_id < 16);
        }
    }
}

FT_TEST(single_page_moe_trace_does_not_divide_by_zero) {
    TraceGenerator g(1, 1, 500, TracePattern::MoE, 1.0);
    FT_ASSERT_EQ(g.accesses().size(), 500u);
    for (const auto& a : g.accesses()) FT_ASSERT_EQ(a.page_id, 0u);
}

FT_TEST(trace_generators_reject_malformed_geometry) {
    FT_ASSERT_THROWS(TraceGenerator(1, 0, 1, TracePattern::Sequential),
                     ErrorCode::InvalidArgument);
    FT_ASSERT_THROWS(TraceGenerator(1, 4, 1, static_cast<TracePattern>(99)),
                     ErrorCode::InvalidArgument);
    FT_ASSERT_THROWS(ZipfGenerator(1, 0, 1.0), ErrorCode::InvalidArgument);
    FT_ASSERT_THROWS(ZipfGenerator(1, 4, std::numeric_limits<double>::infinity()),
                     ErrorCode::InvalidArgument);
    FT_ASSERT_THROWS(generate_moe_trace(1, 4, 5, 0, 1, 1.0, 1),
                     ErrorCode::InvalidArgument);
    FT_ASSERT_THROWS(generate_moe_trace(1, 4, 0, 0, 1, 1.0, 1),
                     ErrorCode::InvalidArgument);
    FT_ASSERT_THROWS(generate_moe_trace(1, 4, 1, 5, 1, 1.0, 1),
                     ErrorCode::InvalidArgument);
}

FT_TEST(splitmix64_deterministic_sequence) {
    SplitMix64 rng(0xDEADBEEF);
    const uint64_t first = rng.next();
    const uint64_t second = rng.next();
    SplitMix64 rng2(0xDEADBEEF);
    FT_ASSERT_EQ(rng2.next(), first);
    FT_ASSERT_EQ(rng2.next(), second);
    FT_ASSERT(first != second);
}

FT_TEST(moe_trace_shapes) {
    auto trace = generate_moe_trace(7, 32, 2, 4, 2, 1.5, 100);
    // 100 tokens x (2 active + shared-hot half the tokens, deduplicated)
    // experts x 2 pages per expert; the exact count depends on dedup.
    FT_ASSERT(trace.size() >= 400u);
    FT_ASSERT(trace.size() <= 700u);
    for (uint64_t p : trace) {
        FT_ASSERT(p < 32 * 2);
    }
    const auto annotated = annotate_next_use(trace);
    FT_ASSERT_EQ(annotated.size(), trace.size());
    for (std::size_t i = 0; i < annotated.size(); ++i) {
        FT_ASSERT_EQ(annotated[i].page_id, trace[i]);
        if (annotated[i].next_use_distance != 0) {
            const std::size_t next = i + annotated[i].next_use_distance;
            FT_ASSERT(next < trace.size());
            FT_ASSERT_EQ(trace[next], trace[i]);
        }
    }
}

int main() { return ft_test::run_all("test_integrity"); }
