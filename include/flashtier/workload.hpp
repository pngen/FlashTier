#pragma once

#include <cstdint>
#include <vector>

namespace flashtier {

// Deterministic workload trace generators. The same seed + configuration
// always produces the same access sequence.

// splitmix64 — deterministic, dependency-free RNG.
class SplitMix64 {
public:
    explicit SplitMix64(uint64_t seed) noexcept : state_(seed) {}
    uint64_t next() noexcept;

private:
    uint64_t state_;
};

// Zipf-like (power-law) page selector. Lower ranks are far more likely.
// Deterministic for a fixed seed.
class ZipfGenerator {
public:
    ZipfGenerator(uint64_t seed, uint64_t n, double exponent = 1.0);
    uint64_t next() noexcept;  // [0, n)

private:
    SplitMix64 rng_;
    uint64_t n_;
    double exponent_;
    std::vector<double> cdf_;
};

enum class TracePattern : int {
    Sequential = 0,   // pages in ascending order, wraps
    RandomUniform = 1,
    Skewed = 2,       // zipf-like hot set + tail
    MoE = 3,          // synthetic mixture-of-experts routing
};

// One access decision produced by a trace.
struct TraceAccess {
    uint64_t page_id = 0;  // logical page within the working set
    bool write = false;
    uint64_t next_use_distance = 0;  // 0 = unknown
};

// Deterministic page-access trace.
class TraceGenerator {
public:
    TraceGenerator(uint64_t seed, uint64_t page_count, uint64_t op_count,
                   TracePattern pattern, double skew_exponent = 1.0);

    const std::vector<TraceAccess>& accesses() const noexcept { return accesses_; }
    uint64_t page_count() const noexcept { return page_count_; }
    uint64_t op_count() const noexcept { return op_count_; }

private:
    void generate(uint64_t seed, uint64_t page_count, uint64_t op_count,
                  TracePattern pattern, double skew_exponent);

    std::vector<TraceAccess> accesses_;
    uint64_t page_count_;
    uint64_t op_count_;
};

// Synthetic MoE-like routing trace: per token, `active_per_token` experts
// are drawn with a Zipf distribution over `expert_count` experts; the first
// `hot_count` experts are shared/hot. Pages are assumed contiguous per
// expert. Returns page ids (logical, within the expert working set).
std::vector<uint64_t> generate_moe_trace(
    uint64_t seed, uint64_t expert_count, uint64_t active_per_token,
    uint64_t hot_count, uint64_t pages_per_expert, double zipf_exponent,
    uint64_t tokens);

// Convert a sequence of logical page ids into a TraceAccess list with
// next-use distance annotations (deterministic; O(n) with a map).
std::vector<TraceAccess> annotate_next_use(const std::vector<uint64_t>& page_ids);

}  // namespace flashtier
