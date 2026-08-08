#pragma once

#include <cstdint>
#include <mutex>

namespace flashtier {

// Per-tier memory budget tracker. Enforces the configured limit minus a
// reserve margin, tracks high-water, and never silently overcommits.
class TierBudget {
public:
    void set_limit(uint64_t limit_bytes, double reserve_margin);

    // Reserve `bytes` within the limit minus reserve. Returns false (and
    // reserves nothing) when the request would breach the reserve margin.
    bool try_reserve(uint64_t bytes);

    // Release a prior reservation. Over-release is an invariant violation,
    // not a reason to silently reset accounting.
    void release(uint64_t bytes);
    void force_reserve(uint64_t bytes);  // used for staging buffers that the
                                         // caller accounts for separately

    uint64_t used() const;
    uint64_t limit() const;
    uint64_t reserve_bytes() const;   // limit * margin
    uint64_t headroom() const;        // usable bytes above current use
    uint64_t high_water() const;
    double margin() const;

private:
    mutable std::mutex mu_;
    uint64_t limit_ = 0;
    uint64_t used_ = 0;
    uint64_t high_water_ = 0;
    double margin_ = 0.0;
};

}  // namespace flashtier
