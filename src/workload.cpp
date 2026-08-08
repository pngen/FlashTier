#include "flashtier/workload.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

#include "flashtier/error.hpp"

namespace flashtier {

uint64_t SplitMix64::next() noexcept {
    uint64_t z = (state_ += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

ZipfGenerator::ZipfGenerator(uint64_t seed, uint64_t n, double exponent)
    : rng_(seed), n_(n), exponent_(exponent) {
    if (n_ == 0) {
        throw Error(ErrorCode::InvalidArgument, "Zipf domain must contain at least one item");
    }
    if (!std::isfinite(exponent_) || exponent_ <= 0.0) {
        throw Error(ErrorCode::InvalidArgument, "Zipf exponent must be finite and positive");
    }
    if (n_ > cdf_.max_size()) {
        throw Error(ErrorCode::InvalidArgument, "Zipf domain is too large");
    }
    cdf_.resize(n_);
    double sum = 0.0;
    for (uint64_t i = 1; i <= n_; ++i) {
        sum += 1.0 / std::pow(static_cast<double>(i), exponent_);
    }
    double acc = 0.0;
    for (uint64_t i = 0; i < n_; ++i) {
        acc += 1.0 / std::pow(static_cast<double>(i + 1), exponent_) / sum;
        cdf_[i] = acc;
    }
    cdf_.back() = 1.0;  // guard against a rounded cumulative sum below one
}

uint64_t ZipfGenerator::next() noexcept {
    const double u = static_cast<double>(rng_.next() >> 11) / 9007199254740992.0;  // [0,1)
    const auto it = std::lower_bound(cdf_.begin(), cdf_.end(), u);
    return it == cdf_.end()
               ? n_ - 1
               : static_cast<uint64_t>(it - cdf_.begin());
}

TraceGenerator::TraceGenerator(uint64_t seed, uint64_t page_count, uint64_t op_count,
                               TracePattern pattern, double skew_exponent)
    : page_count_(page_count), op_count_(op_count) {
    generate(seed, page_count, op_count, pattern, skew_exponent);
}

void TraceGenerator::generate(uint64_t seed, uint64_t page_count, uint64_t op_count,
                              TracePattern pattern, double skew_exponent) {
    if (page_count == 0) {
        throw Error(ErrorCode::InvalidArgument, "trace page count must be positive");
    }
    if (op_count > accesses_.max_size()) {
        throw Error(ErrorCode::InvalidArgument, "trace operation count is too large");
    }
    accesses_.reserve(op_count);
    SplitMix64 rng(seed);

    if (pattern == TracePattern::Sequential) {
        uint64_t pos = rng.next() % page_count;
        for (uint64_t i = 0; i < op_count; ++i) {
            TraceAccess a;
            a.page_id = pos;
            a.write = (rng.next() & 7) == 0;  // ~12.5% writes
            accesses_.push_back(a);
            pos = (pos + 1) % page_count;
        }
    } else if (pattern == TracePattern::RandomUniform) {
        for (uint64_t i = 0; i < op_count; ++i) {
            TraceAccess a;
            a.page_id = rng.next() % page_count;
            a.write = (rng.next() & 7) == 0;
            accesses_.push_back(a);
        }
    } else if (pattern == TracePattern::Skewed) {
        ZipfGenerator zipf(seed ^ 0xA5A5A5A5, page_count, skew_exponent);
        for (uint64_t i = 0; i < op_count; ++i) {
            TraceAccess a;
            a.page_id = zipf.next();
            a.write = (rng.next() & 7) == 0;
            accesses_.push_back(a);
        }
    } else if (pattern == TracePattern::MoE) {
        // Simplified MoE: hot experts (first hot_count pages) are frequent.
        const uint64_t hot_count = std::max<uint64_t>(1, page_count / 8);
        ZipfGenerator zipf(seed ^ 0x5A5A5A5A, hot_count, skew_exponent);
        for (uint64_t i = 0; i < op_count; ++i) {
            TraceAccess a;
            if (page_count == hot_count || (rng.next() & 1)) {
                a.page_id = zipf.next() % hot_count;
            } else {
                a.page_id = hot_count + (rng.next() % (page_count - hot_count));
            }
            a.write = false;
            accesses_.push_back(a);
        }
    } else {
        throw Error(ErrorCode::InvalidArgument, "unknown trace pattern");
    }
}

std::vector<uint64_t> generate_moe_trace(
    uint64_t seed, uint64_t expert_count, uint64_t active_per_token,
    uint64_t hot_count, uint64_t pages_per_expert, double zipf_exponent,
    uint64_t tokens) {
    if (expert_count == 0) {
        throw Error(ErrorCode::InvalidArgument, "expert count must be positive");
    }
    if (active_per_token == 0 || active_per_token > expert_count) {
        throw Error(ErrorCode::InvalidArgument,
                    "active experts per token must be in [1, expert count]");
    }
    if (hot_count > expert_count) {
        throw Error(ErrorCode::InvalidArgument, "hot expert count cannot exceed expert count");
    }
    if (pages_per_expert == 0 ||
        expert_count > std::numeric_limits<uint64_t>::max() / pages_per_expert) {
        throw Error(ErrorCode::InvalidArgument, "invalid pages-per-expert geometry");
    }
    if (!std::isfinite(zipf_exponent) || zipf_exponent <= 0.0) {
        throw Error(ErrorCode::InvalidArgument, "Zipf exponent must be finite and positive");
    }
    std::vector<uint64_t> out;
    uint64_t experts_per_token = active_per_token;
    if (hot_count > 0 && experts_per_token < expert_count) ++experts_per_token;
    if (experts_per_token != 0 &&
        (tokens > std::numeric_limits<std::size_t>::max() / experts_per_token ||
         tokens * experts_per_token >
             std::numeric_limits<std::size_t>::max() / pages_per_expert)) {
        throw Error(ErrorCode::InvalidArgument, "MoE trace is too large");
    }
    const std::size_t reserve_count =
        static_cast<std::size_t>(tokens * experts_per_token * pages_per_expert);
    if (reserve_count > out.max_size()) {
        throw Error(ErrorCode::InvalidArgument, "MoE trace is too large");
    }
    out.reserve(reserve_count);
    SplitMix64 rng(seed);
    ZipfGenerator zipf(seed ^ 0x3D3D3D3D, expert_count, zipf_exponent);

    for (uint64_t t = 0; t < tokens; ++t) {
        // Active experts per token, drawn from the zipf distribution.
        std::vector<uint64_t> chosen;
        chosen.reserve(active_per_token);
        for (uint64_t a = 0; a < active_per_token; ++a) {
            uint64_t e = zipf.next();
            chosen.push_back(e);
        }
        // Shared hot experts are accessed every token deterministically.
        if (hot_count > 0 && (t % 2) == 0) {
            chosen.push_back(rng.next() % hot_count);
        }
        std::sort(chosen.begin(), chosen.end());
        chosen.erase(std::unique(chosen.begin(), chosen.end()), chosen.end());
        for (uint64_t e : chosen) {
            for (uint64_t p = 0; p < pages_per_expert; ++p) {
                out.push_back(e * pages_per_expert + p);
            }
        }
    }
    return out;
}

std::vector<TraceAccess> annotate_next_use(const std::vector<uint64_t>& page_ids) {
    // Next-use distance for each position; deterministic single pass.
    std::vector<uint64_t> last(page_ids.size(), 0);
    std::vector<TraceAccess> out;
    out.reserve(page_ids.size());

    std::unordered_map<uint64_t, uint64_t> next_pos;
    for (std::size_t i = page_ids.size(); i-- > 0;) {
        auto it = next_pos.find(page_ids[i]);
        last[i] = (it == next_pos.end()) ? 0 : it->second - i;
        next_pos[page_ids[i]] = i;
    }
    for (std::size_t i = 0; i < page_ids.size(); ++i) {
        TraceAccess a;
        a.page_id = page_ids[i];
        a.write = false;
        a.next_use_distance = last[i];
        out.push_back(a);
    }
    return out;
}

}  // namespace flashtier
