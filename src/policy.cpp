#include "flashtier/policy.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

#include "flashtier/error.hpp"

namespace flashtier {

// ---------------------------------------------------------------------------
// Deterministic LRU baseline
// ---------------------------------------------------------------------------

void LruPolicy::on_access(const PageMetadata& page) {
    (void)page;  // LRU uses last_access_sequence recorded by the runtime
}

double LruPolicy::score(const PageMetadata& page) const {
    // Older last-access sequence = worse (better victim). Deterministic.
    return page.last_access_sequence == 0
               ? 0.0
               : static_cast<double>(page.last_access_sequence);
}

std::vector<PageId> LruPolicy::rank_victims(
    const std::vector<PageMetadata>& candidates) const {
    std::vector<PageMetadata> c = candidates;
    std::sort(c.begin(), c.end(), [](const PageMetadata& a, const PageMetadata& b) {
        if (a.last_access_sequence != b.last_access_sequence) {
            return a.last_access_sequence < b.last_access_sequence;
        }
        return a.id < b.id;  // deterministic tie-break
    });
    std::vector<PageId> out;
    out.reserve(c.size());
    for (const auto& p : c) out.push_back(p.id);
    return out;
}

std::vector<PageId> LruPolicy::rank_prefetch(
    const std::vector<PageMetadata>& candidates) const {
    std::vector<PageMetadata> c = candidates;
    std::sort(c.begin(), c.end(), [](const PageMetadata& a, const PageMetadata& b) {
        if (a.last_access_sequence != b.last_access_sequence) {
            return a.last_access_sequence > b.last_access_sequence;
        }
        return a.id < b.id;
    });
    std::vector<PageId> out;
    out.reserve(c.size());
    for (const auto& p : c) out.push_back(p.id);
    return out;
}

// ---------------------------------------------------------------------------
// Temperature-aware predictive policy
// ---------------------------------------------------------------------------

double semantic_class_weight(SemanticClass cls) noexcept {
    switch (cls) {
        case SemanticClass::KvCache: return 2.5;
        case SemanticClass::Weight: return 1.5;
        case SemanticClass::MoeExpert: return 2.0;
        case SemanticClass::Activation: return 1.2;
        case SemanticClass::Scratch: return 0.8;
        case SemanticClass::Generic:
        default: return 1.0;
    }
}

PredictivePolicy::PredictivePolicy() = default;

void PredictivePolicy::on_access(const PageMetadata& page) {
    // Frequency is tracked in PageMetadata::access_count; the policy itself
    // is stateless so identical inputs yield identical scores.
    (void)page;
}

double PredictivePolicy::class_weight(SemanticClass cls) const {
    return semantic_class_weight(cls);
}

double PredictivePolicy::transfer_cost(Tier from, uint64_t bytes) const {
    // Relative cost model, deterministic. NVMe is far slower than VRAM;
    // larger pages cost more. Units are normalized into the score.
    const double mb = static_cast<double>(bytes) / (1024.0 * 1024.0);
    switch (from) {
        case Tier::Nvme: return 20.0 + mb * 2.0;
        case Tier::HostPinned: return 2.0 + mb * 0.1;
        default: return 0.0;
    }
}

double PredictivePolicy::score(const PageMetadata& page) const {
    // Heuristic temperature score. Higher = keep in VRAM / promote first.
    // Terms (documented heuristic, not optimal):
    //   recency  : monotonic, bounded weight derived from the global access
    //              sequence (larger sequence = more recently accessed)
    //   frequency: min(1, access_count / 8)
    //   reuse    : next-use hint proximity (execution_order_hint)
    //   class    : semantic_class_weight
    //   pin      : pinned pages are worth +2
    //   dirty    : dirty pages cost a writeback when evicted (+1 to keep)
    //   cost     : transfer cost of loading from the current tier (-)
    double score = 0.0;

    if (page.access_count > 0) {
        const double freq = std::min(1.0, static_cast<double>(page.access_count) / 8.0);
        score += 2.0 * freq;
    } else {
        score += 0.2;  // never touched: cold
    }

    if (page.last_access_sequence > 0) {
        const double age_rank =
            std::log2(1.0 + static_cast<double>(page.last_access_sequence));
        score += age_rank / (1.0 + age_rank);
    }

    if (page.reuse_distance_estimate >= 0) {
        score += 1.0 / (1.0 + static_cast<double>(page.reuse_distance_estimate));
    }
    if (page.execution_order_hint >= 0) {
        score += 0.5;
    }

    score *= class_weight(page.semantic_class);

    if (page.pinned) score += 2.0;
    if (page.dirty) score += 1.0;

    score -= transfer_cost(page.current_tier, page.allocation_size);

    return score;
}

std::vector<PageId> PredictivePolicy::rank_victims(
    const std::vector<PageMetadata>& candidates) const {
    std::vector<PageMetadata> c = candidates;
    std::sort(c.begin(), c.end(), [this](const PageMetadata& a, const PageMetadata& b) {
        const double sa = score(a);
        const double sb = score(b);
        if (sa != sb) return sa < sb;       // lowest score = best victim
        return a.id < b.id;                 // deterministic tie-break
    });
    std::vector<PageId> out;
    out.reserve(c.size());
    for (const auto& p : c) out.push_back(p.id);
    return out;
}

std::vector<PageId> PredictivePolicy::rank_prefetch(
    const std::vector<PageMetadata>& candidates) const {
    std::vector<PageMetadata> c = candidates;
    std::sort(c.begin(), c.end(), [this](const PageMetadata& a, const PageMetadata& b) {
        const double sa = score(a);
        const double sb = score(b);
        if (sa != sb) return sa > sb;       // highest score = promote first
        return a.id < b.id;
    });
    std::vector<PageId> out;
    out.reserve(c.size());
    for (const auto& p : c) out.push_back(p.id);
    return out;
}

std::unique_ptr<Policy> make_policy(PolicyKind kind) {
    switch (kind) {
        case PolicyKind::Lru: return std::make_unique<LruPolicy>();
        case PolicyKind::Predictive: return std::make_unique<PredictivePolicy>();
    }
    throw Error(ErrorCode::InvalidArgument, "unknown placement policy");
}

}  // namespace flashtier
