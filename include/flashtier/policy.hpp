#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "flashtier/config.hpp"
#include "flashtier/page.hpp"
#include "flashtier/page_id.hpp"

namespace flashtier {

// A placement policy decides victim order and promotion candidates. Both
// shipped policies are deterministic: ties are broken by stable page ID.
class Policy {
public:
    virtual ~Policy() = default;

    virtual PolicyKind kind() const noexcept = 0;

    // Update policy-internal state on a page access.
    virtual void on_access(const PageMetadata& page) = 0;

    // Score a page; higher = more valuable (worse victim, better promote
    // candidate). Deterministic for identical metadata.
    virtual double score(const PageMetadata& page) const = 0;

    // Order candidates from worst victim to best. Callers pass only legal
    // candidates (resident in VRAM, not pinned, not in flight).
    virtual std::vector<PageId> rank_victims(const std::vector<PageMetadata>& candidates) const = 0;

    // Order candidates from best promotion target to worst.
    virtual std::vector<PageId> rank_prefetch(const std::vector<PageMetadata>& candidates) const = 0;
};

std::unique_ptr<Policy> make_policy(PolicyKind kind);

// Deterministic LRU baseline. Not presented as optimal.
class LruPolicy final : public Policy {
public:
    PolicyKind kind() const noexcept override { return PolicyKind::Lru; }
    void on_access(const PageMetadata& page) override;
    double score(const PageMetadata& page) const override;
    std::vector<PageId> rank_victims(const std::vector<PageMetadata>& candidates) const override;
    std::vector<PageId> rank_prefetch(const std::vector<PageMetadata>& candidates) const override;
};

// Temperature-aware predictive policy. Heuristic score combining recency,
// frequency, expected next-use distance, semantic class, transfer cost,
// page size, dirty-write penalty, execution hints, and pin state.
// Documented heuristic, not mathematically optimal.
class PredictivePolicy final : public Policy {
public:
    PredictivePolicy();

    PolicyKind kind() const noexcept override { return PolicyKind::Predictive; }
    void on_access(const PageMetadata& page) override;
    double score(const PageMetadata& page) const override;
    std::vector<PageId> rank_victims(const std::vector<PageMetadata>& candidates) const override;
    std::vector<PageId> rank_prefetch(const std::vector<PageMetadata>& candidates) const override;

private:
    double class_weight(SemanticClass cls) const;
    double transfer_cost(Tier from, uint64_t bytes) const;
};

// Weight of a semantic class in predictive scoring (documented constants).
double semantic_class_weight(SemanticClass cls) noexcept;

}  // namespace flashtier
