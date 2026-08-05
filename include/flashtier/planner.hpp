#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "flashtier/config.hpp"
#include "flashtier/page.hpp"
#include "flashtier/page_id.hpp"
#include "flashtier/policy.hpp"
#include "flashtier/tier.hpp"

namespace flashtier {

// Placement planner. Decides where allocations land, whether a demand load
// can proceed, and which evictions free headroom.
class Planner {
public:
    struct Budgets {
        uint64_t vram_limit = 0;
        uint64_t vram_reserve = 0;
        uint64_t vram_used = 0;
        uint64_t host_limit = 0;
        uint64_t host_used = 0;
        uint64_t nvme_limit = 0;
        uint64_t nvme_used = 0;
    };

    struct EvictionDecision {
        PageId id;
        Tier target = Tier::None;  // HostPinned (demote) or Nvme (writeback)
        std::string reason;
    };

    explicit Planner(Budgets budgets);

    void set_budgets(Budgets budgets);

    // VRAM has at least `need_bytes` of usable headroom (limit - reserve -
    // used >= need).
    bool vram_has_headroom(uint64_t need_bytes) const;

    // Choose evictions from legal candidates until `need_bytes` of VRAM
    // headroom is free. Returns decisions worst-victim first. When no legal
    // victim exists and headroom is still insufficient, throws
    // ErrorCode::Budget.
    std::vector<EvictionDecision> plan_evictions(
        uint64_t need_bytes,
        const std::vector<PageMetadata>& resident_vram_candidates,
        const Policy& policy,
        uint64_t host_free_bytes) const;

    // Where a fresh allocation should land: Vram if headroom, else host if
    // room, else Nvme. Throws ErrorCode::Budget when nothing fits.
    Tier choose_allocation_tier(uint64_t bytes, bool allow_pageable) const;

    // Where a demand load should be staged: Vram when headroom exists,
    // otherwise host.
    Tier choose_load_target(uint64_t bytes) const;

private:
    Budgets b_;
};

}  // namespace flashtier
