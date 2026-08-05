#include "flashtier/planner.hpp"

#include <algorithm>

#include "flashtier/error.hpp"

namespace flashtier {

Planner::Planner(Budgets budgets) : b_(budgets) {}

void Planner::set_budgets(Budgets budgets) { b_ = budgets; }

bool Planner::vram_has_headroom(uint64_t need_bytes) const {
    if (b_.vram_limit < b_.vram_reserve) return false;
    const uint64_t usable = b_.vram_limit - b_.vram_reserve;
    return b_.vram_used <= usable - std::min(need_bytes, usable) &&
           need_bytes <= usable - b_.vram_used;
}

std::vector<Planner::EvictionDecision> Planner::plan_evictions(
    uint64_t need_bytes,
    const std::vector<PageMetadata>& resident_vram_candidates,
    const Policy& policy,
    uint64_t host_free_bytes) const {
    std::vector<EvictionDecision> decisions;

    const uint64_t usable = b_.vram_limit - b_.vram_reserve;
    if (need_bytes <= usable - b_.vram_used) {
        return decisions;  // no eviction needed
    }
    uint64_t need = need_bytes - (usable - b_.vram_used);

    const std::vector<PageId> ordered = policy.rank_victims(resident_vram_candidates);
    std::vector<PageMetadata> by_id;
    for (const auto& p : resident_vram_candidates) by_id.push_back(p);

    auto find_by_id = [&](PageId id) -> const PageMetadata* {
        for (const auto& p : by_id) {
            if (p.id == id) return &p;
        }
        return nullptr;
    };

    for (PageId id : ordered) {
        if (need == 0) break;
        const PageMetadata* meta = find_by_id(id);
        if (meta == nullptr) continue;
        if (meta->pinned) continue;
        if (meta->in_flight) continue;

        Tier target = Tier::HostPinned;
        std::string reason = "demote_to_host";
        if (meta->dirty && host_free_bytes < meta->allocation_size) {
            // Dirty pages must be persisted before authority moves: stage
            // through host only if room exists; otherwise write to NVMe.
            target = Tier::Nvme;
            reason = "writeback_to_nvme";
        } else if (!meta->dirty && meta->has_nvme_copy && host_free_bytes < meta->allocation_size) {
            target = Tier::Nvme;
            reason = "drop_to_nvme_clean";
        }

        decisions.push_back({id, target, reason});
        const uint64_t freed = std::min(need, meta->allocation_size);
        need = (freed >= need) ? 0 : need - freed;
    }

    if (need != 0) {
        throw Error(ErrorCode::Budget,
                    "no legal eviction victim exists for requested VRAM headroom",
                    "need " + std::to_string(need) + " bytes more after exhausting candidates");
    }
    return decisions;
}

Tier Planner::choose_allocation_tier(uint64_t bytes, bool allow_pageable) const {
    if (vram_has_headroom(bytes)) return Tier::Vram;
    const uint64_t host_usable = b_.host_limit - b_.host_used;
    if (bytes <= host_usable) return Tier::HostPinned;
    const uint64_t nvme_usable = b_.nvme_limit - b_.nvme_used;
    if (bytes <= nvme_usable) return Tier::Nvme;
    (void)allow_pageable;
    throw Error(ErrorCode::Budget,
                "no tier can host the requested allocation",
                "vram " + std::to_string(b_.vram_used) + "/" + std::to_string(b_.vram_limit) +
                    " host " + std::to_string(b_.host_used) + "/" + std::to_string(b_.host_limit) +
                    " nvme " + std::to_string(b_.nvme_used) + "/" + std::to_string(b_.nvme_limit));
}

Tier Planner::choose_load_target(uint64_t bytes) const {
    return vram_has_headroom(bytes) ? Tier::Vram : Tier::HostPinned;
}

}  // namespace flashtier
