#pragma once

#include <cstdint>
#include <string_view>

#include "flashtier/page_id.hpp"
#include "flashtier/tier.hpp"

namespace flashtier {

// Page state machine states. The authoritative copy lives in exactly one
// tier; transfer states mark in-flight movement. See ARCHITECTURE.md for
// the transition table.
enum class PageState : int {
    Unallocated = 0,
    ResidentNvme = 1,
    LoadingToHost = 2,
    ResidentHost = 3,
    LoadingToVram = 4,
    ResidentVram = 5,
    EvictingToHost = 6,
    EvictingToNvme = 7,
    Error = 8,
    Released = 9,
};

const char* page_state_name(PageState state) noexcept;

// True when the transition is part of the documented state machine.
// Illegal transitions are rejected by the runtime with ErrorCode::State.
bool transition_allowed(PageState from, PageState to) noexcept;

// Semantic classes describe the role of a region. The runtime does not
// depend on any model format; classes bias policy decisions only.
enum class SemanticClass : int {
    Generic = 0,
    Weight = 1,
    KvCache = 2,
    MoeExpert = 3,
    Activation = 4,
    Scratch = 5,
};

const char* semantic_class_name(SemanticClass cls) noexcept;
bool semantic_class_from_name(std::string_view name, SemanticClass& out) noexcept;

// Per-page metadata. The page table is the authority for every field here.
struct PageMetadata {
    static constexpr uint64_t kInvalidOffset = ~uint64_t{0};

    PageId id;

    uint64_t logical_size = 0;      // logical byte size requested by caller
    uint64_t allocation_size = 0;   // aligned physical allocation size
    uint64_t nvme_offset = kInvalidOffset;  // extent offset in the NVMe store

    PageState state = PageState::Unallocated;
    Tier current_tier = Tier::None;  // residency tier (distinguishes pinned/pageable host)
    Tier desired_tier = Tier::None;  // planner's target tier

    bool dirty = false;        // authoritative copy diverges from NVMe copy
    bool pinned = false;       // pinned pages are never evicted
    bool in_flight = false;    // a transfer is active on this page
    bool has_nvme_copy = false;  // a clean copy exists in the NVMe store

    uint64_t last_access_sequence = 0;  // monotonic; drives LRU
    uint64_t access_count = 0;
    uint64_t promotion_count = 0;
    uint64_t demotion_count = 0;

    uint64_t checksum = 0;  // FNV-1a 64 of authoritative content

    SemanticClass semantic_class = SemanticClass::Generic;
    int64_t execution_order_hint = -1;    // -1 = no hint
    int64_t reuse_distance_estimate = -1;  // -1 = unknown

    void* host_ptr = nullptr;  // valid only when state is ResidentHost
    void* vram_ptr = nullptr;  // valid only when state is ResidentVram

    // Apply a state transition, updating derived fields. Throws
    // ErrorCode::State on illegal transitions.
    void transition(PageState to);
};

}  // namespace flashtier
