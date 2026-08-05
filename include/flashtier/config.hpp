#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "flashtier/tier.hpp"

namespace flashtier {

// Placement policies. Both are deterministic: identical inputs produce
// identical decisions (ties are broken by stable page ID).
enum class PolicyKind : int {
    Lru = 0,         // deterministic LRU baseline
    Predictive = 1,  // temperature-aware predictive heuristic
};

const char* policy_kind_name(PolicyKind kind) noexcept;
bool policy_kind_from_name(std::string_view name, PolicyKind& out) noexcept;

enum class PrefetchKind : int {
    Off = 0,
    Sequential = 1,
    Predictive = 2,
};

const char* prefetch_kind_name(PrefetchKind kind) noexcept;
bool prefetch_kind_from_name(std::string_view name, PrefetchKind& out) noexcept;

struct Config {
    // Device
    int device_id = 0;

    // Sizing
    uint64_t page_size = 2 * 1024 * 1024;      // 2 MiB
    uint64_t working_set_bytes = 0;            // 0 = derive from budgets
    uint64_t vram_budget_bytes = 0;            // 0 = auto (fraction of free VRAM)
    uint64_t host_budget_bytes = 0;            // 0 = auto (fraction of free RAM)
    uint64_t nvme_budget_bytes = 0;            // 0 = auto (fraction of free disk)
    std::string nvme_path;                     // empty = system temp dir
    double vram_reserve_margin = 0.10;         // fraction of VRAM budget kept free
    bool use_pageable_fallback = false;        // allow pageable host memory

    // Policy
    PolicyKind policy = PolicyKind::Predictive;
    PrefetchKind prefetch = PrefetchKind::Off;
    uint32_t prefetch_depth = 16;
    uint32_t queue_depth = 8;
    uint32_t worker_threads = 0;  // 0 = queue_depth

    // Workload
    uint32_t iterations = 1;
    uint64_t seed = 0xC0FFEE;

    // Behavior
    bool retain_store = false;
    bool strict = false;         // fail on any integrity/telemetry anomaly
    bool cuda_enabled = true;
    bool no_cleanup = false;     // reserved; use retain_store

    // Output
    std::string output_dir;      // telemetry directory ("" = cwd)
    std::string jsonl_path;      // explicit JSONL telemetry file

    // Automatic budget sizing. Requires detected free resources.
    double auto_vram_fraction = 0.75;
    double auto_host_fraction = 0.25;
    double auto_nvme_fraction = 0.25;
    uint64_t auto_nvme_cap = 64ull * 1024 * 1024 * 1024;  // 64 GiB

    std::string describe() const;  // human-readable configuration block
};

struct ConfigValidation {
    bool ok = false;
    std::vector<std::string> errors;
};

// Validates ranges and internal consistency. Never guesses: malformed or
// unsafe values are rejected.
ConfigValidation validate_config(const Config& cfg) noexcept;

// Strict byte-size parsing. Accepts an unsigned integer optionally followed
// by a binary suffix: KiB, MiB, GiB, TiB (case-insensitive). Bare bytes are
// accepted. Anything else (fractions, unknown suffixes, empty, overflow) is
// rejected with ErrorCode::Config.
uint64_t parse_bytesize(const std::string& text);

// Format a byte count with a binary suffix, e.g. "2.00 MiB".
std::string bytesize_to_string(uint64_t bytes);

}  // namespace flashtier
