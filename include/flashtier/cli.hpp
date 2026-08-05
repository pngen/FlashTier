#pragma once

#include <cstdio>
#include <string>
#include <vector>

#include "flashtier/config.hpp"

namespace flashtier {
namespace cli {

// CLI exit codes (stable contract).
constexpr int kExitOk = 0;
constexpr int kExitError = 1;
constexpr int kExitUsage = 2;
constexpr int kExitCorrupt = 3;
constexpr int kExitUnsupported = 4;
constexpr int kExitBenchmarkFailed = 5;
constexpr int kExitIntegrity = 6;
constexpr int kExitFatal = 7;

struct Options {
    std::string backend;       // "" = auto; explicit names: cuda, hip, level_zero, vulkan, metal, cpu
    int device_id = 0;
    uint64_t page_size = 2 * 1024 * 1024;
    uint64_t working_set_bytes = 0;
    uint64_t vram_budget_bytes = 0;
    uint64_t host_budget_bytes = 0;
    uint64_t nvme_budget_bytes = 0;
    std::string nvme_path;
    PolicyKind policy = PolicyKind::Predictive;
    PrefetchKind prefetch = PrefetchKind::Off;
    uint32_t prefetch_depth = 16;
    uint32_t queue_depth = 8;
    uint32_t worker_threads = 0;
    uint32_t iterations = 1;
    uint64_t seed = 0xC0FFEE;
    double vram_reserve_margin = 0.10;
    std::string output_dir;
    bool jsonl = false;        // stream JSONL to stdout
    bool retain_store = false;
    bool strict = false;
    bool no_cuda = false;

    // Benchmark extras.
    uint64_t um_prefetch_pages = 8;
    double auto_vram_fraction = 0.75;

    std::string command;       // "" | inspect | capabilities | verify | benchmark
    std::string subcommand;    // tiers | oversubscription | prefetch | sparse-experts | unified-memory
    bool show_help = false;
    bool show_version = false;
    std::vector<std::string> errors;

    Config to_config() const;
};

// Parses argv into Options. Errors are collected in `errors` (usage exit).
Options parse_args(int argc, char** argv);

void print_usage(FILE* out);
void print_version(FILE* out);
const char* version_string();

}  // namespace cli
}  // namespace flashtier
