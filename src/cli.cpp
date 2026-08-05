#include "flashtier/cli.hpp"

#include <cstring>

#include "flashtier/error.hpp"

namespace flashtier {
namespace cli {

const char* version_string() {
    static const std::string s =
        std::string("FlashTier ") + FLASHTIER_VERSION + " (" + FLASHTIER_GIT_REVISION + ")";
    return s.c_str();
}

void print_version(FILE* out) {
    std::fprintf(out, "FlashTier %s (git %s)\n", FLASHTIER_VERSION, FLASHTIER_GIT_REVISION);
}

void print_usage(FILE* out) {
    std::fprintf(out,
                 "FlashTier %s - heterogeneous-memory runtime (VRAM / pinned RAM / NVMe)\n"
                 "\n"
                 "Usage:\n"
                 "  flashtier <command> [options]\n"
                 "  flashtier --version\n"
                 "  flashtier --help\n"
                 "\n"
                 "Commands:\n"
                 "  inspect                 Report OS, CPU, RAM, GPU, CUDA, and storage\n"
                 "  capabilities            Report supported backend capabilities\n"
                 "  verify                  Run promote/demote integrity cycles\n"
                 "  benchmark               Run the tier bandwidth benchmark\n"
                 "  benchmark tiers         Tier bandwidth benchmark\n"
                 "  benchmark oversubscription  Working sets exceeding VRAM budget\n"
                 "  benchmark prefetch      Compare prefetch off/sequential/predictive\n"
                 "  benchmark sparse-experts   Synthetic MoE-like routing workload\n"
                 "  benchmark unified-memory   Compare explicit vs cudaMallocManaged\n"
                 "\n"
                 "Options:\n"
                 "  --backend <name>         Device backend: auto (default), cuda, hip,\n"
                 "                           level_zero, vulkan, metal, cpu\n"
                 "  --device <id>            Device index within the backend (default 0)\n"
                 "  --page-size <bytes>      Page size (KiB/MiB/GiB/TiB suffixes)\n"
                 "  --working-set <bytes>    Logical working set\n"
                 "  --vram-budget <bytes>    VRAM budget (0 = auto)\n"
                 "  --host-budget <bytes>    Pinned host budget (0 = auto)\n"
                 "  --nvme-budget <bytes>    NVMe budget (0 = auto)\n"
                 "  --nvme-path <path>       Backing store path (default: temp dir)\n"
                 "  --policy <lru|predictive>\n"
                 "  --prefetch <off|sequential|predictive>\n"
                 "  --prefetch-depth <n>     Bounded prefetch queue depth\n"
                 "  --iterations <n>         Workload iterations\n"
                 "  --seed <n>               Deterministic workload seed\n"
                 "  --queue-depth <n>        Transfer worker/queue depth\n"
                 "  --jsonl                  Stream JSONL telemetry to stdout\n"
                 "  --output <dir>           Telemetry output directory (default: cwd)\n"
                 "  --retain-store           Keep the NVMe backing store after exit\n"
                 "  --strict                 Fail on any telemetry or integrity anomaly\n"
                 "  --no-cuda                Force CPU-only runtime mode\n"
                 "  -h, --help               Show this help\n"
                 "  -v, --version            Show version\n",
                 FLASHTIER_VERSION);
}

namespace {

bool next_arg(int& i, int argc, char** argv, std::string& out, Options& o,
              const char* opt) {
    if (i + 1 >= argc) {
        o.errors.push_back(std::string("missing value for ") + opt);
        return false;
    }
    out = argv[++i];
    return true;
}

bool parse_uint(const std::string& s, uint64_t& out) {
    if (s.empty()) return false;
    uint64_t v = 0;
    for (char c : s) {
        if (c < '0' || c > '9') return false;
        const uint64_t d = static_cast<uint64_t>(c - '0');
        if (v > (UINT64_MAX - d) / 10) return false;
        v = v * 10 + d;
    }
    out = v;
    return true;
}

}  // namespace

Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            o.show_help = true;
            continue;
        }
        if (arg == "--version" || arg == "-v") {
            o.show_version = true;
            continue;
        }
        if (arg == "--jsonl") {
            o.jsonl = true;
            continue;
        }
        if (arg == "--retain-store") {
            o.retain_store = true;
            continue;
        }
        if (arg == "--strict") {
            o.strict = true;
            continue;
        }
        if (arg == "--no-cuda") {
            o.no_cuda = true;
            continue;
        }

        auto value_for = [&](const char* name, std::string& dst) -> bool {
            return next_arg(i, argc, argv, dst, o, name);
        };

        if (arg == "--device") {
            std::string v;
            if (value_for("--device", v)) {
                uint64_t n = 0;
                if (!parse_uint(v, n) || n > 1024) {
                    o.errors.push_back("--device expects a nonnegative integer");
                } else {
                    o.device_id = static_cast<int>(n);
                }
            }
        } else if (arg == "--backend") {
            std::string v;
            if (value_for("--backend", v)) {
                const std::string known[] = {"auto", "cuda", "hip", "level_zero",
                                             "vulkan", "metal", "cpu"};
                bool ok = false;
                for (const auto& k : known) {
                    if (v == k) {
                        ok = true;
                        break;
                    }
                }
                if (!ok) {
                    o.errors.push_back(
                        "--backend: expected auto, cuda, hip, level_zero, vulkan, metal, or cpu");
                } else {
                    o.backend = v;
                }
            }
        } else if (arg == "--page-size") {
            std::string v;
            if (value_for("--page-size", v)) {
                try {
                    o.page_size = parse_bytesize(v);
                } catch (const Error& e) {
                    o.errors.push_back("--page-size: " + std::string(e.what()));
                }
            }
        } else if (arg == "--working-set") {
            std::string v;
            if (value_for("--working-set", v)) {
                try {
                    o.working_set_bytes = parse_bytesize(v);
                } catch (const Error& e) {
                    o.errors.push_back("--working-set: " + std::string(e.what()));
                }
            }
        } else if (arg == "--vram-budget") {
            std::string v;
            if (value_for("--vram-budget", v)) {
                try {
                    o.vram_budget_bytes = parse_bytesize(v);
                } catch (const Error& e) {
                    o.errors.push_back("--vram-budget: " + std::string(e.what()));
                }
            }
        } else if (arg == "--host-budget") {
            std::string v;
            if (value_for("--host-budget", v)) {
                try {
                    o.host_budget_bytes = parse_bytesize(v);
                } catch (const Error& e) {
                    o.errors.push_back("--host-budget: " + std::string(e.what()));
                }
            }
        } else if (arg == "--nvme-budget") {
            std::string v;
            if (value_for("--nvme-budget", v)) {
                try {
                    o.nvme_budget_bytes = parse_bytesize(v);
                } catch (const Error& e) {
                    o.errors.push_back("--nvme-budget: " + std::string(e.what()));
                }
            }
        } else if (arg == "--nvme-path") {
            std::string v;
            if (value_for("--nvme-path", v)) o.nvme_path = v;
        } else if (arg == "--policy") {
            std::string v;
            if (value_for("--policy", v)) {
                if (!policy_kind_from_name(v, o.policy)) {
                    o.errors.push_back("--policy: expected lru or predictive");
                }
            }
        } else if (arg == "--prefetch") {
            std::string v;
            if (value_for("--prefetch", v)) {
                if (!prefetch_kind_from_name(v, o.prefetch)) {
                    o.errors.push_back("--prefetch: expected off, sequential, or predictive");
                }
            }
        } else if (arg == "--prefetch-depth") {
            std::string v;
            if (value_for("--prefetch-depth", v)) {
                uint64_t n = 0;
                if (!parse_uint(v, n) || n == 0 || n > 1 << 20) {
                    o.errors.push_back("--prefetch-depth expects a positive integer");
                } else {
                    o.prefetch_depth = static_cast<uint32_t>(n);
                }
            }
        } else if (arg == "--queue-depth") {
            std::string v;
            if (value_for("--queue-depth", v)) {
                uint64_t n = 0;
                if (!parse_uint(v, n) || n == 0 || n > 1024) {
                    o.errors.push_back("--queue-depth expects a positive integer");
                } else {
                    o.queue_depth = static_cast<uint32_t>(n);
                }
            }
        } else if (arg == "--iterations") {
            std::string v;
            if (value_for("--iterations", v)) {
                uint64_t n = 0;
                if (!parse_uint(v, n) || n == 0 || n > 1000000) {
                    o.errors.push_back("--iterations expects a positive integer");
                } else {
                    o.iterations = static_cast<uint32_t>(n);
                }
            }
        } else if (arg == "--seed") {
            std::string v;
            if (value_for("--seed", v)) {
                uint64_t n = 0;
                if (!parse_uint(v, n)) {
                    o.errors.push_back("--seed expects an unsigned integer");
                } else {
                    o.seed = n;
                }
            }
        } else if (arg == "--output") {
            std::string v;
            if (value_for("--output", v)) o.output_dir = v;
        } else if (arg == "--um-prefetch-pages") {
            std::string v;
            if (value_for("--um-prefetch-pages", v)) {
                uint64_t n = 0;
                if (!parse_uint(v, n) || n == 0) {
                    o.errors.push_back("--um-prefetch-pages expects a positive integer");
                } else {
                    o.um_prefetch_pages = n;
                }
            }
        } else if (arg == "--auto-vram-fraction") {
            std::string v;
            if (value_for("--auto-vram-fraction", v)) {
                try {
                    const double d = std::stod(v);
                    if (d <= 0.0 || d > 1.0) {
                        o.errors.push_back("--auto-vram-fraction must be in (0, 1]");
                    } else {
                        o.auto_vram_fraction = d;
                    }
                } catch (...) {
                    o.errors.push_back("--auto-vram-fraction expects a number");
                }
            }
        } else if (arg == "inspect" || arg == "capabilities" || arg == "verify") {
            if (!o.command.empty()) {
                o.errors.push_back("multiple commands given");
            }
            o.command = arg;
        } else if (arg == "benchmark") {
            if (!o.command.empty()) {
                o.errors.push_back("multiple commands given");
            }
            o.command = "benchmark";
            if (i + 1 < argc) {
                const std::string sub = argv[i + 1];
                if (sub == "tiers" || sub == "oversubscription" || sub == "prefetch" ||
                    sub == "sparse-experts" || sub == "unified-memory") {
                    o.subcommand = sub;
                    ++i;
                }
            }
        } else {
            o.errors.push_back("unknown argument: " + arg);
        }
    }
    return o;
}

Config Options::to_config() const {
    Config c;
    c.backend = backend;
    c.device_id = device_id;
    c.page_size = page_size;
    c.working_set_bytes = working_set_bytes;
    c.vram_budget_bytes = vram_budget_bytes;
    c.host_budget_bytes = host_budget_bytes;
    c.nvme_budget_bytes = nvme_budget_bytes;
    c.nvme_path = nvme_path;
    c.policy = policy;
    c.prefetch = prefetch;
    c.prefetch_depth = prefetch_depth;
    c.queue_depth = queue_depth;
    c.worker_threads = worker_threads;
    c.iterations = iterations;
    c.seed = seed;
    c.vram_reserve_margin = vram_reserve_margin;
    c.output_dir = output_dir;
    c.retain_store = retain_store;
    c.strict = strict;
    c.cuda_enabled = !no_cuda;
    c.auto_vram_fraction = auto_vram_fraction;
    return c;
}

}  // namespace cli
}  // namespace flashtier
