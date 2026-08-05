#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "flashtier/bench.hpp"
#include "flashtier/cli.hpp"
#include "flashtier/config.hpp"
#include "flashtier/device_info.hpp"
#include "flashtier/error.hpp"
#include "flashtier/runtime.hpp"
#include "flashtier/telemetry.hpp"

#if FLASHTIER_HAVE_CUDA
#include "flashtier/backends/unified_memory.hpp"
#include "flashtier/backends/vram_backend.hpp"
#endif

using namespace flashtier;

namespace {

std::string gb(double bytes) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.2f GiB", bytes / (1024.0 * 1024.0 * 1024.0));
    return buf;
}

void print_inspect(const SystemInfo& info, int device_id) {
    std::printf("=== FlashTier inspect ===\n");
    std::printf("build:        FlashTier %s (git %s)\n", FLASHTIER_VERSION,
                FLASHTIER_GIT_REVISION);
    std::printf("os:           %s\n", info.os.c_str());
    std::printf("cpu:          %s\n", info.cpu.c_str());
    std::printf("system ram:   total %s free %s\n",
                gb(static_cast<double>(info.total_ram_bytes)).c_str(),
                gb(static_cast<double>(info.free_ram_bytes)).c_str());
    if (info.gpus.empty()) {
        std::printf("gpu:          none detected\n");
    } else {
        for (std::size_t i = 0; i < info.gpus.size(); ++i) {
            const GpuInfo& g = info.gpus[i];
            std::printf("gpu[%zu]:       %s (CUDA %s)\n", i, g.name.c_str(),
                        g.cuda_available ? "available" : "unavailable");
            if (g.cuda_available) {
                std::printf("  compute capability: %d.%d\n", g.major, g.minor);
                std::printf("  vram: total %s free %s\n",
                            gb(static_cast<double>(g.total_bytes)).c_str(),
                            gb(static_cast<double>(g.free_bytes)).c_str());
                std::printf("  unified memory: %s\n", g.unified_memory ? "yes" : "no");
                std::printf("  concurrent managed access: %s\n",
                            g.concurrent_managed_access ? "yes" : "no");
            }
        }
    }
    std::printf("cuda runtime: %s\n",
                info.cuda_runtime_version
                    ? ("version " + std::to_string(info.cuda_runtime_version)).c_str()
                    : "not available in this build");
    std::printf("cuda driver:  %s\n",
                info.cuda_driver_version
                    ? ("version " + std::to_string(info.cuda_driver_version)).c_str()
                    : "not available");
    std::printf("nvme path:    %s\n",
                info.nvme_path.empty() ? "(temp dir)" : info.nvme_path.c_str());
    std::printf("disk:         total %s free %s\n",
                gb(static_cast<double>(info.disk_total_bytes)).c_str(),
                gb(static_cast<double>(info.disk_free_bytes)).c_str());
    std::printf("directstorage:%s\n", info.directstorage_detected ? "probe present" : "not integrated (v0.1 probe)");
    std::printf("gds:          %s\n", info.gds_detected ? "cuFile detected" : "not detected (Linux-first future backend)");
    const GpuInfo primary = info.primary_gpu();
    if (primary.cuda_available) {
        std::printf("selected device: %s (device %d)\n", primary.name.c_str(), device_id);
    }
    std::printf("=== end inspect ===\n");
}

void print_capabilities(const SystemInfo& info) {
    std::printf("=== FlashTier capabilities ===\n");
    std::printf("platform:     %s\n", info.os.c_str());
    std::printf("cuda build:   %s\n", info.cuda_enabled ? "enabled" : "disabled (CPU-only)");
    if (info.cuda_enabled) {
        std::printf("gpu:          %s\n",
                    info.primary_gpu().cuda_available ? info.primary_gpu().name.c_str()
                                                      : "none available");
        if (info.primary_gpu().cuda_available) {
            const GpuInfo& g = info.primary_gpu();
            std::printf("  compute capability: %d.%d\n", g.major, g.minor);
            std::printf("  unified memory: %s\n", g.unified_memory ? "yes" : "no");
            std::printf("  concurrent managed access: %s\n",
                        g.concurrent_managed_access ? "yes" : "no");
        }
    }
    std::printf("tiers:        vram / host_pinned / nvme\n");
    std::printf("  vram:       %s\n",
                info.primary_gpu().cuda_available ? "explicit CUDA allocations"
                                                  : "unavailable in this build");
    std::printf("  host:       %s\n", "pinned when CUDA present; pageable fallback only when explicitly enabled");
    std::printf("  nvme:       %s\n", "file-backed store; Windows overlapped I/O (IOCP)");
    std::printf("policies:     lru (deterministic baseline), predictive (temperature-aware heuristic)\n");
    std::printf("prefetch:     off, sequential, predictive (bounded queue, cancellation)\n");
    std::printf("unified memory comparison benchmark: %s\n",
                (info.cuda_enabled && info.primary_gpu().cuda_available &&
                 info.primary_gpu().unified_memory)
                    ? "supported (bounded)"
                    : "not supported by this build/device");
    std::printf("directstorage: %s\n", "experimental probe only; no v0.1 integration");
    std::printf("gpudirect storage: %s\n", "interface-ready; Linux-first, not implemented in v0.1");
    std::printf("=== end capabilities ===\n");
}

int run_inspect(const cli::Options& o) {
    try {
        SystemInfo info = probe_system(o.device_id, o.nvme_path);
        print_inspect(info, o.device_id);
        return cli::kExitOk;
    } catch (const Error& e) {
        std::fprintf(stderr, "flashtier: inspect failed: %s\n", e.what());
        return cli::kExitError;
    }
}

int run_capabilities(const cli::Options& o) {
    try {
        SystemInfo info = probe_system(o.device_id, o.nvme_path);
        print_capabilities(info);
        return cli::kExitOk;
    } catch (const Error& e) {
        std::fprintf(stderr, "flashtier: capabilities failed: %s\n", e.what());
        return cli::kExitError;
    }
}

int run_verify(const cli::Options& o) {
    std::printf("=== FlashTier verify ===\n");
    Config cfg = o.to_config();
    try {
        const auto v = validate_config(cfg);
        if (!v.ok) {
            for (const auto& e : v.errors) std::fprintf(stderr, "config: %s\n", e.c_str());
            return cli::kExitUsage;
        }
        Runtime rt(cfg);
        rt.start();
        std::printf("config:\n%s\n", cfg.describe().c_str());

        const uint64_t page_count = cfg.working_set_bytes / cfg.page_size == 0
                                        ? 64
                                        : cfg.working_set_bytes / cfg.page_size;
        std::printf("verifying %llu pages through full promote/demote cycles\n",
                    static_cast<unsigned long long>(page_count));

        std::vector<PageId> ids;
        for (uint64_t i = 0; i < page_count; ++i) {
            ids.push_back(rt.allocate_page(cfg.page_size));
        }
        for (PageId id : ids) rt.fill_page(id);

        const int rounds = static_cast<int>(cfg.iterations == 0 ? 1 : cfg.iterations);
        for (int r = 0; r < rounds; ++r) {
            for (PageId id : ids) {
                rt.demote_to_nvme(id);
                rt.verify_page(id);
                rt.promote(id);
                rt.verify_page(id);
                rt.demote_to_host(id);
                rt.verify_page(id);
            }
        }
        for (PageId id : ids) rt.free_page(id);

        const auto agg = rt.aggregates();
        std::printf("integrity: %llu checks, %llu failures\n",
                    static_cast<unsigned long long>(agg.integrity_checks),
                    static_cast<unsigned long long>(agg.integrity_failures));
        std::printf("telemetry:\n%s", format_aggregates(agg).c_str());
        rt.shutdown();
        if (agg.integrity_failures != 0) {
            return cli::kExitIntegrity;
        }
        std::printf("verify: OK\n");
        return cli::kExitOk;
    } catch (const Error& e) {
        std::fprintf(stderr, "flashtier: verify failed: %s", e.what());
        if (!e.detail().empty()) std::fprintf(stderr, " (%s)", e.detail().c_str());
        std::fprintf(stderr, "\n");
        return e.code() == ErrorCode::Integrity ? cli::kExitIntegrity : cli::kExitError;
    }
}

}  // namespace

int main(int argc, char** argv) {
    cli::Options o = cli::parse_args(argc, argv);

    if (o.show_version) {
        cli::print_version(stdout);
        return cli::kExitOk;
    }
    if (o.show_help) {
        cli::print_usage(stdout);
        return cli::kExitOk;
    }
    if (!o.errors.empty()) {
        for (const auto& e : o.errors) {
            std::fprintf(stderr, "flashtier: %s\n", e.c_str());
        }
        cli::print_usage(stderr);
        return cli::kExitUsage;
    }
    if (o.command.empty()) {
        cli::print_usage(stderr);
        return cli::kExitUsage;
    }

    if (o.command == "inspect") return run_inspect(o);
    if (o.command == "capabilities") return run_capabilities(o);
    if (o.command == "verify") return run_verify(o);

    // benchmark
    return flashtier::bench::run_benchmark_command(o);
}
