#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "flashtier/backends/backend_registry.hpp"
#include "flashtier/bench.hpp"
#include "flashtier/cli.hpp"
#include "flashtier/config.hpp"
#include "flashtier/device_info.hpp"
#include "flashtier/error.hpp"
#include "flashtier/runtime.hpp"
#include "flashtier/telemetry.hpp"

#if FLASHTIER_HAVE_CUDA
#include "flashtier/backends/unified_memory.hpp"
#endif

using namespace flashtier;

namespace {

std::string gb(double bytes) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.2f GiB", bytes / (1024.0 * 1024.0 * 1024.0));
    return buf;
}

std::string yes_no(bool v) { return v ? "yes" : "no"; }

void print_device_line(const DeviceInfo& d) {
    std::printf("    %-14s %-32s arch=%-10s discrete=%s shared_mem=%s total=%s free=%s\n",
                d.id.c_str(), d.name.c_str(), d.architecture.c_str(),
                yes_no(d.discrete).c_str(), yes_no(d.memory_shared).c_str(),
                gb(static_cast<double>(d.total_memory)).c_str(),
                gb(static_cast<double>(d.free_memory)).c_str());
}

void print_backends(const BackendRegistry& registry) {
    std::printf("compiled backends: ");
    std::string compiled;
    for (const auto& b : registry.compiled_backends()) {
        if (!compiled.empty()) compiled += ", ";
        compiled += b;
    }
    std::printf("%s\n", compiled.empty() ? "(none)" : compiled.c_str());
    for (const BackendStatus& status : registry.probe_all()) {
        std::printf("  backend %-12s vendor=%-28s %s\n", status.name.c_str(),
                    status.vendor.c_str(), status.available ? "available" : "unavailable");
        if (!status.available) {
            std::printf("    reason: %s\n",
                        status.reason.empty() ? "(no devices)" : status.reason.c_str());
        } else {
            for (const auto& d : status.devices) {
                print_device_line(d);
            }
        }
    }
}

// Resolve the selected backend/device without opening anything (used by
// inspect/capabilities). Returns the backend name; "cpu" means CPU-only.
std::string resolve_selected_backend(const cli::Options& o, std::string& reason) {
    BackendRegistry& registry = BackendRegistry::instance();
    if (!o.backend.empty() && o.backend != "auto") {
        if (!registry.has(o.backend)) {
            throw Error(ErrorCode::Config,
                        "requested backend is not compiled into this build: " + o.backend);
        }
        reason = "explicit --backend " + o.backend;
        return o.backend;
    }
    return registry.select_automatic(reason);
}

void print_selected_device(const cli::Options& o) {
    std::string reason;
    const std::string backend = resolve_selected_backend(o, reason);
    std::printf("selected backend: %s (%s)\n", backend.c_str(), reason.c_str());
    if (backend == "cpu") {
        std::printf("cpu-only fallback state: active (no accelerator tier; no GPU execution claimed)\n");
        return;
    }
    std::printf("cpu-only fallback state: inactive\n");
    BackendRegistry& registry = BackendRegistry::instance();
    std::unique_ptr<DeviceBackend> dev = registry.create(backend);
    const std::vector<DeviceInfo> devices = dev->enumerate_devices();
    if (o.device_id >= static_cast<int>(devices.size())) {
        std::printf("selected device: invalid index %d (available: %zu)\n", o.device_id,
                    devices.size());
        return;
    }
    const DeviceInfo& d = devices[static_cast<std::size_t>(o.device_id)];
    std::printf("selected device: %s (%s)\n", d.id.c_str(), d.name.c_str());
    std::printf("  architecture: %s | discrete: %s | shared memory: %s | total: %s\n",
                d.architecture.c_str(), yes_no(d.discrete).c_str(),
                yes_no(d.memory_shared).c_str(),
                gb(static_cast<double>(d.total_memory)).c_str());
    DeviceCapabilities caps = dev->probe_capabilities();
    std::printf("  explicit allocation: %s | async H2D: %s | async D2H: %s | pinned host: %s\n",
                yes_no(caps.explicit_allocation).c_str(),
                yes_no(caps.async_host_to_device).c_str(),
                yes_no(caps.async_device_to_host).c_str(),
                yes_no(caps.pinned_host_allocation).c_str());
    std::printf("  unified memory: %s | concurrent managed access: %s | prefetch: %s | advice: %s\n",
                yes_no(caps.unified_memory).c_str(),
                yes_no(caps.concurrent_managed_access).c_str(),
                yes_no(caps.memory_prefetch).c_str(),
                yes_no(caps.memory_advice).c_str());
    std::printf("  direct storage: %s | p2p: %s | multi-device: %s | hw page faults: %s\n",
                yes_no(caps.direct_storage).c_str(), yes_no(caps.peer_to_peer).c_str(),
                yes_no(caps.multi_device).c_str(),
                yes_no(caps.hardware_page_fault).c_str());
    std::printf("  max allocation: %s | alignment: %llu | transfer granularity: %llu | queues: %u\n",
                gb(static_cast<double>(caps.max_allocation_size)).c_str(),
                static_cast<unsigned long long>(caps.alignment_bytes),
                static_cast<unsigned long long>(caps.transfer_granularity),
                caps.queue_count);
    if (!caps.note.empty()) {
        std::printf("  note: %s\n", caps.note.c_str());
    }
}

void print_inspect(const SystemInfo& info, const cli::Options& o) {
    std::printf("=== FlashTier inspect ===\n");
    std::printf("build:        FlashTier %s (git %s)\n", FLASHTIER_VERSION,
                FLASHTIER_GIT_REVISION);
    std::printf("os:           %s\n", info.os.c_str());
    std::printf("cpu:          %s\n", info.cpu.c_str());
    std::printf("system ram:   total %s free %s\n",
                gb(static_cast<double>(info.total_ram_bytes)).c_str(),
                gb(static_cast<double>(info.free_ram_bytes)).c_str());
    std::printf("accelerators:\n");
    print_backends(BackendRegistry::instance());
    if (info.cuda_enabled && info.cuda_runtime_version != 0) {
        std::printf("cuda runtime: version %d\n", info.cuda_runtime_version);
        std::printf("cuda driver:  version %d\n", info.cuda_driver_version);
    }
    std::printf("nvme path:    %s\n",
                info.nvme_path.empty() ? "(temp dir)" : info.nvme_path.c_str());
    std::printf("disk:         total %s free %s\n",
                gb(static_cast<double>(info.disk_total_bytes)).c_str(),
                gb(static_cast<double>(info.disk_free_bytes)).c_str());
    std::printf("directstorage: %s\n",
                info.directstorage_detected ? "probe present" : "not integrated (v0.1 probe)");
    std::printf("gds:          %s\n",
                info.gds_detected ? "cuFile detected" : "not detected (Linux-first future backend)");
    print_selected_device(o);
    std::printf("=== end inspect ===\n");
}

void print_capabilities(const SystemInfo& info, const cli::Options& o) {
    std::printf("=== FlashTier capabilities ===\n");
    std::printf("platform:     %s\n", info.os.c_str());
    std::printf("accelerators:\n");
    print_backends(BackendRegistry::instance());
    std::printf("tiers:        device (Tier::Vram) / host_pinned / nvme\n");
    std::printf("  device:     driven through the vendor-neutral DeviceBackend contract\n");
    std::printf("  host:       pinned via the active device backend; pageable fallback only "
                "when explicitly enabled\n");
    std::printf("  nvme:       file-backed store; Windows overlapped I/O (IOCP)\n");
    std::printf("policies:     lru (deterministic baseline), predictive (temperature-aware heuristic)\n");
    std::printf("prefetch:     off, sequential, predictive (bounded queue, cancellation)\n");
    std::printf("unified memory comparison benchmark: %s\n",
                (info.cuda_enabled && info.primary_gpu().cuda_available &&
                 info.primary_gpu().unified_memory)
                    ? "supported (bounded, CUDA)"
                    : "not supported by this build/device");
    std::printf("directstorage: %s\n", "experimental probe only; no v0.1 integration");
    std::printf("gpudirect storage: %s\n", "interface-ready; Linux-first, not implemented in v0.1");
    print_selected_device(o);
    std::printf("=== end capabilities ===\n");
}

int run_inspect(const cli::Options& o) {
    try {
        SystemInfo info = probe_system(o.device_id, o.nvme_path);
        print_inspect(info, o);
        return cli::kExitOk;
    } catch (const Error& e) {
        std::fprintf(stderr, "flashtier: inspect failed: %s\n", e.what());
        return cli::kExitError;
    }
}

int run_capabilities(const cli::Options& o) {
    try {
        SystemInfo info = probe_system(o.device_id, o.nvme_path);
        print_capabilities(info, o);
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
        const auto v = validate_config(cfg, BackendRegistry::instance().compiled_backends());
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
