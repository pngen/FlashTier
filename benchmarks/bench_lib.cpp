// Shared benchmark implementations used by both the flashtier CLI and the
// standalone benchmark executables. Methodology documented in
// BENCHMARKS.md: deterministic seeds, printed configuration blocks,
// capacity validation before allocation, reserve margins, and no claims
// beyond what is measured.

#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <string>
#include <vector>

#include "flashtier/backends/backend_registry.hpp"
#include "flashtier/backends/device_backend.hpp"
#include "flashtier/backends/nvme_backend.hpp"
#include "flashtier/bench.hpp"
#include "flashtier/cli.hpp"
#include "flashtier/device_info.hpp"
#include "flashtier/error.hpp"
#include "flashtier/integrity.hpp"
#include "flashtier/runtime.hpp"
#include "flashtier/telemetry.hpp"
#include "flashtier/workload.hpp"

#if FLASHTIER_HAVE_CUDA
#include "flashtier/backends/unified_memory.hpp"
#endif

namespace flashtier {
namespace bench {

namespace {

using Clock = std::chrono::steady_clock;

std::string temp_store_path(const char* tag) {
    return (std::filesystem::temp_directory_path() /
            (std::string("ft-") + tag + "-store-" + std::to_string(::time(nullptr)) + ".bin"))
        .string();
}

// Resolve the device backend for a benchmark run (same rules as the
// runtime: --backend explicit or automatic with cpu fallback). Never
// throws for "cpu": returns a null backend, which means CPU-only mode.
std::unique_ptr<DeviceBackend> resolve_bench_backend(const cli::Options& o,
                                                     std::string& backend_name,
                                                     std::string& reason) {
    BackendRegistry& registry = BackendRegistry::instance();
    std::string name = o.backend.empty() || o.backend == "auto"
                           ? registry.select_automatic(reason)
                           : o.backend;
    if (!registry.has(name)) {
        throw Error(ErrorCode::Config,
                    "requested backend is not compiled into this build: " + name);
    }
    backend_name = name;
    if (name == "cpu") {
        reason = "cpu backend: CPU-only mode (no accelerator tier)";
        return nullptr;
    }
    return registry.create(name);
}

struct BenchHeader {
    std::string device;
    std::string arch;
    std::string backend;
    std::string os;
    std::string cpu;
};

BenchHeader probe_header(const cli::Options& o) {
    SystemInfo info = probe_system(o.device_id, o.nvme_path);
    BenchHeader h;
    h.os = info.os;
    h.cpu = info.cpu;
    std::string reason;
    std::unique_ptr<DeviceBackend> dev = resolve_bench_backend(o, h.backend, reason);
    if (dev != nullptr) {
        const std::vector<DeviceInfo> devices = dev->enumerate_devices();
        if (!devices.empty()) {
            const DeviceInfo& d = devices[static_cast<std::size_t>(
                std::min(o.device_id, static_cast<int>(devices.size()) - 1))];
            h.device = d.name;
            h.arch = d.architecture;
        } else {
            h.device = "(backend '" + h.backend + "' compiled but no devices)";
        }
    } else {
        h.device = "(no accelerator; CPU-only mode)";
    }
    return h;
}

void print_header(const char* name, const BenchHeader& h, const Config& cfg) {
    std::printf("=== FlashTier benchmark: %s ===\n", name);
    if (h.arch.empty()) {
        std::printf("device: %s\n", h.device.c_str());
    } else {
        std::printf("device: %s (arch %s)\n", h.device.c_str(), h.arch.c_str());
    }
    std::printf("os: %s | cpu: %s\n", h.os.c_str(), h.cpu.c_str());
    std::printf("config:\n%s\n", cfg.describe().c_str());
}

struct EffectiveBudgets {
    std::string backend;       // resolved backend name ("" = none yet)
    std::string device_name;
    uint64_t vram = 0;
    uint64_t host = 0;
    uint64_t nvme = 0;
    uint64_t free_vram = 0;
    uint64_t free_ram = 0;
    uint64_t free_disk = 0;
    bool device_shared_memory = false;
};

// Effective budgets for a benchmark with explicit safe-sizing checks:
// refuses budgets that exceed detected free capacity, and caps automatic
// defaults so local validation stays small and bounded. Explicit
// user-provided sizes (--vram-budget, --host-budget, --nvme-budget) bypass
// the caps but are still validated against free capacity.
EffectiveBudgets resolve_budgets(const cli::Options& o, double vram_fraction,
                                 uint64_t vram_cap_bytes,
                                 uint64_t host_cap_bytes,
                                 uint64_t nvme_cap_bytes) {
    SystemInfo info = probe_system(o.device_id, o.nvme_path);
    EffectiveBudgets b;
    b.free_ram = info.free_ram_bytes;
    b.free_disk = info.disk_free_bytes;

    std::string reason;
    std::unique_ptr<DeviceBackend> dev = resolve_bench_backend(o, b.backend, reason);
    if (dev != nullptr) {
        const std::vector<DeviceInfo> devices = dev->enumerate_devices();
        if (devices.empty()) {
            b.vram = 0;
        } else {
            const int idx = std::min(o.device_id, static_cast<int>(devices.size()) - 1);
            dev->open(idx);
            b.free_vram = dev->free_memory();
            b.device_name = devices[static_cast<std::size_t>(idx)].name;
            b.device_shared_memory = devices[static_cast<std::size_t>(idx)].memory_shared;
            const uint64_t wanted =
                o.vram_budget_bytes != 0
                    ? o.vram_budget_bytes
                    : std::min(
                          static_cast<uint64_t>(static_cast<double>(b.free_vram) * vram_fraction),
                          vram_cap_bytes);
            if (wanted > b.free_vram) {
                throw Error(ErrorCode::Budget,
                            "requested device-memory budget exceeds free device memory",
                            "budget " + std::to_string(wanted) +
                                " free " + std::to_string(b.free_vram));
            }
            b.vram = wanted;
            dev->close();
        }
    } else {
        b.vram = 0;
    }

    b.host = o.host_budget_bytes != 0
                 ? o.host_budget_bytes
                 : std::min(
                       static_cast<uint64_t>(static_cast<double>(b.free_ram) * 0.25),
                       host_cap_bytes);
    if (b.host > b.free_ram) {
        throw Error(ErrorCode::Budget,
                    "requested host budget exceeds free system RAM",
                    "budget " + std::to_string(b.host) +
                        " free " + std::to_string(b.free_ram));
    }

    b.nvme = o.nvme_budget_bytes != 0
                 ? o.nvme_budget_bytes
                 : std::min(static_cast<uint64_t>(static_cast<double>(b.free_disk) * 0.25),
                            nvme_cap_bytes);
    if (b.nvme > b.free_disk) {
        throw Error(ErrorCode::Budget,
                    "requested NVMe budget exceeds free disk", std::to_string(b.nvme));
    }
    return b;
}

Config base_config(const cli::Options& o) { return o.to_config(); }

void emit_begin(Runtime& rt, const std::string& name, const Config& cfg) {
    TelemetryEvent ev;
    ev.type = EventType::BenchmarkBegin;
    ev.reason = name;
    ev.policy = policy_kind_name(cfg.policy);
    ev.logical_size = static_cast<int64_t>(cfg.working_set_bytes);
    rt.emit_telemetry(ev);
}

void emit_end(Runtime& rt, const std::string& name, double throughput) {
    TelemetryEvent ev;
    ev.type = EventType::BenchmarkEnd;
    ev.reason = name;
    ev.throughput = throughput;
    rt.emit_telemetry(ev);
}

std::vector<uint8_t> page_payload(uint64_t page_size, uint64_t seed, uint64_t page_id) {
    std::vector<uint8_t> data(page_size);
    fill_pattern(data.data(), data.size(), seed, page_id);
    return data;
}

double gbps(uint64_t bytes, double seconds) {
    return seconds > 0.0 ? static_cast<double>(bytes) / 1e9 / seconds : 0.0;
}

std::string yes_no_str(bool v) { return v ? "yes" : "no"; }

// Bounded local-validation defaults. Multi-gigabyte hardware runs require
// explicit --working-set / budget flags; defaults never scale with total
// free RAM or VRAM.
constexpr uint64_t kBenchVramCap = 1ull << 30;        // 1 GiB
constexpr uint64_t kBenchHostCap = 512ull << 20;      // 512 MiB
constexpr uint64_t kBenchNvmeCap = 8ull << 30;        // 8 GiB
constexpr uint64_t kBenchCpuHostBudget = 128ull << 20;  // CPU-only host budget
constexpr uint64_t kBenchMaxWorkingSet = 256ull << 20;  // 256 MiB
constexpr uint64_t kBenchMaxExperts = 128;

// Map a typed runtime error to a CLI exit code.
int exit_for_error(const Error& e) {
    switch (e.code()) {
        case ErrorCode::Integrity: return cli::kExitIntegrity;
        case ErrorCode::Unsupported: return cli::kExitUnsupported;
        case ErrorCode::Budget:
        case ErrorCode::Config:
        case ErrorCode::InvalidArgument: return cli::kExitUsage;
        case ErrorCode::StoreCorrupt: return cli::kExitCorrupt;
        case ErrorCode::Cuda:
        case ErrorCode::Io: return cli::kExitFatal;
        default: return cli::kExitError;
    }
}

void report_error(const Error& e) {
    std::fprintf(stderr, "flashtier: %s", e.what());
    if (!e.detail().empty()) std::fprintf(stderr, " (%s)", e.detail().c_str());
    std::fprintf(stderr, "\n");
}

}  // namespace

// ---------------------------------------------------------------------------
// benchmark tiers
// ---------------------------------------------------------------------------

int run_tiers(const cli::Options& o) {
    const BenchHeader h = probe_header(o);
    Config cfg = base_config(o);
    std::printf("=== FlashTier benchmark: tiers ===\n");
    std::printf("device: %s\n", h.device.c_str());
    std::printf("os: %s | cpu: %s\n", h.os.c_str(), h.cpu.c_str());

    const uint64_t buf_bytes = cfg.working_set_bytes != 0
                                   ? cfg.working_set_bytes
                                   : 64ull * 1024 * 1024;
    const uint64_t page_size = cfg.page_size;
    if (buf_bytes % page_size != 0) {
        std::fprintf(stderr, "flashtier: tier buffer must be a multiple of page size\n");
        return cli::kExitUsage;
    }

    struct Row {
        const char* path;
        double gb_s = 0.0;
        double us = 0.0;
    };
    std::vector<Row> rows;

    // 1. host memory copy (pageable)
    {
        std::vector<uint8_t> a(buf_bytes, 0x11), b(buf_bytes, 0x22);
        double best_us = 0.0;
        for (int i = 0; i < 8; ++i) {
            const auto t0 = Clock::now();
            std::memcpy(b.data(), a.data(), buf_bytes);
            const double us =
                std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
            if (i == 0 || us < best_us) best_us = us;
        }
        rows.push_back({"host_memcpy", gbps(buf_bytes, best_us / 1e6), best_us});
    }

    // Device tier rows through the vendor-neutral backend contract.
    std::string backend_name;
    std::string reason;
    std::unique_ptr<DeviceBackend> dev = resolve_bench_backend(o, backend_name, reason);
    if (dev != nullptr) {
        const std::vector<DeviceInfo> devices = dev->enumerate_devices();
        if (devices.empty()) {
            dev.reset();
        }
    }
    if (dev != nullptr) {
        const int idx = std::min(o.device_id, static_cast<int>(dev->enumerate_devices().size()) - 1);
        dev->open(idx);
        DeviceStream* stream = dev->create_stream();
        DeviceEvent* ev_start = dev->create_event();
        DeviceEvent* ev_end = dev->create_event();

        // Pinned host buffer from the backend.
        void* pinned = dev->allocate_host_pinned(buf_bytes);
        std::memset(pinned, 0x33, buf_bytes);

        // 2. pinned host -> device
        {
            void* d = dev->allocate(buf_bytes);
            double best_us = 0.0;
            for (int i = 0; i < 8; ++i) {
                if (dev->capabilities().event_timing) {
                    dev->record_event(ev_start, stream);
                    dev->async_copy_host_to_device(d, pinned, buf_bytes, stream);
                    dev->record_event(ev_end, stream);
                    dev->sync_stream(stream);
                    const double us = dev->event_elapsed_us(ev_start, ev_end);
                    if (i == 0 || us < best_us) best_us = us;
                } else {
                    const auto t0 = Clock::now();
                    dev->async_copy_host_to_device(d, pinned, buf_bytes, stream);
                    dev->sync_stream(stream);
                    const double us =
                        std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
                    if (i == 0 || us < best_us) best_us = us;
                }
            }
            dev->free(d);
            rows.push_back({"pinned_host->device", gbps(buf_bytes, best_us / 1e6), best_us});
        }

        // 3. device -> pinned host
        {
            void* d = dev->allocate(buf_bytes);
            dev->async_copy_host_to_device(d, pinned, buf_bytes, stream);
            dev->sync_stream(stream);
            double best_us = 0.0;
            for (int i = 0; i < 8; ++i) {
                if (dev->capabilities().event_timing) {
                    dev->record_event(ev_start, stream);
                    dev->async_copy_device_to_host(pinned, d, buf_bytes, stream);
                    dev->record_event(ev_end, stream);
                    dev->sync_stream(stream);
                    const double us = dev->event_elapsed_us(ev_start, ev_end);
                    if (i == 0 || us < best_us) best_us = us;
                } else {
                    const auto t0 = Clock::now();
                    dev->async_copy_device_to_host(pinned, d, buf_bytes, stream);
                    dev->sync_stream(stream);
                    const double us =
                        std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
                    if (i == 0 || us < best_us) best_us = us;
                }
            }
            dev->free(d);
            rows.push_back({"device->pinned_host", gbps(buf_bytes, best_us / 1e6), best_us});
        }

        // NVMe store for the remaining paths.
        const std::string store_path = o.nvme_path.empty()
                                           ? temp_store_path("tiers")
                                           : o.nvme_path + "/ft-tiers-store.bin";
        NvmeBackend store;
        store.open(store_path, std::max<uint64_t>(buf_bytes * 2, 64ull * 1024 * 1024),
                   page_size);
        const uint64_t k_extents = buf_bytes / page_size;
        std::vector<uint64_t> region;
        region.reserve(k_extents);
        for (uint64_t i = 0; i < k_extents; ++i) {
            uint64_t off = 0;
            if (!store.allocate_extent(off)) {
                std::fprintf(stderr, "flashtier: store too small for tier benchmark\n");
                return cli::kExitUsage;
            }
            region.push_back(off);
        }
        std::sort(region.begin(), region.end());
        const uint64_t base = region.front();

        // 4. host -> NVMe
        {
            double best_us = 0.0;
            for (int i = 0; i < 8; ++i) {
                const auto t0 = Clock::now();
                auto op = store.write_async(base, pinned, buf_bytes);
                op->wait();
                const double us =
                    std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
                if (i == 0 || us < best_us) best_us = us;
            }
            rows.push_back({"host->nvme", gbps(buf_bytes, best_us / 1e6), best_us});
        }

        // 5. NVMe -> host
        {
            double best_us = 0.0;
            for (int i = 0; i < 8; ++i) {
                const auto t0 = Clock::now();
                auto op = store.read_async(base, pinned, buf_bytes);
                op->wait();
                const double us =
                    std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
                if (i == 0 || us < best_us) best_us = us;
            }
            rows.push_back({"nvme->host", gbps(buf_bytes, best_us / 1e6), best_us});
        }

        // 6. full NVMe -> host -> device
        {
            void* d = dev->allocate(buf_bytes);
            double best_us = 0.0;
            for (int i = 0; i < 8; ++i) {
                const auto t0 = Clock::now();
                auto op = store.read_async(base, pinned, buf_bytes);
                op->wait();
                dev->async_copy_host_to_device(d, pinned, buf_bytes, stream);
                dev->sync_stream(stream);
                const double us =
                    std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
                if (i == 0 || us < best_us) best_us = us;
            }
            dev->free(d);
            rows.push_back({"nvme->host->device (full)", gbps(buf_bytes, best_us / 1e6),
                            best_us});
        }

        // 7. full device -> host -> NVMe
        {
            void* d = dev->allocate(buf_bytes);
            dev->async_copy_host_to_device(d, pinned, buf_bytes, stream);
            dev->sync_stream(stream);
            double best_us = 0.0;
            for (int i = 0; i < 8; ++i) {
                const auto t0 = Clock::now();
                dev->async_copy_device_to_host(pinned, d, buf_bytes, stream);
                dev->sync_stream(stream);
                auto op = store.write_async(base, pinned, buf_bytes);
                op->wait();
                const double us =
                    std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
                if (i == 0 || us < best_us) best_us = us;
            }
            dev->free(d);
            rows.push_back({"device->host->nvme (full)", gbps(buf_bytes, best_us / 1e6),
                            best_us});
        }

        store.close();
        NvmeBackend::destroy_file(store_path);
        dev->free_host_pinned(pinned);
        dev->destroy_event(ev_end);
        dev->destroy_event(ev_start);
        dev->destroy_stream(stream);
        dev->close();
    } else
    {
        // CPU-only (or --no-cuda): the NVMe rows still measure real storage;
        // the CUDA rows report an explicit unsupported result.
        const std::string store_path = o.nvme_path.empty()
                                           ? temp_store_path("tiers")
                                           : o.nvme_path + "/ft-tiers-store.bin";
        NvmeBackend store;
        store.open(store_path, std::max<uint64_t>(buf_bytes * 2, 64ull * 1024 * 1024),
                   page_size);
        const uint64_t k_extents = buf_bytes / page_size;
        std::vector<uint64_t> region;
        region.reserve(k_extents);
        for (uint64_t i = 0; i < k_extents; ++i) {
            uint64_t off = 0;
            if (!store.allocate_extent(off)) {
                std::fprintf(stderr, "flashtier: store too small for tier benchmark\n");
                return cli::kExitUsage;
            }
            region.push_back(off);
        }
        std::sort(region.begin(), region.end());
        const uint64_t base = region.front();
        std::vector<uint8_t> buf(buf_bytes, 0x44);
        {
            double best_us = 0.0;
            for (int i = 0; i < 8; ++i) {
                const auto t0 = Clock::now();
                auto op = store.write_async(base, buf.data(), buf_bytes);
                op->wait();
                const double us =
                    std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
                if (i == 0 || us < best_us) best_us = us;
            }
            rows.push_back({"host->nvme", gbps(buf_bytes, best_us / 1e6), best_us});
        }
        {
            double best_us = 0.0;
            for (int i = 0; i < 8; ++i) {
                const auto t0 = Clock::now();
                auto op = store.read_async(base, buf.data(), buf_bytes);
                op->wait();
                const double us =
                    std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
                if (i == 0 || us < best_us) best_us = us;
            }
            rows.push_back({"nvme->host", gbps(buf_bytes, best_us / 1e6), best_us});
        }
        store.close();
        NvmeBackend::destroy_file(store_path);

        rows.push_back({"pinned_host->vram", 0.0, 0.0});
        rows.push_back({"vram->pinned_host", 0.0, 0.0});
        rows.push_back({"nvme->host->vram (full)", 0.0, 0.0});
        rows.push_back({"vram->host->nvme (full)", 0.0, 0.0});
    }

    std::printf("\nTier bandwidth (best of 8, buffer %zu bytes):\n",
                static_cast<std::size_t>(buf_bytes));
    for (const auto& r : rows) {
        if (r.gb_s > 0.0) {
            std::printf("  %-24s %8.2f GB/s  (%10.1f us)\n", r.path, r.gb_s, r.us);
        } else {
            std::printf("  %-24s unsupported (no accelerator backend)\n", r.path);
        }
    }
    std::printf("Note: full-path rows include host staging; NVMe numbers depend on the backing volume.\n");
    return cli::kExitOk;
}

// ---------------------------------------------------------------------------
// benchmark oversubscription
// ---------------------------------------------------------------------------

int run_oversubscription(const cli::Options& o) {
    EffectiveBudgets b =
        resolve_budgets(o, 0.50, kBenchVramCap, kBenchHostCap, kBenchNvmeCap);
    if (b.vram == 0) {
        std::printf("=== FlashTier benchmark: oversubscription ===\n");
        std::printf("note: no accelerator backend available; oversubscription runs against the host budget\n");
        // CPU-only default: small bounded host budget so the working set
        // genuinely oversubscribes it and spills to NVMe.
        b.host = std::min(b.host, kBenchCpuHostBudget);
    }

    Config cfg = base_config(o);
    if (!FLASHTIER_HAVE_CUDA) cfg.cuda_enabled = false;  // CPU-only builds run host/NVMe tiers
    if (o.vram_budget_bytes == 0 && b.vram != 0) cfg.vram_budget_bytes = b.vram;
    if (o.host_budget_bytes == 0) cfg.host_budget_bytes = b.host;
    if (o.nvme_budget_bytes == 0) cfg.nvme_budget_bytes = b.nvme;

    const BenchHeader h = probe_header(o);
    print_header("oversubscription", h, cfg);

    const auto validation = validate_config(cfg, BackendRegistry::instance().compiled_backends());
    if (!validation.ok) {
        for (const auto& e : validation.errors) {
            std::fprintf(stderr, "config: %s\n", e.c_str());
        }
        return cli::kExitUsage;
    }

    Runtime rt(cfg);
    rt.start();
    emit_begin(rt, "oversubscription", cfg);

    const uint64_t base_tier = b.vram != 0 ? b.vram : b.host;
    const double ratios[] = {1.00, 1.25, 1.50, 2.00};

    uint64_t total_ops = 0;
    double total_seconds = 0.0;
    int rc = cli::kExitOk;
    for (const double ratio : ratios) {
        const uint64_t ws = static_cast<uint64_t>(static_cast<double>(base_tier) * ratio);
        const uint64_t ws_pages = ws / cfg.page_size;
        const uint64_t logical_ws = ws_pages * cfg.page_size;
        if (ws_pages == 0) continue;

        // Safety: the full working set must fit host + NVMe budgets.
        const uint64_t spill = logical_ws > b.vram ? logical_ws - b.vram : 0;
        if (spill > b.host + b.nvme) {
            std::printf("phase ratio=%.2f: skipped (working set %s does not fit host+NVMe budgets)\n",
                        ratio, bytesize_to_string(logical_ws).c_str());
            continue;
        }

        std::printf("phase ratio=%.2f logical=%s pages=%llu\n", ratio,
                    bytesize_to_string(logical_ws).c_str(),
                    static_cast<unsigned long long>(ws_pages));

        std::vector<PageId> ids;
        ids.reserve(ws_pages);
        for (uint64_t i = 0; i < ws_pages; ++i) {
            const SemanticClass cls = (i % 16) == 0   ? SemanticClass::KvCache
                                      : (i % 16) == 8 ? SemanticClass::MoeExpert
                                                      : SemanticClass::Generic;
            ids.push_back(rt.allocate_page(cfg.page_size, cls));
        }

        for (PageId id : ids) {
            const auto payload = page_payload(cfg.page_size, cfg.seed, id.value);
            rt.write_page(id, payload.data());
        }

        TraceGenerator trace(cfg.seed + static_cast<uint64_t>(
                                             static_cast<uint64_t>(ratio * 1000.0)),
                             ws_pages, ws_pages * 8, TracePattern::Skewed, 1.2);
        const auto t0 = Clock::now();
        std::vector<uint8_t> buf(cfg.page_size);
        for (const auto& a : trace.accesses()) {
            PageId id{ids[a.page_id]};
            if (a.write) {
                const auto payload = page_payload(cfg.page_size, cfg.seed, id.value);
                rt.write_page(id, payload.data());
            } else {
                rt.read_page(id, buf.data());
            }
        }
        const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
        total_ops += trace.accesses().size();
        total_seconds += secs;

        uint64_t verified = 0;
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (i < 32 || i % 97 == 0) {
                rt.verify_page(ids[i]);
                ++verified;
            }
        }
        const auto agg = rt.aggregates();
        std::printf("  accesses=%llu throughput=%.1f ops/s\n",
                    static_cast<unsigned long long>(trace.accesses().size()),
                    static_cast<double>(trace.accesses().size()) / secs);
        std::printf("  demand_faults=%llu vram_hits=%.1f%% host_hits=%.1f%% nvme_hits=%.1f%%\n",
                    static_cast<unsigned long long>(agg.demand_faults),
                    agg.vram_hit_rate() * 100.0, agg.host_hit_rate() * 100.0,
                    agg.nvme_hit_rate() * 100.0);
        std::printf("  evictions=%llu writebacks=%llu stalled=%.1f ms prefetch_hit_rate=%.1f%%\n",
                    static_cast<unsigned long long>(agg.evictions),
                    static_cast<unsigned long long>(agg.writebacks),
                    agg.total_stall_us / 1000.0, agg.prefetch_hit_rate() * 100.0);
        std::printf("  integrity: %llu pages verified, %llu failures\n",
                    static_cast<unsigned long long>(verified),
                    static_cast<unsigned long long>(agg.integrity_failures));
        if (agg.integrity_failures != 0) {
            std::fprintf(stderr, "flashtier: integrity failure in oversubscription phase\n");
            rc = cli::kExitIntegrity;
        }

        for (PageId id : ids) rt.free_page(id);
    }

    const auto agg = rt.aggregates();
    std::printf("\ntelemetry summary:\n%s", format_aggregates(agg).c_str());
    emit_end(rt, "oversubscription",
             total_seconds > 0.0 ? static_cast<double>(total_ops) / total_seconds : 0.0);
    rt.shutdown();
    if (rc != cli::kExitOk) return rc;
    std::printf("oversubscription: OK (working sets exceeded the %s budget without failure)\n",
                bytesize_to_string(b.vram != 0 ? b.vram : b.host).c_str());
    return cli::kExitOk;
}

// ---------------------------------------------------------------------------
// benchmark prefetch
// ---------------------------------------------------------------------------

int run_prefetch(const cli::Options& o) {
    EffectiveBudgets b =
        resolve_budgets(o, 0.50, kBenchVramCap, kBenchHostCap, kBenchNvmeCap);
    // CPU-only default: the host budget is the hot tier and stays small so
    // the working set genuinely spills to NVMe and prefetch has real work.
    const uint64_t base_tier =
        b.vram != 0 ? b.vram : std::min(b.host, kBenchCpuHostBudget);

    Config cfg = base_config(o);
    if (!FLASHTIER_HAVE_CUDA) cfg.cuda_enabled = false;  // CPU-only builds run host/NVMe tiers
    if (o.vram_budget_bytes == 0 && b.vram != 0) cfg.vram_budget_bytes = b.vram;
    if (o.host_budget_bytes == 0) cfg.host_budget_bytes = b.vram != 0 ? b.host : base_tier;
    if (o.nvme_budget_bytes == 0) cfg.nvme_budget_bytes = b.nvme;

    const BenchHeader h = probe_header(o);
    print_header("prefetch", h, cfg);

    // Bounded default: at most 256 MiB; larger runs require --working-set.
    const uint64_t ws = o.working_set_bytes != 0
                            ? o.working_set_bytes
                            : std::min(base_tier * 2, kBenchMaxWorkingSet);
    const uint64_t ws_pages = ws / cfg.page_size;
    const uint64_t logical_ws = ws_pages * cfg.page_size;
    if (ws_pages == 0) {
        std::fprintf(stderr, "flashtier: working set too small for page size\n");
        return cli::kExitUsage;
    }
    if (logical_ws > b.host + b.nvme) {
        std::fprintf(stderr,
                     "flashtier: working set %s does not fit host+NVMe budgets (%s)\n",
                     bytesize_to_string(logical_ws).c_str(),
                     bytesize_to_string(b.host + b.nvme).c_str());
        return cli::kExitUsage;
    }

    // Partially predictable trace: 75% sequential + 25% random bursts.
    const uint64_t ops = ws_pages * 8;
    std::vector<uint64_t> trace;
    trace.reserve(ops);
    {
        SplitMix64 rng(cfg.seed);
        uint64_t pos = rng.next() % ws_pages;
        for (uint64_t i = 0; i < ops; ++i) {
            if ((rng.next() & 3) != 0) {
                trace.push_back(pos);
                pos = (pos + 1) % ws_pages;
            } else {
                pos = rng.next() % ws_pages;
                trace.push_back(pos);
                pos = (pos + 1) % ws_pages;
            }
        }
    }
    const auto annotated = annotate_next_use(trace);

    struct RunResult {
        const char* label;
        double ops_per_s = 0.0;
        double stall_ms = 0.0;
        uint64_t demand_faults = 0;
        uint64_t prefetch_issued = 0;
        uint64_t prefetch_hits = 0;
        uint64_t prefetch_waste = 0;
    };

    const int modes = 3;
    const char* labels[modes] = {"off", "sequential", "predictive"};
    RunResult results[modes];

    for (int m = 0; m < modes; ++m) {
        Runtime rt(cfg);
        rt.start();
        emit_begin(rt, "prefetch/" + std::string(labels[m]), cfg);

        std::vector<PageId> ids;
        ids.reserve(ws_pages);
        for (uint64_t i = 0; i < ws_pages; ++i) {
            ids.push_back(rt.allocate_page(cfg.page_size));
        }
        for (PageId id : ids) {
            const auto payload = page_payload(cfg.page_size, cfg.seed, id.value);
            rt.write_page(id, payload.data());
        }

        const uint32_t depth = cfg.prefetch_depth;
        const auto t0 = Clock::now();
        std::vector<uint8_t> buf(cfg.page_size);
        for (std::size_t i = 0; i < annotated.size(); ++i) {
            if (m == 1) {
                // Sequential: prefetch the next `depth` pages after the
                // current position every `depth` accesses.
                if (i % depth == 0) {
                    const uint64_t start_p = annotated[i].page_id;
                    std::vector<PageId> ahead;
                    ahead.reserve(depth);
                    for (uint32_t k = 1; k <= depth && start_p + k < ws_pages; ++k) {
                        ahead.push_back(ids[start_p + k]);
                    }
                    rt.prefetch(ahead);
                }
            } else if (m == 2) {
                // Predictive: the next trace entry is the page we will touch
                // next; when its annotated reuse distance is small, prefetch
                // it before we reach it.
                if (i + 1 < annotated.size()) {
                    const auto& next = annotated[i + 1];
                    if (next.next_use_distance > 0 && next.next_use_distance <= depth) {
                        rt.prefetch({ids[next.page_id]});
                    }
                }
            }
            rt.read_page(ids[annotated[i].page_id], buf.data());
        }
        const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
        const auto agg = rt.aggregates();

        results[m].ops_per_s = static_cast<double>(annotated.size()) / secs;
        results[m].stall_ms = agg.total_stall_us / 1000.0;
        results[m].demand_faults = agg.demand_faults;
        results[m].prefetch_issued = agg.prefetches_issued;
        results[m].prefetch_hits = agg.prefetch_hits;
        results[m].prefetch_waste = agg.prefetch_waste;

        uint64_t verified = 0;
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (i < 16 || i % 131 == 0) {
                rt.verify_page(ids[i]);
                ++verified;
            }
        }
        std::printf("prefetch=%s: throughput=%.1f ops/s stalls=%.1fms faults=%llu "
                    "prefetch issued=%llu hits=%llu waste=%llu integrity=%llu ok\n",
                    labels[m], results[m].ops_per_s, results[m].stall_ms,
                    static_cast<unsigned long long>(results[m].demand_faults),
                    static_cast<unsigned long long>(results[m].prefetch_issued),
                    static_cast<unsigned long long>(results[m].prefetch_hits),
                    static_cast<unsigned long long>(results[m].prefetch_waste),
                    static_cast<unsigned long long>(verified));
        if (agg.integrity_failures != 0) {
            rt.shutdown();
            return cli::kExitIntegrity;
        }
        emit_end(rt, "prefetch/" + std::string(labels[m]), results[m].ops_per_s);
        for (PageId id : ids) rt.free_page(id);
        rt.shutdown();
    }

    std::printf("\nprefetch comparison:\n");
    std::printf("  %-12s %12s %10s %8s %10s %10s\n", "mode", "ops/s", "stall(ms)",
                "faults", "pf hits", "pf waste");
    for (int m = 0; m < modes; ++m) {
        std::printf("  %-12s %12.1f %10.1f %8llu %10llu %10llu\n", labels[m],
                    results[m].ops_per_s, results[m].stall_ms,
                    static_cast<unsigned long long>(results[m].demand_faults),
                    static_cast<unsigned long long>(results[m].prefetch_hits),
                    static_cast<unsigned long long>(results[m].prefetch_waste));
    }
    return cli::kExitOk;
}

// ---------------------------------------------------------------------------
// benchmark sparse-experts
// ---------------------------------------------------------------------------

int run_sparse_experts(const cli::Options& o) {
    EffectiveBudgets b =
        resolve_budgets(o, 0.50, kBenchVramCap, kBenchHostCap, kBenchNvmeCap);
    const uint64_t base_tier =
        b.vram != 0 ? b.vram : std::min(b.host, kBenchCpuHostBudget);

    Config cfg = base_config(o);
    if (!FLASHTIER_HAVE_CUDA) cfg.cuda_enabled = false;  // CPU-only builds run host/NVMe tiers
    if (o.vram_budget_bytes == 0 && b.vram != 0) cfg.vram_budget_bytes = b.vram;
    if (o.host_budget_bytes == 0) cfg.host_budget_bytes = b.vram != 0 ? b.host : base_tier;
    if (o.nvme_budget_bytes == 0) cfg.nvme_budget_bytes = b.nvme;

    const BenchHeader h = probe_header(o);
    print_header("sparse-experts", h, cfg);

    // Bounded synthetic workload; larger hardware runs need explicit sizes.
    const uint64_t expert_count = std::min<uint64_t>(
        kBenchMaxExperts, std::max<uint64_t>(16, (base_tier / cfg.page_size) * 2));
    const uint64_t hot_count = std::max<uint64_t>(1, expert_count / 16);
    const uint64_t active_per_token = 2;
    const uint64_t tokens = std::min<uint64_t>(
        2048, std::max<uint64_t>(1, (expert_count * 8) / active_per_token));

    if (expert_count * cfg.page_size > b.host + b.nvme) {
        std::fprintf(stderr, "flashtier: expert working set exceeds host+NVMe budgets\n");
        return cli::kExitUsage;
    }

    std::printf("experts=%llu hot=%llu active_per_token=%llu tokens=%llu zipf=1.2\n",
                static_cast<unsigned long long>(expert_count),
                static_cast<unsigned long long>(hot_count),
                static_cast<unsigned long long>(active_per_token),
                static_cast<unsigned long long>(tokens));

    const auto trace =
        generate_moe_trace(cfg.seed, expert_count, active_per_token, hot_count, 1, 1.2, tokens);
    const auto annotated = annotate_next_use(trace);

    struct RunResult {
        const char* label;
        double ops_per_s = 0.0;
        double stall_ms = 0.0;
        uint64_t faults = 0;
        double vram_hit_rate = 0.0;
    };

    const int modes = 2;
    const char* labels[modes] = {"no_residency_hint", "hot_expert_staging"};
    RunResult results[modes];

    for (int m = 0; m < modes; ++m) {
        Runtime rt(cfg);
        rt.start();
        emit_begin(rt, "sparse-experts/" + std::string(labels[m]), cfg);

        std::vector<PageId> ids;
        ids.reserve(expert_count);
        for (uint64_t i = 0; i < expert_count; ++i) {
            ids.push_back(rt.allocate_page(cfg.page_size, SemanticClass::MoeExpert));
        }
        for (PageId id : ids) {
            const auto payload = page_payload(cfg.page_size, cfg.seed, id.value);
            rt.write_page(id, payload.data());
        }

        if (m == 1) {
            for (uint64_t e = 0; e < hot_count; ++e) {
                rt.pin(ids[e], true);
                rt.promote(ids[e]);
            }
        }

        const auto t0 = Clock::now();
        std::vector<uint8_t> buf(cfg.page_size);
        for (std::size_t i = 0; i < annotated.size(); ++i) {
            if (m == 1 && i + 1 < annotated.size()) {
                const auto& next = annotated[i + 1];
                if (next.next_use_distance > 0 && next.next_use_distance <= 4) {
                    rt.prefetch({ids[next.page_id]});
                }
            }
            rt.read_page(ids[annotated[i].page_id], buf.data());
        }
        const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
        const auto agg = rt.aggregates();

        results[m].ops_per_s = static_cast<double>(annotated.size()) / secs;
        results[m].stall_ms = agg.total_stall_us / 1000.0;
        results[m].faults = agg.demand_faults;
        results[m].vram_hit_rate = agg.vram_hit_rate();

        uint64_t verified = 0;
        for (std::size_t i = 0; i < ids.size(); i += 7) {
            rt.verify_page(ids[i]);
            ++verified;
        }
        std::printf("mode=%s: throughput=%.1f ops/s stalls=%.1fms faults=%llu "
                    "vram_hit=%.1f%% integrity=%llu ok\n",
                    labels[m], results[m].ops_per_s, results[m].stall_ms,
                    static_cast<unsigned long long>(results[m].faults),
                    results[m].vram_hit_rate * 100.0,
                    static_cast<unsigned long long>(verified));
        if (agg.integrity_failures != 0) {
            rt.shutdown();
            return cli::kExitIntegrity;
        }
        emit_end(rt, "sparse-experts/" + std::string(labels[m]), results[m].ops_per_s);
        for (PageId id : ids) rt.free_page(id);
        rt.shutdown();
    }

    std::printf("\nsparse-expert comparison (synthetic MoE trace; not a transformer engine):\n");
    std::printf("  %-24s %10s %10s %8s %10s\n", "mode", "ops/s", "stall(ms)", "faults",
                "vram_hit");
    for (int m = 0; m < modes; ++m) {
        std::printf("  %-24s %10.1f %10.1f %8llu %9.1f%%\n", labels[m],
                    results[m].ops_per_s, results[m].stall_ms,
                    static_cast<unsigned long long>(results[m].faults),
                    results[m].vram_hit_rate * 100.0);
    }
    return cli::kExitOk;
}

// ---------------------------------------------------------------------------
// benchmark unified-memory
// ---------------------------------------------------------------------------

int run_unified_memory(const cli::Options& o) {
#if FLASHTIER_HAVE_CUDA
    // This comparison is CUDA-specific by definition (cudaMallocManaged
    // vs FlashTier's explicit strategy); it requires the cuda backend.
    std::string backend_name;
    std::string reason;
    std::unique_ptr<DeviceBackend> dev = resolve_bench_backend(o, backend_name, reason);
    if (dev == nullptr || backend_name != "cuda") {
        std::printf("=== FlashTier benchmark: unified-memory ===\n");
        std::printf("unsupported: requires the cuda backend with a GPU (selected: %s)\n",
                    backend_name.c_str());
        return cli::kExitUnsupported;
    }
    const std::vector<DeviceInfo> devices = dev->enumerate_devices();
    if (devices.empty()) {
        std::printf("=== FlashTier benchmark: unified-memory ===\n");
        std::printf("unsupported: cuda backend compiled but no device available\n");
        return cli::kExitUnsupported;
    }
    const int idx = std::min(o.device_id, static_cast<int>(devices.size()) - 1);
    dev->open(idx);
    const DeviceCapabilities caps = dev->capabilities();
    const DeviceInfo& gpu = devices[static_cast<std::size_t>(idx)];
    dev->close();
    if (!caps.unified_memory) {
        std::printf("=== FlashTier benchmark: unified-memory ===\n");
        std::printf("unsupported: device %s reports no unified memory support\n",
                    gpu.name.c_str());
        return cli::kExitUnsupported;
    }

    const uint64_t vram_budget = o.vram_budget_bytes != 0
                                     ? o.vram_budget_bytes
                                     : std::min(
                                           static_cast<uint64_t>(static_cast<double>(gpu.free_memory) * 0.50),
                                           kBenchVramCap);
    // Bounded default (2 GiB max); larger runs require explicit sizes.
    const uint64_t ws = o.working_set_bytes != 0 ? o.working_set_bytes : vram_budget * 2;
    if (ws > gpu.free_memory) {
        std::fprintf(stderr,
                     "flashtier: unified-memory working set %s exceeds free device memory %s; "
                     "this benchmark is bounded and refuses unsafe sizes\n",
                     bytesize_to_string(ws).c_str(),
                     bytesize_to_string(gpu.free_memory).c_str());
        return cli::kExitUsage;
    }

    std::printf("=== FlashTier benchmark: unified-memory ===\n");
    std::printf("device: %s (arch %s, UM=%s, CAM=%s)\n", gpu.name.c_str(),
                gpu.architecture.c_str(), yes_no_str(caps.unified_memory).c_str(),
                yes_no_str(caps.concurrent_managed_access).c_str());
    std::printf("working set: %s (2x of %s device-memory budget)\n",
                bytesize_to_string(ws).c_str(), bytesize_to_string(vram_budget).c_str());
    std::printf("page size: %s, prefetch distance: %llu pages, iterations: %u, seed: %llu\n",
                bytesize_to_string(o.page_size).c_str(),
                static_cast<unsigned long long>(o.um_prefetch_pages), o.iterations,
                static_cast<unsigned long long>(o.seed));

    UmResult um = run_unified_memory_benchmark(o.device_id, ws, o.page_size,
                                               o.um_prefetch_pages, o.iterations, o.seed);
    std::printf("\n[driver-managed cudaMallocManaged + cudaMemPrefetchAsync]\n");
    std::printf("  ops=%llu ops/s=%.1f duration=%.2fs moved=%s bandwidth=%.2f GB/s mismatches=%llu\n",
                static_cast<unsigned long long>(um.ops), um.ops_per_s, um.duration_s,
                bytesize_to_string(um.bytes_moved).c_str(), um.bandwidth_gb_s,
                static_cast<unsigned long long>(um.mismatches));
    std::printf("  driver hints: memory advice %s, prefetch %s "
                "(WDDM drivers may reject these even when managed memory is reported)\n",
                yes_no_str(um.advice_used).c_str(), yes_no_str(um.prefetch_used).c_str());
    if (um.mismatches != 0) {
        std::fprintf(stderr, "flashtier: unified memory benchmark found %llu mismatches\n",
                     static_cast<unsigned long long>(um.mismatches));
        return cli::kExitIntegrity;
    }

    // Explicit FlashTier run over the same geometry.
    Config cfg = base_config(o);
    cfg.vram_budget_bytes = vram_budget;
    cfg.working_set_bytes = ws;
    cfg.prefetch = PrefetchKind::Sequential;
    cfg.prefetch_depth = static_cast<uint32_t>(o.um_prefetch_pages);
    cfg.retain_store = false;
    const auto validation = validate_config(cfg, BackendRegistry::instance().compiled_backends());
    if (!validation.ok) return cli::kExitUsage;

    const uint64_t ws_pages = ws / cfg.page_size;
    Runtime rt(cfg);
    rt.start();
    emit_begin(rt, "unified-memory/explicit", cfg);
    std::vector<PageId> ids;
    ids.reserve(ws_pages);
    for (uint64_t i = 0; i < ws_pages; ++i) {
        ids.push_back(rt.allocate_page(cfg.page_size));
    }
    for (uint64_t it = 0; it < cfg.iterations; ++it) {
        for (uint64_t p = 0; p < ws_pages; ++p) {
            const auto payload = page_payload(cfg.page_size, cfg.seed, ids[p].value);
            rt.write_page(ids[p], payload.data());
            if (p + 1 < ws_pages) {
                // Clamp the sequential prefetch to the valid page range.
                const uint64_t remaining = ws_pages - (p + 2) + 1;
                const uint64_t count =
                    std::min<uint64_t>(cfg.prefetch_depth - 1, remaining);
                rt.prefetch_sequential(ids[p + 1], count);
            }
        }
    }
    uint64_t mismatches = 0;
    for (uint64_t it = 0; it < cfg.iterations; ++it) {
        std::vector<uint8_t> buf(cfg.page_size);
        for (uint64_t p = 0; p < ws_pages; ++p) {
            rt.read_page(ids[p], buf.data());
            if (verify_pattern(buf.data(), buf.size(), cfg.seed, ids[p].value, 64)
                    .has_value()) {
                ++mismatches;
            }
        }
    }
    const auto agg = rt.aggregates();
    const uint64_t explicit_ops = ws_pages * cfg.iterations;
    const double explicit_dur_s =
        agg.total_stall_us > 0 ? agg.total_stall_us / 1e6 : 1.0;
    std::printf("\n[FlashTier explicit VRAM/host/NVMe strategy, same geometry]\n");
    std::printf("  ops=%llu ops/s=%.1f demand_faults=%llu evictions=%llu writebacks=%llu "
                "mismatches=%llu\n",
                static_cast<unsigned long long>(explicit_ops),
                static_cast<double>(explicit_ops) / explicit_dur_s,
                static_cast<unsigned long long>(agg.demand_faults),
                static_cast<unsigned long long>(agg.evictions),
                static_cast<unsigned long long>(agg.writebacks),
                static_cast<unsigned long long>(mismatches));

    std::printf("\ncomparison scope: same machine, same working-set geometry, same page size, ");
    std::printf("same deterministic touch pattern. Mechanisms differ: driver-managed paging vs ");
    std::printf("FlashTier's explicit transfers. These numbers are not a latency-equivalence claim.\n");
    if (mismatches != 0) {
        rt.shutdown();
        return cli::kExitIntegrity;
    }
    emit_end(rt, "unified-memory/explicit",
             static_cast<double>(explicit_ops) / explicit_dur_s);
    for (PageId id : ids) rt.free_page(id);
    rt.shutdown();
    return cli::kExitOk;
#else
    (void)o;
    std::printf("=== FlashTier benchmark: unified-memory ===\n");
    std::printf("unsupported: requires the cuda backend compiled into this build\n");
    return cli::kExitUnsupported;
#endif
}

// ---------------------------------------------------------------------------
// dispatch
// ---------------------------------------------------------------------------

int run_benchmark_command(const cli::Options& o) {
    try {
        if (o.subcommand == "tiers") return run_tiers(o);
        if (o.subcommand == "oversubscription") return run_oversubscription(o);
        if (o.subcommand == "prefetch") return run_prefetch(o);
        if (o.subcommand == "sparse-experts") return run_sparse_experts(o);
        if (o.subcommand == "unified-memory") return run_unified_memory(o);
        return run_tiers(o);  // bare `flashtier benchmark`
    } catch (const Error& e) {
        report_error(e);
        return exit_for_error(e);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "flashtier: benchmark failed: %s\n", e.what());
        return cli::kExitError;
    }
}

int guard(const cli::Options& o, int (*fn)(const cli::Options&)) {
    try {
        return fn(o);
    } catch (const Error& e) {
        report_error(e);
        return exit_for_error(e);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "flashtier: benchmark failed: %s\n", e.what());
        return cli::kExitError;
    }
}

}  // namespace bench
}  // namespace flashtier
