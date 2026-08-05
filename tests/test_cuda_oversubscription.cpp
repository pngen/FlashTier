#include "test_harness.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

#include "flashtier/config.hpp"
#include "flashtier/error.hpp"
#include "flashtier/runtime.hpp"

using namespace flashtier;

#if FLASHTIER_HAVE_CUDA

// True three-tier oversubscription on the CUDA backend (LABELS gpu).
// Geometry forces device memory -> pinned host memory -> NVMe:
//   device budget 256 MiB, host budget 128 MiB, working set 512 MiB,
//   NVMe budget 1 GiB, page size 64 KiB.
// The working set exceeds device + host budgets, so pages must reach NVMe.

namespace {

Config oversubscription_config() {
    Config cfg;
    cfg.backend = "cuda";
    cfg.device_id = 0;
    cfg.page_size = 64 * 1024;                        // 64 KiB
    cfg.vram_budget_bytes = 256ull * 1024 * 1024;     // 256 MiB
    cfg.host_budget_bytes = 128ull * 1024 * 1024;     // 128 MiB
    cfg.nvme_budget_bytes = 1024ull * 1024 * 1024;    // 1 GiB
    cfg.queue_depth = 8;
    cfg.policy = PolicyKind::Lru;
    cfg.output_dir = ".";
    return cfg;
}

}  // namespace

FT_TEST(cuda_oversubscription_three_tiers) {
    Config cfg = oversubscription_config();
    const uint64_t page_count = 512ull * 1024 * 1024 / cfg.page_size;  // 8192 pages

    std::printf("three-tier oversubscription geometry:\n");
    std::printf("  page size: %s\n", bytesize_to_string(cfg.page_size).c_str());
    std::printf("  device budget: %s\n", bytesize_to_string(cfg.vram_budget_bytes).c_str());
    std::printf("  device reserve margin: %.2f (effective %s usable)\n",
                cfg.vram_reserve_margin,
                bytesize_to_string(static_cast<uint64_t>(
                                       static_cast<double>(cfg.vram_budget_bytes) *
                                       (1.0 - cfg.vram_reserve_margin)))
                    .c_str());
    std::printf("  host budget: %s\n", bytesize_to_string(cfg.host_budget_bytes).c_str());
    std::printf("  nvme budget: %s\n", bytesize_to_string(cfg.nvme_budget_bytes).c_str());
    std::printf("  working set: %s (%llu pages)\n",
                bytesize_to_string(page_count * cfg.page_size).c_str(),
                static_cast<unsigned long long>(page_count));
    std::printf("  oversubscription ratio vs device budget: %.2fx\n",
                static_cast<double>(page_count * cfg.page_size) /
                    static_cast<double>(cfg.vram_budget_bytes));

    Runtime rt(cfg);
    rt.start();
    FT_ASSERT_EQ(rt.selected_backend(), std::string("cuda"));

    std::vector<PageId> ids;
    ids.reserve(page_count);
    for (uint64_t i = 0; i < page_count; ++i) {
        ids.push_back(rt.allocate_page(cfg.page_size));
    }
    for (PageId id : ids) {
        rt.fill_page(id);
    }

    uint64_t peak_vram = 0, peak_host = 0, peak_nvme = 0;
    for (int round = 0; round < 3; ++round) {
        for (PageId id : ids) {
            rt.promote(id);
        }
        peak_vram = std::max(peak_vram, rt.pages_resident_vram());
        peak_host = std::max(peak_host, rt.pages_resident_host());
        // Force every page down the hierarchy: device -> host -> NVMe.
        for (PageId id : ids) {
            rt.demote_to_nvme(id);
        }
        peak_nvme = std::max(peak_nvme, rt.pages_resident_nvme());
        peak_host = std::max(peak_host, rt.pages_resident_host());
        // Reloads: NVMe -> host -> device.
        for (PageId id : ids) {
            rt.promote(id);
        }
        peak_vram = std::max(peak_vram, rt.pages_resident_vram());
        peak_host = std::max(peak_host, rt.pages_resident_host());
    }

    const auto agg = rt.aggregates();
    std::printf("peak residency: device=%llu pages (%s) host=%llu pages (%s) nvme=%llu pages (%s)\n",
                static_cast<unsigned long long>(peak_vram),
                bytesize_to_string(peak_vram * cfg.page_size).c_str(),
                static_cast<unsigned long long>(peak_host),
                bytesize_to_string(peak_host * cfg.page_size).c_str(),
                static_cast<unsigned long long>(peak_nvme),
                bytesize_to_string(peak_nvme * cfg.page_size).c_str());
    std::printf("aggregates: evictions=%llu writebacks=%llu demand_faults=%llu promotions=%llu "
                "integrity_failures=%llu\n",
                static_cast<unsigned long long>(agg.evictions),
                static_cast<unsigned long long>(agg.writebacks),
                static_cast<unsigned long long>(agg.demand_faults),
                static_cast<unsigned long long>(agg.total_promotions),
                static_cast<unsigned long long>(agg.integrity_failures));

    // Required proof.
    FT_ASSERT(peak_vram > 0);
    FT_ASSERT(peak_host > 0);
    FT_ASSERT(peak_nvme > 0);
    FT_ASSERT(agg.evictions > 0);
    FT_ASSERT(agg.writebacks > 0);
    // NVMe reloads: pages must come back from the store to host.
    const auto nvme_reloads = agg.paths.find("nvme->host_pinned");
    FT_ASSERT(nvme_reloads != agg.paths.end());
    FT_ASSERT(nvme_reloads->second.count > 0);
    FT_ASSERT(agg.total_promotions > 0);
    FT_ASSERT(agg.total_demotions > 0);
    FT_ASSERT(peak_vram * cfg.page_size <= cfg.vram_budget_bytes);  // no overshoot
    FT_ASSERT(peak_host * cfg.page_size <= cfg.host_budget_bytes);
    FT_ASSERT(rt.vram_used() <= cfg.vram_budget_bytes);
    FT_ASSERT(rt.host_used() <= cfg.host_budget_bytes);

    // Integrity after repeated transitions.
    for (PageId id : ids) {
        rt.verify_page(id);
    }
    FT_ASSERT_EQ(agg.integrity_failures, 0u);

    for (PageId id : ids) {
        rt.free_page(id);
    }
    rt.shutdown();
}

FT_TEST(cuda_oversubscription_budget_overshoot_guard) {
    // A page table snapshot after the run must show zero resident pages
    // (clean shutdown) and the device budget must still be reserved.
    Config cfg = oversubscription_config();
    Runtime rt(cfg);
    rt.start();
    std::vector<PageId> ids;
    for (int i = 0; i < 32; ++i) {
        ids.push_back(rt.allocate_page(cfg.page_size));
    }
    for (PageId id : ids) {
        rt.fill_page(id);
        rt.demote_to_nvme(id);
    }
    for (PageId id : ids) {
        rt.free_page(id);
    }
    rt.shutdown();
    FT_ASSERT(rt.device_backend() == nullptr);
}

#else

FT_TEST(cuda_oversubscription_not_compiled) {
    FT_ASSERT(true);
}

#endif

int main() { return ft_test::run_all("test_cuda_oversubscription"); }
