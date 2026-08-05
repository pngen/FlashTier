#include "test_harness.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

#include "flashtier/config.hpp"
#include "flashtier/error.hpp"
#include "flashtier/runtime.hpp"

using namespace flashtier;

namespace {

Config gpu_test_config() {
    Config cfg;
    cfg.backend = "cuda";
    cfg.device_id = 0;
    cfg.page_size = 2 * 1024 * 1024;
    cfg.vram_budget_bytes = 256ull * 1024 * 1024;  // small device budget
    cfg.host_budget_bytes = 128ull * 1024 * 1024;  // constrained host budget
    cfg.nvme_budget_bytes = 1024ull * 1024 * 1024;
    cfg.queue_depth = 8;
    cfg.policy = PolicyKind::Lru;
    cfg.output_dir = ".";
    return cfg;
}

}  // namespace

#if FLASHTIER_HAVE_CUDA

// Governed-runtime validation on the CUDA backend (LABELS gpu): small
// device budget, working set larger than the device budget, genuine
// three-tier spill, integrity, and deterministic shutdown.

FT_TEST(cuda_runtime_backend_selected) {
    Config cfg = gpu_test_config();
    Runtime rt(cfg);
    rt.start();
    FT_ASSERT_EQ(rt.selected_backend(), std::string("cuda"));
    FT_ASSERT(rt.device_backend() != nullptr);
    FT_ASSERT_EQ(rt.device_backend()->backend_name(), std::string("cuda"));
    rt.shutdown();
}

FT_TEST(cuda_runtime_three_tier_residency_and_integrity) {
    Config cfg = gpu_test_config();
    Runtime rt(cfg);
    rt.start();

    // Working set 384 MiB > device budget 256 MiB and > device+host
    // (256+128 = 384 MiB): some pages must reach NVMe.
    const uint64_t page_count = 192;  // 384 MiB at 2 MiB pages
    std::vector<PageId> ids;
    ids.reserve(page_count);
    for (uint64_t i = 0; i < page_count; ++i) {
        ids.push_back(rt.allocate_page(cfg.page_size));
    }
    for (PageId id : ids) {
        rt.fill_page(id);
    }
    // Demand churn: every page promoted then evicted, then reloaded.
    // Capture peak residency during the rounds: after evict-all the
    // instantaneous device count is zero by design.
    uint64_t peak_vram = 0, peak_host = 0, peak_nvme = 0;
    for (int round = 0; round < 2; ++round) {
        for (PageId id : ids) {
            rt.promote(id);
        }
        peak_vram = std::max(peak_vram, rt.pages_resident_vram());
        peak_host = std::max(peak_host, rt.pages_resident_host());
        for (PageId id : ids) {
            rt.evict(id);
        }
        peak_nvme = std::max(peak_nvme, rt.pages_resident_nvme());
        peak_host = std::max(peak_host, rt.pages_resident_host());
    }
    // Verify residency spread and integrity.
    FT_ASSERT(peak_vram > 0);   // some pages resided in device memory
    FT_ASSERT(peak_host > 0);   // some pages resided in pinned host memory
    FT_ASSERT(peak_nvme > 0);   // working set exceeds device + host budgets
    FT_ASSERT(peak_vram * cfg.page_size <= cfg.vram_budget_bytes);  // no overshoot
    FT_ASSERT(rt.pages_resident_vram() + rt.pages_resident_host() +
                  rt.pages_resident_nvme() ==
              page_count);
    FT_ASSERT(rt.vram_used() <= cfg.vram_budget_bytes);  // no budget overshoot
    for (PageId id : ids) {
        rt.verify_page(id);
    }
    // Reload everything once more and verify again.
    for (PageId id : ids) {
        rt.promote(id);
        rt.verify_page(id);
    }
    const auto agg = rt.aggregates();
    FT_ASSERT_EQ(agg.integrity_failures, 0u);
    FT_ASSERT(agg.evictions > 0);
    FT_ASSERT(agg.writebacks > 0);  // dirty pages persisted to NVMe
    for (PageId id : ids) {
        rt.free_page(id);
    }
    rt.shutdown();
}

FT_TEST(cuda_runtime_shutdown_is_deterministic) {
    Config cfg = gpu_test_config();
    Runtime rt(cfg);
    rt.start();
    std::vector<PageId> ids;
    for (int i = 0; i < 32; ++i) {
        ids.push_back(rt.allocate_page(cfg.page_size));
    }
    for (PageId id : ids) {
        rt.fill_page(id);
    }
    for (PageId id : ids) {
        rt.demote_to_nvme(id);
    }
    for (PageId id : ids) {
        rt.free_page(id);
    }
    rt.shutdown();
    FT_ASSERT(rt.device_backend() == nullptr);
}

#else

FT_TEST(cuda_runtime_not_compiled) {
    Config cfg = gpu_test_config();
    cfg.backend = "cuda";
    Runtime rt(cfg);
    FT_ASSERT_THROWS(rt.start(), ErrorCode::Config);  // unknown backend
}

#endif

int main() { return ft_test::run_all("test_cuda_runtime"); }
