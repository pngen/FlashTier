// Oversubscribed working-set example: the logical working set exceeds the
// configured VRAM budget (in this CPU-only example, the host budget) and
// FlashTier keeps every page intact through repeated eviction cycles.

#include <cstdio>
#include <vector>

#include "flashtier/config.hpp"
#include "flashtier/runtime.hpp"
#include "flashtier/workload.hpp"

using namespace flashtier;

int main() {
    Config cfg;
    cfg.cuda_enabled = false;  // CPU-only by default for examples
    cfg.page_size = 64 * 1024;
    cfg.host_budget_bytes = 4 * 1024 * 1024;   // small on purpose
    cfg.nvme_budget_bytes = 32 * 1024 * 1024;
    cfg.policy = PolicyKind::Predictive;
    cfg.output_dir = ".";

    Runtime rt(cfg);
    rt.start();

    // Working set: 16 pages = 4x the host budget.
    const uint64_t page_count = 16;
    std::vector<PageId> ids;
    for (uint64_t i = 0; i < page_count; ++i) {
        ids.push_back(rt.allocate_page(cfg.page_size));
    }
    for (PageId id : ids) rt.fill_page(id);

    // Skewed access pattern: most accesses hit a hot subset, so the
    // predictive policy should keep hot pages resident.
    TraceGenerator trace(cfg.seed, page_count, 4096, TracePattern::Skewed, 1.2);
    std::vector<uint8_t> buf(cfg.page_size);
    for (const auto& a : trace.accesses()) {
        rt.read_page(ids[a.page_id], buf.data());
    }

    std::printf("residency after workload: vram=%llu host=%llu nvme=%llu\n",
                static_cast<unsigned long long>(rt.pages_resident_vram()),
                static_cast<unsigned long long>(rt.pages_resident_host()),
                static_cast<unsigned long long>(rt.pages_resident_nvme()));

    for (PageId id : ids) rt.verify_page(id);
    std::printf("integrity: all %llu pages verified after oversubscription\n",
                static_cast<unsigned long long>(page_count));

    for (PageId id : ids) rt.free_page(id);
    rt.shutdown();

    const auto agg = rt.aggregates();
    std::printf("\naggregate telemetry:\n%s", format_aggregates(agg).c_str());
    std::printf("oversubscribed_working_set: OK\n");
    return 0;
}
