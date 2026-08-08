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
    cfg.host_budget_bytes = 4 * cfg.page_size;  // four resident pages
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

    const uint64_t vram_pages = rt.pages_resident_vram();
    const uint64_t host_pages = rt.pages_resident_host();
    const uint64_t nvme_pages = rt.pages_resident_nvme();
    std::printf("residency after workload: vram=%llu host=%llu nvme=%llu\n",
                static_cast<unsigned long long>(vram_pages),
                static_cast<unsigned long long>(host_pages),
                static_cast<unsigned long long>(nvme_pages));
    if (nvme_pages == 0) {
        std::fprintf(stderr, "oversubscription example did not spill any page to NVMe\n");
        return 1;
    }

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
