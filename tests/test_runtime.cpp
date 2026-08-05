#include "test_harness.hpp"

#include <vector>

#include "flashtier/config.hpp"
#include "flashtier/error.hpp"
#include "flashtier/integrity.hpp"
#include "flashtier/page_table.hpp"
#include "flashtier/planner.hpp"
#include "flashtier/runtime.hpp"
#include "flashtier/telemetry.hpp"

using namespace flashtier;

namespace {

// Small page size keeps the CPU-only runtime test fast.
Config test_config() {
    Config cfg;
    cfg.cuda_enabled = false;
    cfg.page_size = 64 * 1024;
    cfg.host_budget_bytes = 256 * 1024 * 1024;
    cfg.nvme_budget_bytes = 256 * 1024 * 1024;
    cfg.vram_budget_bytes = 0;  // ignored in CPU-only mode
    cfg.output_dir = ".";
    cfg.retain_store = false;
    cfg.queue_depth = 4;
    cfg.policy = PolicyKind::Lru;
    return cfg;
}

}  // namespace

FT_TEST(cpu_runtime_alloc_write_read_cycle) {
    Config cfg = test_config();
    Runtime rt(cfg);
    rt.start();
    PageId id = rt.allocate_page(128 * 1024);
    FT_ASSERT(rt.metadata(id).state == PageState::ResidentHost ||
              rt.metadata(id).state == PageState::ResidentNvme);

    std::vector<uint8_t> data(128 * 1024, 0xAB);
    rt.write_page(id, data.data());
    FT_ASSERT_EQ(rt.page_checksum(id), fnv1a64(data.data(), data.size()));

    std::vector<uint8_t> out(128 * 1024, 0);
    rt.read_page(id, out.data());
    FT_ASSERT(std::memcmp(data.data(), out.data(), data.size()) == 0);

    rt.free_page(id);
    rt.shutdown();
}

FT_TEST(cpu_runtime_promote_demote_cycles_preserve_data) {
    Config cfg = test_config();
    Runtime rt(cfg);
    rt.start();
    PageId id = rt.allocate_page(64 * 1024);
    std::vector<uint8_t> data(64 * 1024);
    fill_pattern(data.data(), data.size(), cfg.seed, id.value);
    rt.write_page(id, data.data());

    // Drive the page through every residency state and verify content.
    rt.demote_to_nvme(id);
    FT_ASSERT(rt.metadata(id).state == PageState::ResidentNvme);
    rt.verify_page(id);

    rt.promote(id);
    FT_ASSERT(rt.metadata(id).state == PageState::ResidentHost);
    rt.verify_page(id);

    rt.demote_to_host(id);
    FT_ASSERT(rt.metadata(id).state == PageState::ResidentHost);
    rt.verify_page(id);

    rt.promote(id);
    rt.verify_page(id);

    std::vector<uint8_t> out(64 * 1024, 0);
    rt.read_page(id, out.data());
    FT_ASSERT(std::memcmp(data.data(), out.data(), data.size()) == 0);

    rt.free_page(id);
    rt.shutdown();
}

FT_TEST(cpu_runtime_eviction_cycles_keep_integrity) {
    Config cfg = test_config();
    cfg.host_budget_bytes = 6 * 1024 * 1024;
    cfg.nvme_budget_bytes = 16 * 1024 * 1024;
    Runtime rt(cfg);
    rt.start();

    // Working set of 8 pages exceeds the host budget: repeated
    // promote/demote/evict cycles must preserve every page's content.
    std::vector<PageId> ids;
    std::vector<std::vector<uint8_t>> contents;
    for (int i = 0; i < 8; ++i) {
        PageId id = rt.allocate_page(64 * 1024, SemanticClass::KvCache);
        ids.push_back(id);
        std::vector<uint8_t> data(64 * 1024);
        fill_pattern(data.data(), data.size(), cfg.seed, id.value);
        rt.write_page(id, data.data());
        contents.push_back(std::move(data));
    }
    for (int round = 0; round < 3; ++round) {
        for (std::size_t i = 0; i < ids.size(); ++i) {
            rt.promote(ids[i]);
            rt.verify_page(ids[i]);
        }
        for (std::size_t i = 0; i < ids.size(); ++i) {
            rt.evict(ids[i]);
        }
        for (std::size_t i = 0; i < ids.size(); ++i) {
            rt.verify_page(ids[i]);
            std::vector<uint8_t> out(64 * 1024, 0);
            rt.read_page(ids[i], out.data());
            FT_ASSERT(std::memcmp(contents[i].data(), out.data(), 64 * 1024) == 0);
        }
    }
    for (PageId id : ids) rt.free_page(id);
    rt.shutdown();
}

FT_TEST(cpu_runtime_prefetch_hit_and_waste_accounting) {
    Config cfg = test_config();
    Runtime rt(cfg);
    rt.start();
    std::vector<PageId> ids;
    for (int i = 0; i < 8; ++i) {
        ids.push_back(rt.allocate_page(64 * 1024));
    }
    // Prefetch the whole set, then touch half of it.
    rt.prefetch(ids);
    for (int i = 0; i < 4; ++i) {
        std::vector<uint8_t> out(64 * 1024, 0);
        rt.read_page(ids[i], out.data());
    }
    rt.flush_telemetry();
    const auto agg = rt.aggregates();
    FT_ASSERT(agg.prefetches_issued >= 8);
    FT_ASSERT(agg.prefetch_hits >= 4);
    for (PageId id : ids) rt.free_page(id);
    rt.shutdown();
}

FT_TEST(cpu_runtime_pinned_pages_never_evicted) {
    Config cfg = test_config();
    cfg.host_budget_bytes = 3 * 1024 * 1024;
    cfg.nvme_budget_bytes = 8 * 1024 * 1024;
    Runtime rt(cfg);
    rt.start();
    PageId pinned = rt.allocate_page(64 * 1024, SemanticClass::Generic, /*pinned=*/true);
    rt.fill_page(pinned);
    rt.pin(pinned, true);

    std::vector<PageId> others;
    for (int i = 0; i < 4; ++i) {
        others.push_back(rt.allocate_page(64 * 1024));
    }
    // Churn the host budget; the pinned page must stay resident in host
    // while the others spill to NVMe.
    for (int round = 0; round < 4; ++round) {
        for (PageId id : others) {
            rt.promote(id);
            rt.evict(id);
        }
    }
    FT_ASSERT(rt.metadata(pinned).state == PageState::ResidentHost);
    rt.verify_page(pinned);
    for (PageId id : others) rt.free_page(id);
    rt.free_page(pinned);
    rt.shutdown();
}

FT_TEST(cpu_runtime_frees_all_resources_and_reopens_store) {
    Config cfg = test_config();
    cfg.retain_store = true;
    Runtime rt(cfg);
    rt.start();
    PageId a = rt.allocate_page(64 * 1024);
    rt.fill_page(a);
    rt.demote_to_nvme(a);
    rt.free_page(a);
    rt.shutdown();
    // Store retained; a second runtime must be able to reopen it.
    Runtime rt2(cfg);
    rt2.start();
    PageId b = rt2.allocate_page(64 * 1024);
    rt2.fill_page(b);
    rt2.free_page(b);
    rt2.shutdown();
}

FT_TEST(cpu_runtime_illegal_state_access_rejected) {
    Config cfg = test_config();
    Runtime rt(cfg);
    rt.start();
    PageId missing{9999};
    FT_ASSERT_THROWS(rt.read_page(missing, nullptr), ErrorCode::NotFound);
    FT_ASSERT_THROWS(rt.metadata(missing), ErrorCode::NotFound);
    rt.shutdown();
}

FT_TEST(cpu_runtime_unwritable_page_state_raises) {
    Config cfg = test_config();
    Runtime rt(cfg);
    rt.start();
    // A freed page is released; touching it must fail cleanly.
    PageId id = rt.allocate_page(64 * 1024);
    rt.free_page(id);
    std::vector<uint8_t> data(64 * 1024, 1);
    FT_ASSERT_THROWS(rt.write_page(id, data.data()), ErrorCode::NotFound);
    rt.shutdown();
}

int main() { return ft_test::run_all("test_runtime"); }
