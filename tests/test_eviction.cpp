#include "test_harness.hpp"

#include "flashtier/error.hpp"
#include "flashtier/planner.hpp"

using namespace flashtier;

namespace {

Planner::Budgets budgets(uint64_t vram_limit, uint64_t vram_used,
                         uint64_t host_limit, uint64_t host_used,
                         uint64_t nvme_limit, uint64_t nvme_used) {
    Planner::Budgets b;
    b.vram_limit = vram_limit;
    b.vram_reserve = vram_limit / 10;  // 10% margin
    b.vram_used = vram_used;
    b.host_limit = host_limit;
    b.host_used = host_used;
    b.nvme_limit = nvme_limit;
    b.nvme_used = nvme_used;
    return b;
}

PageMetadata page(uint64_t id, bool pinned = false, bool dirty = false,
                  bool has_nvme_copy = false) {
    PageMetadata p;
    p.id = PageId{id};
    p.allocation_size = 2 * 1024 * 1024;
    p.pinned = pinned;
    p.dirty = dirty;
    p.has_nvme_copy = has_nvme_copy;
    p.state = PageState::ResidentVram;
    p.current_tier = Tier::Vram;
    p.last_access_sequence = id;
    return p;
}

}  // namespace

FT_TEST(planner_headroom_accounts_for_reserve) {
    Planner p(budgets(1000, 500, 1000, 0, 1000, 0));
    // usable = 1000 - 100 = 900; used 500 -> headroom 400.
    FT_ASSERT(p.vram_has_headroom(400));
    FT_ASSERT(!p.vram_has_headroom(401));
}

FT_TEST(planner_no_eviction_needed_when_headroom_ok) {
    Planner p(budgets(1000, 100, 1000, 0, 1000, 0));
    std::vector<PageMetadata> cands = {page(1), page(2)};
    LruPolicy pol;
    auto decisions = p.plan_evictions(0, cands, pol, 1000);
    FT_ASSERT_EQ(decisions.size(), 0u);
}

FT_TEST(planner_selects_oldest_victim_first) {
    Planner p(budgets(1000, 950, 1000, 0, 1000, 0));
    // usable = 900, used 950 impossible; use consistent numbers instead:
    p.set_budgets(budgets(1000, 880, 1000, 0, 1000, 0));
    std::vector<PageMetadata> cands = {
        page(1), page(2), page(3),
    };
    LruPolicy pol;
    auto decisions = p.plan_evictions(100, cands, pol, 1000);
    FT_ASSERT_EQ(decisions.size(), 1u);
    FT_ASSERT_EQ(decisions[0].id.value, 1u);
    FT_ASSERT_EQ(decisions[0].target, Tier::HostPinned);
}

FT_TEST(planner_never_chooses_pinned_victims) {
    Planner p(budgets(1000, 880, 1000, 0, 1000, 0));
    std::vector<PageMetadata> cands = {
        page(1, /*pinned=*/true),
        page(2, /*pinned=*/true),
        page(3),
    };
    LruPolicy pol;
    auto decisions = p.plan_evictions(100, cands, pol, 1000);
    FT_ASSERT_EQ(decisions.size(), 1u);
    FT_ASSERT_EQ(decisions[0].id.value, 3u);
}

FT_TEST(planner_fails_with_no_legal_victim) {
    Planner p(budgets(1000, 880, 1000, 0, 1000, 0));
    std::vector<PageMetadata> cands = {
        page(1, /*pinned=*/true),
    };
    LruPolicy pol;
    FT_ASSERT_THROWS(p.plan_evictions(100, cands, pol, 1000), ErrorCode::Budget);
}

FT_TEST(planner_dirty_pages_without_host_room_go_to_nvme) {
    Planner p(budgets(1000, 880, 1000, 0, 1000, 0));
    std::vector<PageMetadata> cands = {
        page(1, /*pinned=*/false, /*dirty=*/true, /*has_nvme_copy=*/false),
    };
    LruPolicy pol;
    // host_free = 0 -> dirty page must be written back to NVMe.
    auto decisions = p.plan_evictions(100, cands, pol, 0);
    FT_ASSERT_EQ(decisions.size(), 1u);
    FT_ASSERT_EQ(decisions[0].target, Tier::Nvme);
}

FT_TEST(planner_clean_pages_with_nvme_copy_and_no_host_room_drop) {
    Planner p(budgets(1000, 880, 1000, 0, 1000, 0));
    std::vector<PageMetadata> cands = {
        page(1, /*pinned=*/false, /*dirty=*/false, /*has_nvme_copy=*/true),
    };
    LruPolicy pol;
    auto decisions = p.plan_evictions(100, cands, pol, 0);
    FT_ASSERT_EQ(decisions.size(), 1u);
    FT_ASSERT_EQ(decisions[0].target, Tier::Nvme);
}

FT_TEST(planner_evicts_multiple_victims_when_needed) {
    Planner p(budgets(1000, 880, 1000, 0, 1000, 0));
    std::vector<PageMetadata> cands = {page(1), page(2), page(3), page(4)};
    LruPolicy pol;
    // usable = 900, used = 880 -> 20 headroom. Need 5 MiB of freed space:
    // shortfall = 5 MiB - 20 = 5242840 bytes -> 3 victims of 2 MiB each.
    auto decisions = p.plan_evictions(20 + 5ull * 1024 * 1024, cands, pol, 1000);
    FT_ASSERT_EQ(decisions.size(), 3u);
    FT_ASSERT_EQ(decisions[0].id.value, 1u);
    FT_ASSERT_EQ(decisions[2].id.value, 3u);
}

FT_TEST(planner_allocation_tier_priority) {
    Planner p(budgets(1000, 100, 1000, 0, 1000, 0));
    FT_ASSERT_EQ(p.choose_allocation_tier(100, false), Tier::Vram);
    Planner p2(budgets(1000, 950, 1000, 0, 1000, 0));
    FT_ASSERT_EQ(p2.choose_allocation_tier(100, false), Tier::HostPinned);
    Planner p3(budgets(1000, 950, 100, 95, 1000, 0));
    FT_ASSERT_EQ(p3.choose_allocation_tier(100, false), Tier::Nvme);
    Planner p4(budgets(1000, 950, 100, 100, 10, 10));
    FT_ASSERT_THROWS(p4.choose_allocation_tier(100, false), ErrorCode::Budget);
}

int main() { return ft_test::run_all("test_eviction"); }
