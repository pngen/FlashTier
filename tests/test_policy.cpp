#include "test_harness.hpp"

#include <vector>

#include "flashtier/policy.hpp"

using namespace flashtier;

namespace {

PageMetadata make_page(uint64_t id, uint64_t last_access, uint64_t accesses,
                       bool pinned = false, bool dirty = false,
                       SemanticClass cls = SemanticClass::Generic) {
    PageMetadata p;
    p.id = PageId{id};
    p.last_access_sequence = last_access;
    p.access_count = accesses;
    p.pinned = pinned;
    p.dirty = dirty;
    p.semantic_class = cls;
    p.allocation_size = 2 * 1024 * 1024;
    return p;
}

}  // namespace

FT_TEST(lru_victim_order_is_oldest_first) {
    LruPolicy p;
    std::vector<PageMetadata> cands = {
        make_page(1, 100, 1),
        make_page(2, 50, 1),
        make_page(3, 200, 1),
    };
    auto victims = p.rank_victims(cands);
    FT_ASSERT_EQ(victims[0].value, 2u);
    FT_ASSERT_EQ(victims[1].value, 1u);
    FT_ASSERT_EQ(victims[2].value, 3u);
}

FT_TEST(lru_ties_break_by_page_id) {
    LruPolicy p;
    std::vector<PageMetadata> cands = {
        make_page(9, 100, 1),
        make_page(3, 100, 1),
        make_page(1, 100, 1),
    };
    auto victims = p.rank_victims(cands);
    FT_ASSERT_EQ(victims[0].value, 1u);
    FT_ASSERT_EQ(victims[1].value, 3u);
    FT_ASSERT_EQ(victims[2].value, 9u);
}

FT_TEST(lru_is_deterministic) {
    LruPolicy p;
    std::vector<PageMetadata> cands = {
        make_page(5, 40, 1),
        make_page(2, 10, 1),
        make_page(8, 90, 1),
    };
    const auto a = p.rank_victims(cands);
    const auto b = p.rank_victims(cands);
    FT_ASSERT(a == b);
}

FT_TEST(predictive_prefers_hot_and_pinned_pages) {
    PredictivePolicy p;
    PageMetadata cold = make_page(1, 5, 1);
    PageMetadata hot = make_page(2, 9000, 500);
    PageMetadata pinned = make_page(3, 9001, 10, /*pinned=*/true);
    FT_ASSERT(p.score(hot) > p.score(cold));
    FT_ASSERT(p.score(pinned) > p.score(cold));

    std::vector<PageMetadata> cands = {cold, hot, pinned};
    auto victims = p.rank_victims(cands);
    FT_ASSERT_EQ(victims.back().value, 3u);  // pinned is the worst victim
    FT_ASSERT_EQ(victims.front().value, 1u); // coldest is the best victim
}

FT_TEST(predictive_prefers_dirty_writes_for_eviction_penalty) {
    PredictivePolicy p;
    PageMetadata clean = make_page(1, 50, 5);
    PageMetadata dirty = make_page(2, 51, 5);
    dirty.dirty = true;
    FT_ASSERT(p.score(dirty) < p.score(clean));
}

FT_TEST(predictive_class_weights_matter) {
    PredictivePolicy p;
    PageMetadata generic = make_page(1, 50, 5);
    PageMetadata kv = make_page(2, 51, 5);
    kv.semantic_class = SemanticClass::KvCache;
    FT_ASSERT(p.score(kv) > p.score(generic));
}

FT_TEST(predictive_is_deterministic) {
    PredictivePolicy p;
    std::vector<PageMetadata> cands;
    for (uint64_t i = 0; i < 64; ++i) {
        cands.push_back(make_page(i, i * 7, (i * 3) % 17, i % 5 == 0, i % 3 == 0));
    }
    const auto a = p.rank_victims(cands);
    const auto b = p.rank_victims(cands);
    FT_ASSERT(a == b);
    FT_ASSERT_EQ(a.size(), 64u);
}

FT_TEST(rank_prefetch_orders_promotion_candidates) {
    PredictivePolicy p;
    std::vector<PageMetadata> cands = {
        make_page(1, 10, 2),
        make_page(2, 100, 30),
        make_page(3, 99, 4),
    };
    auto order = p.rank_prefetch(cands);
    FT_ASSERT_EQ(order[0].value, 2u);  // hottest first
}

int main() { return ft_test::run_all("test_policy"); }
