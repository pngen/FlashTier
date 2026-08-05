#include "test_harness.hpp"

#include <cstring>

#include "flashtier/error.hpp"
#include "flashtier/page.hpp"
#include "flashtier/page_id.hpp"
#include "flashtier/page_table.hpp"

using namespace flashtier;

FT_TEST(page_ids_are_stable_and_ordered) {
    PageId a{5}, b{5}, c{6};
    FT_ASSERT(a == b);
    FT_ASSERT(a != c);
    FT_ASSERT(a < c);
    FT_ASSERT_EQ(a.to_string(), "5");
}

FT_TEST(legal_transitions_apply) {
    PageMetadata p;
    p.id = PageId{1};
    p.transition(PageState::ResidentVram);
    FT_ASSERT_EQ(p.state, PageState::ResidentVram);
    FT_ASSERT_EQ(p.current_tier, Tier::Vram);
    p.transition(PageState::EvictingToHost);
    p.transition(PageState::ResidentHost);
    FT_ASSERT_EQ(p.current_tier, Tier::HostPinned);
    p.transition(PageState::EvictingToNvme);
    p.transition(PageState::ResidentNvme);
    FT_ASSERT_EQ(p.current_tier, Tier::Nvme);
    p.transition(PageState::LoadingToHost);
    p.transition(PageState::ResidentHost);
    p.transition(PageState::LoadingToVram);
    p.transition(PageState::ResidentVram);
    p.transition(PageState::Released);
}

FT_TEST(illegal_transitions_are_rejected) {
    PageMetadata p;
    p.id = PageId{2};
    FT_ASSERT_THROWS(p.transition(PageState::EvictingToHost), ErrorCode::State);  // from Unallocated
    p.transition(PageState::ResidentVram);
    FT_ASSERT_THROWS(p.transition(PageState::ResidentVram), ErrorCode::State);    // no-op not allowed
    FT_ASSERT_THROWS(p.transition(PageState::ResidentNvme), ErrorCode::State);    // no direct hop
    FT_ASSERT_THROWS(p.transition(PageState::LoadingToVram), ErrorCode::State);
    p.transition(PageState::Released);
    FT_ASSERT_THROWS(p.transition(PageState::Unallocated), ErrorCode::State);     // terminal
    FT_ASSERT_THROWS(p.transition(PageState::Released), ErrorCode::State);
}

FT_TEST(transition_table_is_symmetric_with_spec) {
    FT_ASSERT(transition_allowed(PageState::Unallocated, PageState::ResidentHost));
    FT_ASSERT(transition_allowed(PageState::Unallocated, PageState::ResidentVram));
    FT_ASSERT(transition_allowed(PageState::Unallocated, PageState::ResidentNvme));
    FT_ASSERT(transition_allowed(PageState::Unallocated, PageState::LoadingToVram));
    FT_ASSERT(transition_allowed(PageState::LoadingToVram, PageState::ResidentVram));
    FT_ASSERT(transition_allowed(PageState::ResidentVram, PageState::EvictingToHost));
    FT_ASSERT(transition_allowed(PageState::EvictingToHost, PageState::ResidentHost));
    FT_ASSERT(transition_allowed(PageState::ResidentHost, PageState::EvictingToNvme));
    FT_ASSERT(transition_allowed(PageState::EvictingToNvme, PageState::ResidentNvme));
    FT_ASSERT(transition_allowed(PageState::ResidentNvme, PageState::LoadingToHost));
    FT_ASSERT(transition_allowed(PageState::LoadingToHost, PageState::ResidentHost));
    FT_ASSERT(transition_allowed(PageState::ResidentHost, PageState::LoadingToVram));
    FT_ASSERT(transition_allowed(PageState::LoadingToVram, PageState::Error));
    FT_ASSERT(transition_allowed(PageState::Error, PageState::Released));
    FT_ASSERT(!transition_allowed(PageState::ResidentVram, PageState::ResidentNvme));
    FT_ASSERT(!transition_allowed(PageState::ResidentNvme, PageState::ResidentVram));
    FT_ASSERT(!transition_allowed(PageState::Released, PageState::ResidentHost));
}

FT_TEST(page_table_insert_lookup_erase) {
    PageTable t;
    PageMetadata m;
    m.id = PageId{7};
    m.state = PageState::ResidentHost;
    t.insert(m);
    FT_ASSERT(t.contains(PageId{7}));
    FT_ASSERT_EQ(t.size(), 1u);
    bool seen = false;
    t.with(PageId{7}, [&](PageMetadata& p) {
        seen = true;
        p.access_count = 3;
    });
    FT_ASSERT(seen);
    FT_ASSERT_EQ(t.copy_of(PageId{7}).access_count, 3u);
    t.erase(PageId{7});
    FT_ASSERT(!t.contains(PageId{7}));
    FT_ASSERT_THROWS(t.copy_of(PageId{7}), ErrorCode::NotFound);
    FT_ASSERT_THROWS(t.with(PageId{7}, [](PageMetadata&){}), ErrorCode::NotFound);
}

int main() { return ft_test::run_all("test_page_table"); }
