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
    FT_ASSERT_THROWS(p.transition(PageState::LoadingToVram), ErrorCode::State);
    // ResidentVram -> ResidentNvme is the legal clean drop (valid NVMe copy,
    // no transfer); the next transition from ResidentNvme must be a load.
    p.dirty = false;
    p.has_nvme_copy = true;
    p.nvme_offset = 4096;
    p.transition(PageState::ResidentNvme);
    FT_ASSERT_THROWS(p.transition(PageState::ResidentVram), ErrorCode::State);    // no direct hop
    p.transition(PageState::Released);
    FT_ASSERT_THROWS(p.transition(PageState::Unallocated), ErrorCode::State);     // terminal
    FT_ASSERT_THROWS(p.transition(PageState::Released), ErrorCode::State);
}

FT_TEST(clean_drop_requires_a_valid_nvme_copy) {
    PageMetadata p;
    p.id = PageId{3};
    p.transition(PageState::ResidentVram);

    FT_ASSERT_THROWS(p.transition(PageState::ResidentNvme), ErrorCode::Invariant);
    FT_ASSERT_EQ(p.state, PageState::ResidentVram);

    p.has_nvme_copy = true;
    FT_ASSERT_THROWS(p.transition(PageState::ResidentNvme), ErrorCode::Invariant);
    p.nvme_offset = 4096;
    p.dirty = true;
    FT_ASSERT_THROWS(p.transition(PageState::ResidentNvme), ErrorCode::Invariant);
    p.dirty = false;
    p.transition(PageState::ResidentNvme);
}

FT_TEST(transition_table_is_symmetric_with_spec) {
    FT_ASSERT(transition_allowed(PageState::Unallocated, PageState::ResidentHost));
    FT_ASSERT(transition_allowed(PageState::Unallocated, PageState::ResidentVram));
    FT_ASSERT(transition_allowed(PageState::Unallocated, PageState::ResidentNvme));
    FT_ASSERT(transition_allowed(PageState::Unallocated, PageState::LoadingToVram));
    FT_ASSERT(transition_allowed(PageState::LoadingToVram, PageState::ResidentVram));
    FT_ASSERT(transition_allowed(PageState::ResidentVram, PageState::EvictingToHost));
    FT_ASSERT(transition_allowed(PageState::EvictingToHost, PageState::ResidentHost));
    FT_ASSERT(transition_allowed(PageState::ResidentVram, PageState::ResidentNvme));  // clean drop
    FT_ASSERT(!transition_allowed(PageState::ResidentVram, PageState::ResidentHost));
    FT_ASSERT(transition_allowed(PageState::ResidentHost, PageState::EvictingToNvme));
    FT_ASSERT(transition_allowed(PageState::EvictingToNvme, PageState::ResidentNvme));
    FT_ASSERT(transition_allowed(PageState::ResidentNvme, PageState::LoadingToHost));
    FT_ASSERT(transition_allowed(PageState::LoadingToHost, PageState::ResidentHost));
    FT_ASSERT(transition_allowed(PageState::ResidentHost, PageState::LoadingToVram));
    FT_ASSERT(transition_allowed(PageState::LoadingToVram, PageState::Error));
    FT_ASSERT(transition_allowed(PageState::Error, PageState::Released));
    FT_ASSERT(!transition_allowed(PageState::ResidentVram, PageState::ResidentHost));
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

FT_TEST(page_table_rejects_duplicate_ids_without_overwriting) {
    PageTable t;
    PageMetadata original;
    original.id = PageId{11};
    original.logical_size = 123;
    t.insert(original);

    PageMetadata duplicate;
    duplicate.id = original.id;
    duplicate.logical_size = 999;
    FT_ASSERT_THROWS(t.insert(duplicate), ErrorCode::Invariant);
    FT_ASSERT_EQ(t.size(), 1u);
    FT_ASSERT_EQ(t.copy_of(original.id).logical_size, 123u);
}

int main() { return ft_test::run_all("test_page_table"); }
