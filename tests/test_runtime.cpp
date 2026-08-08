#include "test_harness.hpp"

#include <atomic>
#include <cmath>
#include <filesystem>
#include <limits>
#include <mutex>
#include <thread>
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
    cfg.output_dir = std::filesystem::temp_directory_path().string();
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
    if (rt.metadata(id).state == PageState::ResidentHost) {
        FT_ASSERT_EQ(rt.metadata(id).current_tier, Tier::HostPageable);
    }

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

FT_TEST(cpu_runtime_prefetch_of_resident_pages_is_an_accounting_noop) {
    Config cfg = test_config();
    Runtime rt(cfg);
    rt.start();
    std::vector<PageId> ids;
    for (int i = 0; i < 8; ++i) {
        ids.push_back(rt.allocate_page(64 * 1024));
    }
    const auto before = rt.aggregates();
    // Every page is already resident, so this is not a prefetch issue or hit.
    rt.prefetch(ids);
    for (int i = 0; i < 4; ++i) {
        std::vector<uint8_t> out(64 * 1024, 0);
        rt.read_page(ids[i], out.data());
    }
    rt.flush_telemetry();
    const auto agg = rt.aggregates();
    FT_ASSERT_EQ(agg.prefetches_issued, before.prefetches_issued);
    FT_ASSERT_EQ(agg.prefetch_hits, before.prefetch_hits);
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

FT_TEST(tier_budget_rejects_invalid_margins_overflow_and_overrelease) {
    TierBudget budget;
    FT_ASSERT_THROWS(budget.set_limit(1024, -0.1), ErrorCode::InvalidArgument);
    FT_ASSERT_THROWS(budget.set_limit(1024, std::numeric_limits<double>::quiet_NaN()),
                     ErrorCode::InvalidArgument);
    FT_ASSERT_THROWS(budget.set_limit(1024, 1.0), ErrorCode::InvalidArgument);

    budget.set_limit(1024, 0.0);
    FT_ASSERT(budget.try_reserve(512));
    FT_ASSERT_THROWS(budget.release(513), ErrorCode::Invariant);
    FT_ASSERT_EQ(budget.used(), 512u);
    budget.release(512);
    budget.force_reserve(std::numeric_limits<uint64_t>::max());
    FT_ASSERT_THROWS(budget.force_reserve(1), ErrorCode::Budget);
}

FT_TEST(cpu_runtime_rejects_invalid_lifecycle_and_buffer_calls) {
    Config cfg = test_config();
    Runtime rt(cfg);
    FT_ASSERT_THROWS(rt.allocate_page(cfg.page_size), ErrorCode::State);
    rt.start();
    FT_ASSERT_THROWS(rt.start(), ErrorCode::State);

    PageId id = rt.allocate_page(cfg.page_size);
    FT_ASSERT_THROWS(rt.write_page(id, nullptr), ErrorCode::InvalidArgument);
    FT_ASSERT_THROWS(rt.read_page(id, nullptr), ErrorCode::InvalidArgument);
    rt.free_page(id);
    rt.shutdown();
    FT_ASSERT_THROWS(rt.allocate_page(cfg.page_size), ErrorCode::State);
}

FT_TEST(cpu_runtime_multi_extent_nvme_pages_do_not_overlap) {
    Config cfg = test_config();
    Runtime rt(cfg);
    rt.start();

    PageId wide = rt.allocate_page(2 * cfg.page_size);
    PageId adjacent = rt.allocate_page(cfg.page_size);
    std::vector<uint8_t> wide_data(2 * cfg.page_size);
    for (std::size_t i = 0; i < wide_data.size(); ++i) {
        wide_data[i] = static_cast<uint8_t>((i * 131u + 17u) & 0xffu);
    }
    std::vector<uint8_t> adjacent_data(cfg.page_size, 0xD7);
    rt.write_page(wide, wide_data.data());
    rt.write_page(adjacent, adjacent_data.data());
    rt.demote_to_nvme(wide);
    rt.demote_to_nvme(adjacent);

    std::vector<uint8_t> wide_out(wide_data.size());
    std::vector<uint8_t> adjacent_out(adjacent_data.size());
    rt.read_page(wide, wide_out.data());
    rt.read_page(adjacent, adjacent_out.data());
    FT_ASSERT(wide_out == wide_data);
    FT_ASSERT(adjacent_out == adjacent_data);

    rt.free_page(wide);
    rt.free_page(adjacent);
    FT_ASSERT_EQ(rt.nvme_used(), 0u);
    rt.shutdown();
}

FT_TEST(cpu_runtime_rejects_alignment_overflow_without_allocating_an_id) {
    Config cfg = test_config();
    Runtime rt(cfg);
    rt.start();
    FT_ASSERT_THROWS(rt.allocate_page(std::numeric_limits<uint64_t>::max()),
                     ErrorCode::InvalidArgument);
    PageId first = rt.allocate_page(cfg.page_size);
    FT_ASSERT_EQ(first.value, 1u);
    rt.free_page(first);
    rt.shutdown();
}

FT_TEST(cpu_runtime_free_releases_residency_and_cached_nvme_extent) {
    Config cfg = test_config();
    Runtime rt(cfg);
    rt.start();
    PageId id = rt.allocate_page(cfg.page_size);
    rt.fill_page(id);
    rt.demote_to_nvme(id);
    rt.promote(id);
    FT_ASSERT_EQ(rt.metadata(id).state, PageState::ResidentHost);
    FT_ASSERT_EQ(rt.metadata(id).current_tier, Tier::HostPageable);
    FT_ASSERT(rt.host_used() != 0);
    FT_ASSERT(rt.nvme_used() != 0);
    rt.free_page(id);
    FT_ASSERT_EQ(rt.host_used(), 0u);
    FT_ASSERT_EQ(rt.nvme_used(), 0u);
    rt.shutdown();
}

FT_TEST(cpu_runtime_concurrent_free_has_one_owner_and_no_leak) {
    Config cfg = test_config();
    Runtime rt(cfg);
    rt.start();
    PageId id = rt.allocate_page(cfg.page_size);
    rt.fill_page(id);
    rt.demote_to_nvme(id);
    rt.promote(id);  // page now owns host residency plus a cached NVMe extent

    std::atomic<bool> go{false};
    std::atomic<int> successes{0};
    std::atomic<int> expected_failures{0};
    std::exception_ptr unexpected[2];
    std::thread threads[2];
    for (int i = 0; i < 2; ++i) {
        threads[i] = std::thread([&, i] {
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            try {
                rt.free_page(id);
                successes.fetch_add(1, std::memory_order_relaxed);
            } catch (const Error& e) {
                if (e.code() == ErrorCode::NotFound || e.code() == ErrorCode::State) {
                    expected_failures.fetch_add(1, std::memory_order_relaxed);
                } else {
                    unexpected[i] = std::current_exception();
                }
            } catch (...) {
                unexpected[i] = std::current_exception();
            }
        });
    }
    go.store(true, std::memory_order_release);
    for (auto& thread : threads) thread.join();
    for (const auto& failure : unexpected) {
        if (failure) std::rethrow_exception(failure);
    }
    FT_ASSERT_EQ(successes.load(std::memory_order_relaxed), 1);
    FT_ASSERT_EQ(expected_failures.load(std::memory_order_relaxed), 1);
    FT_ASSERT_EQ(rt.host_used(), 0u);
    FT_ASSERT_EQ(rt.nvme_used(), 0u);
    rt.shutdown();
}

FT_TEST(cpu_runtime_shutdown_releases_live_pages_and_supports_restart) {
    Config cfg = test_config();
    Runtime rt(cfg);
    rt.start();
    PageId host_page = rt.allocate_page(cfg.page_size);
    PageId cached_page = rt.allocate_page(cfg.page_size);
    rt.fill_page(host_page);
    rt.fill_page(cached_page);
    rt.demote_to_nvme(cached_page);
    rt.promote(cached_page);
    FT_ASSERT(rt.host_used() != 0);
    FT_ASSERT(rt.nvme_used() != 0);

    rt.shutdown();
    FT_ASSERT_EQ(rt.host_used(), 0u);
    FT_ASSERT_EQ(rt.vram_used(), 0u);
    FT_ASSERT_EQ(rt.nvme_used(), 0u);
    FT_ASSERT_EQ(rt.pages_resident_host(), 0u);
    FT_ASSERT_EQ(rt.pages_resident_vram(), 0u);
    FT_ASSERT_EQ(rt.pages_resident_nvme(), 0u);

    rt.start();
    PageId restarted = rt.allocate_page(cfg.page_size);
    rt.fill_page(restarted);
    rt.verify_page(restarted);
    rt.free_page(restarted);
    rt.shutdown();
}

FT_TEST(cpu_runtime_instances_use_isolated_automatic_stores) {
    Config cfg = test_config();
    Runtime first(cfg);
    Runtime second(cfg);
    first.start();
    second.start();

    PageId a = first.allocate_page(cfg.page_size);
    PageId b = second.allocate_page(cfg.page_size);
    std::vector<uint8_t> a_data(cfg.page_size, 0x31);
    std::vector<uint8_t> b_data(cfg.page_size, 0xE2);
    first.write_page(a, a_data.data());
    second.write_page(b, b_data.data());
    first.demote_to_nvme(a);
    second.demote_to_nvme(b);

    std::vector<uint8_t> a_out(cfg.page_size);
    std::vector<uint8_t> b_out(cfg.page_size);
    first.read_page(a, a_out.data());
    second.read_page(b, b_out.data());
    FT_ASSERT(a_out == a_data);
    FT_ASSERT(b_out == b_data);

    first.free_page(a);
    second.free_page(b);
    first.shutdown();
    second.shutdown();
}

FT_TEST(cpu_runtime_queued_prefetch_records_a_real_hit) {
    Config cfg = test_config();
    cfg.queue_depth = 1;
    cfg.worker_threads = 1;
    Runtime rt(cfg);
    rt.start();
    PageId id = rt.allocate_page(cfg.page_size);
    rt.fill_page(id);
    rt.demote_to_nvme(id);
    const auto before = rt.aggregates();
    rt.prefetch({id, id});
    std::vector<uint8_t> out(cfg.page_size);
    rt.read_page(id, out.data());
    const auto after = rt.aggregates();
    FT_ASSERT_EQ(after.prefetches_issued, before.prefetches_issued + 1);
    FT_ASSERT_EQ(after.prefetch_hits, before.prefetch_hits + 1);
    rt.free_page(id);
    rt.shutdown();
}

FT_TEST(cpu_runtime_bounded_queue_wakes_all_demand_submitters) {
    Config cfg = test_config();
    cfg.queue_depth = 1;
    cfg.worker_threads = 1;
    cfg.host_budget_bytes = cfg.page_size;
    constexpr std::size_t kPages = 16;
    Runtime rt(cfg);
    rt.start();

    std::vector<PageId> ids;
    for (std::size_t i = 0; i < kPages; ++i) {
        PageId id = rt.allocate_page(cfg.page_size);
        std::vector<uint8_t> data(cfg.page_size, static_cast<uint8_t>(i + 1));
        rt.write_page(id, data.data());
        rt.demote_to_nvme(id);
        ids.push_back(id);
    }

    std::atomic<bool> go{false};
    std::vector<std::exception_ptr> failures(kPages);
    std::vector<std::thread> threads;
    threads.reserve(kPages);
    for (std::size_t i = 0; i < kPages; ++i) {
        threads.emplace_back([&, i] {
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            try {
                std::vector<uint8_t> out(cfg.page_size);
                rt.read_page(ids[i], out.data());
                if (out.front() != static_cast<uint8_t>(i + 1) ||
                    out.back() != static_cast<uint8_t>(i + 1)) {
                    throw ft_test::Failure{"concurrent demand returned corrupt data"};
                }
            } catch (...) {
                failures[i] = std::current_exception();
            }
        });
    }
    go.store(true, std::memory_order_release);
    for (auto& thread : threads) thread.join();
    for (const auto& failure : failures) {
        if (failure) std::rethrow_exception(failure);
    }

    for (PageId id : ids) rt.free_page(id);
    FT_ASSERT_EQ(rt.host_used(), 0u);
    FT_ASSERT_EQ(rt.nvme_used(), 0u);
    rt.shutdown();
}

FT_TEST(cpu_runtime_reclaims_stale_nvme_extents_under_pressure) {
    Config cfg = test_config();
    cfg.host_budget_bytes = 2 * cfg.page_size;
    cfg.nvme_budget_bytes = 3 * cfg.page_size;
    cfg.queue_depth = 1;
    cfg.worker_threads = 1;
    Runtime rt(cfg);
    rt.start();

    std::vector<PageId> ids;
    std::vector<std::vector<uint8_t>> expected;
    for (uint8_t value = 1; value <= 4; ++value) {
        ids.push_back(rt.allocate_page(cfg.page_size));
        expected.emplace_back(cfg.page_size, value);
    }

    // The first two writes dirty the full host tier. Writing the third page
    // consumes the final free NVMe extent while leaving that page's original
    // extent stale. The fourth write can succeed only by reclaiming that stale
    // resident-page extent; treating all allocated extents as authoritative
    // reproduces the former false "NVMe store full" failure here.
    for (std::size_t i = 0; i < ids.size(); ++i) {
        rt.write_page(ids[i], expected[i].data());
    }

    for (std::size_t i = 0; i < ids.size(); ++i) {
        std::vector<uint8_t> actual(cfg.page_size);
        rt.read_page(ids[i], actual.data());
        FT_ASSERT(actual == expected[i]);
    }
    for (PageId id : ids) rt.free_page(id);
    FT_ASSERT_EQ(rt.host_used(), 0u);
    FT_ASSERT_EQ(rt.nvme_used(), 0u);
    rt.shutdown();
}

FT_TEST(cpu_runtime_shutdown_cancels_queued_demands_without_orphans) {
    Config cfg = test_config();
    cfg.queue_depth = 1;
    cfg.worker_threads = 1;
    cfg.host_budget_bytes = cfg.page_size;
    constexpr std::size_t kPages = 8;

    for (int cycle = 0; cycle < 3; ++cycle) {
        Runtime rt(cfg);
        rt.start();
        std::vector<PageId> ids;
        for (std::size_t i = 0; i < kPages; ++i) {
            PageId id = rt.allocate_page(cfg.page_size);
            std::vector<uint8_t> data(cfg.page_size,
                                      static_cast<uint8_t>(cycle * 16 + i + 1));
            rt.write_page(id, data.data());
            rt.demote_to_nvme(id);
            ids.push_back(id);
        }

        std::atomic<std::size_t> ready{0};
        std::atomic<bool> go{false};
        std::atomic<int> cancelled{0};
        std::vector<std::exception_ptr> unexpected(kPages);
        std::vector<std::thread> threads;
        for (std::size_t i = 0; i < kPages; ++i) {
            threads.emplace_back([&, i] {
                ready.fetch_add(1, std::memory_order_release);
                while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
                try {
                    std::vector<uint8_t> out(cfg.page_size);
                    rt.read_page(ids[i], out.data());
                } catch (const Error& e) {
                    if (e.code() == ErrorCode::Cancelled ||
                        e.code() == ErrorCode::State) {
                        cancelled.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        unexpected[i] = std::current_exception();
                    }
                } catch (...) {
                    unexpected[i] = std::current_exception();
                }
            });
        }
        while (ready.load(std::memory_order_acquire) != kPages) {
            std::this_thread::yield();
        }
        go.store(true, std::memory_order_release);
        rt.shutdown();
        for (auto& thread : threads) thread.join();
        for (const auto& failure : unexpected) {
            if (failure) std::rethrow_exception(failure);
        }
        FT_ASSERT(cancelled.load(std::memory_order_relaxed) != 0);
        FT_ASSERT_EQ(rt.host_used(), 0u);
        FT_ASSERT_EQ(rt.vram_used(), 0u);
        FT_ASSERT_EQ(rt.nvme_used(), 0u);
        FT_ASSERT_EQ(rt.pages_resident_host(), 0u);
        FT_ASSERT_EQ(rt.pages_resident_nvme(), 0u);
    }
}

int main() { return ft_test::run_all("test_runtime"); }
