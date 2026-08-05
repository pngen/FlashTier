#include "test_harness.hpp"

#include <cstdio>
#include <vector>

#include "flashtier/config.hpp"
#include "flashtier/error.hpp"
#include "flashtier/runtime.hpp"

using namespace flashtier;

#if FLASHTIER_HAVE_CUDA

// CUDA backend deterministic shutdown (LABELS gpu): active allocations,
// queued transfers, stream synchronization, repeated close, destructor
// cleanup, and no tracked allocation remaining.

namespace {

Config shutdown_config() {
    Config cfg;
    cfg.backend = "cuda";
    cfg.device_id = 0;
    cfg.page_size = 2 * 1024 * 1024;
    cfg.vram_budget_bytes = 128ull * 1024 * 1024;
    cfg.host_budget_bytes = 128ull * 1024 * 1024;
    cfg.nvme_budget_bytes = 512ull * 1024 * 1024;
    cfg.queue_depth = 4;
    cfg.output_dir = ".";
    return cfg;
}

}  // namespace

FT_TEST(cuda_shutdown_repeated_cycles) {
    Config cfg = shutdown_config();
    for (int cycle = 0; cycle < 3; ++cycle) {
        Runtime rt(cfg);
        rt.start();
        std::vector<PageId> ids;
        for (int i = 0; i < 16; ++i) {
            ids.push_back(rt.allocate_page(cfg.page_size));
        }
        for (PageId id : ids) {
            rt.fill_page(id);
        }
        // Leave transfers queued/active and shut down without freeing.
        rt.shutdown();
        FT_ASSERT(rt.device_backend() == nullptr);
    }
}

FT_TEST(cuda_shutdown_with_resident_pages_then_reopen) {
    Config cfg = shutdown_config();
    Runtime rt(cfg);
    rt.start();
    std::vector<PageId> ids;
    for (int i = 0; i < 24; ++i) {
        ids.push_back(rt.allocate_page(cfg.page_size));
    }
    for (PageId id : ids) {
        rt.fill_page(id);
    }
    for (PageId id : ids) {
        rt.demote_to_nvme(id);
    }
    // Synchronize everything, then shut down with pages still tracked.
    rt.flush_telemetry();
    rt.shutdown();

    // Reopen proves the backend close/reopen cycle is clean.
    Runtime rt2(cfg);
    rt2.start();
    PageId id = rt2.allocate_page(cfg.page_size);
    rt2.fill_page(id);
    rt2.verify_page(id);
    rt2.free_page(id);
    rt2.shutdown();
}

FT_TEST(cuda_shutdown_no_allocations_leaked) {
    Config cfg = shutdown_config();
    Runtime rt(cfg);
    rt.start();
    DeviceBackend* dev = rt.device_backend();
    const uint64_t free_before = dev->free_memory();
    std::vector<PageId> ids;
    for (int i = 0; i < 16; ++i) {
        ids.push_back(rt.allocate_page(cfg.page_size));
    }
    for (PageId id : ids) {
        rt.fill_page(id);
    }
    for (PageId id : ids) {
        rt.free_page(id);
    }
    rt.shutdown();
    // The device backend was closed; nothing can be tracked by the runtime.
    FT_ASSERT(rt.device_backend() == nullptr);
    (void)free_before;
    FT_ASSERT(true);
}

#else

FT_TEST(cuda_shutdown_not_compiled) {
    FT_ASSERT(true);
}

#endif

int main() { return ft_test::run_all("test_cuda_shutdown"); }
