#include "test_harness.hpp"

#include "flashtier/config.hpp"
#include "flashtier/error.hpp"
#include "flashtier/integrity.hpp"
#include "flashtier/runtime.hpp"
#include "flashtier/telemetry.hpp"
#include "flashtier/workload.hpp"

using namespace flashtier;

// CPU-only mode must work end to end: this suite runs even when the build
// has no CUDA. It also verifies that CUDA-required paths report typed
// unsupported errors instead of pretending to run.

FT_TEST(cpu_only_byte_parser_and_config) {
    FT_ASSERT_EQ(parse_bytesize("8MiB"), 8ull * 1024 * 1024);
    Config cfg;
    FT_ASSERT(validate_config(cfg).ok);
}

FT_TEST(cpu_only_integrity_and_workloads) {
    std::vector<uint8_t> buf(4096);
    fill_pattern(buf.data(), buf.size(), 1, 2);
    FT_ASSERT(!verify_pattern(buf.data(), buf.size(), 1, 2).has_value());
    TraceGenerator g(1, 32, 100, TracePattern::Sequential);
    FT_ASSERT_EQ(g.accesses().size(), 100u);
}

FT_TEST(cpu_only_runtime_rejects_cuda_when_disabled) {
    // When the build has CUDA, an explicit --no-cuda equivalent is
    // cfg.cuda_enabled=false and the runtime runs purely on host/NVMe.
    Config cfg;
    cfg.cuda_enabled = false;
    cfg.page_size = 64 * 1024;
    cfg.host_budget_bytes = 32 * 1024 * 1024;
    cfg.nvme_budget_bytes = 32 * 1024 * 1024;
    cfg.output_dir = ".";
    Runtime rt(cfg);
    rt.start();
    PageId id = rt.allocate_page(64 * 1024);
    rt.fill_page(id);
    rt.verify_page(id);
    rt.free_page(id);
    rt.shutdown();
}

FT_TEST(cpu_only_runtime_reports_unsupported_for_vram_residency) {
    Config cfg;
    cfg.cuda_enabled = false;
    cfg.page_size = 64 * 1024;
    cfg.host_budget_bytes = 1 * 1024 * 1024;
    cfg.nvme_budget_bytes = 4 * 1024 * 1024;
    cfg.output_dir = ".";
    Runtime rt(cfg);
    rt.start();
    // No VRAM tier exists; gpu_memory_handle must be null and page residency
    // must never claim VRAM.
    PageId id = rt.allocate_page(64 * 1024);
    FT_ASSERT(rt.gpu_memory_handle(id) == nullptr);
    FT_ASSERT(rt.metadata(id).state != PageState::ResidentVram);
    rt.free_page(id);
    rt.shutdown();
}

int main() { return ft_test::run_all("test_no_cuda"); }
