// Minimal FlashTier example: allocate one page, write, read, and drive it
// through the full tier hierarchy with integrity checks.

#include <cstdio>
#include <cstring>
#include <vector>

#include "flashtier/config.hpp"
#include "flashtier/integrity.hpp"
#include "flashtier/runtime.hpp"

using namespace flashtier;

int main() {
    Config cfg;
    cfg.cuda_enabled = false;  // CPU-only by default for examples
    cfg.page_size = 64 * 1024;
    cfg.host_budget_bytes = 32 * 1024 * 1024;
    cfg.nvme_budget_bytes = 32 * 1024 * 1024;
    cfg.output_dir = ".";

    Runtime rt(cfg);
    rt.start();

    const PageId id = rt.allocate_page(cfg.page_size, SemanticClass::Weight);
    std::printf("allocated page %s (%s)\n", id.to_string().c_str(),
                bytesize_to_string(cfg.page_size).c_str());

    std::vector<uint8_t> payload(cfg.page_size);
    fill_pattern(payload.data(), payload.size(), cfg.seed, id.value);
    rt.write_page(id, payload.data());
    std::printf("wrote %s (checksum %016llX)\n", bytesize_to_string(cfg.page_size).c_str(),
                static_cast<unsigned long long>(rt.page_checksum(id)));

    std::vector<uint8_t> read_back(cfg.page_size, 0);
    rt.read_page(id, read_back.data());
    std::printf("read back: %s\n",
                std::memcmp(payload.data(), read_back.data(), payload.size()) == 0
                    ? "content matches"
                    : "CONTENT MISMATCH");

    rt.demote_to_nvme(id);
    std::printf("demoted to NVMe: state=%s\n",
                page_state_name(rt.metadata(id).state));
    rt.verify_page(id);
    std::printf("verified after NVMe demotion\n");

    rt.promote(id);
    std::printf("promoted: state=%s\n", page_state_name(rt.metadata(id).state));
    rt.verify_page(id);
    std::printf("verified after promotion\n");

    rt.free_page(id);
    rt.shutdown();

    const auto agg = rt.aggregates();
    std::printf("\naggregate telemetry:\n%s", format_aggregates(agg).c_str());
    std::printf("minimal_tiered_buffer: OK\n");
    return 0;
}
