#include "test_harness.hpp"

#include <cstdio>
#include <filesystem>
#include <vector>

#include "flashtier/telemetry.hpp"

using namespace flashtier;

namespace {

TelemetryEvent sample_event() {
    TelemetryEvent ev;
    ev.type = EventType::TransferEnd;
    ev.page_id = 42;
    ev.src = Tier::Nvme;
    ev.dst = Tier::HostPinned;
    ev.bytes = 2 * 1024 * 1024;
    ev.duration_us = 1234.5;
    ev.bandwidth_gb_s = 1.7;
    ev.policy = "predictive";
    ev.reason = "demand";
    ev.path = "nvme->host_pinned";
    ev.vram_used = 100;
    ev.cache_hit = false;
    return ev;
}

}  // namespace

FT_TEST(jsonl_serialization_contains_key_fields) {
    TelemetryEvent ev = sample_event();
    ev.type = EventType::PageRead;  // cache_hit is only serialized for access events
    const std::string line = telemetry_event_to_jsonl(ev);
    FT_ASSERT(line.find("\"schema\":\"1.0\"") != std::string::npos);
    FT_ASSERT(line.find("\"event\":\"page_read\"") != std::string::npos);
    FT_ASSERT(line.find("\"page_id\":42") != std::string::npos);
    FT_ASSERT(line.find("\"src_tier\":\"nvme\"") != std::string::npos);
    FT_ASSERT(line.find("\"dst_tier\":\"host_pinned\"") != std::string::npos);
    FT_ASSERT(line.find("\"bytes\":2097152") != std::string::npos);
    FT_ASSERT(line.find("\"policy\":\"predictive\"") != std::string::npos);
    FT_ASSERT(line.find("\"path\":\"nvme->host_pinned\"") != std::string::npos);
    FT_ASSERT(line.find("\"cache_hit\":false") != std::string::npos);
    // One JSON object per line.
    FT_ASSERT(line.find('\n') == std::string::npos);
}

FT_TEST(jsonl_escapes_strings) {
    TelemetryEvent ev = sample_event();
    ev.reason = "line\nbreak \"quoted\"";
    const std::string line = telemetry_event_to_jsonl(ev);
    FT_ASSERT(line.find("line\\nbreak") != std::string::npos);
    FT_ASSERT(line.find("\\\"quoted\\\"") != std::string::npos);
}

FT_TEST(sink_writes_human_and_jsonl_files) {
    const std::string dir = std::filesystem::temp_directory_path().string();
    const std::string human = dir + "/ft-test-telemetry-human.txt";
    const std::string jsonl = dir + "/ft-test-telemetry.jsonl";
    {
        TelemetrySink sink(human, jsonl, false);
        sink.emit(sample_event());
        sink.emit(sample_event());
        sink.flush();
    }
    std::FILE* f = std::fopen(jsonl.c_str(), "rb");
    FT_ASSERT(f != nullptr);
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fclose(f);
    FT_ASSERT(size > 0);
    std::remove(human.c_str());
    std::remove(jsonl.c_str());
}

FT_TEST(sink_assigns_monotonic_sequences) {
    TelemetrySink sink("", "", false);
    const uint64_t s1 = sink.emit_sequenced(sample_event());
    const uint64_t s2 = sink.emit_sequenced(sample_event());
    FT_ASSERT_EQ(s1, 1u);
    FT_ASSERT_EQ(s2, 2u);
    FT_ASSERT_EQ(sink.events_emitted(), 2u);
}

FT_TEST(aggregator_computes_hit_rates_and_path_stats) {
    TelemetryAggregator agg;
    for (int i = 0; i < 60; ++i) {
        TelemetryEvent read;
        read.type = EventType::PageRead;
        read.src = Tier::Vram;
        read.cache_hit = true;
        read.page_id = static_cast<uint64_t>(i);
        agg.record(read);
    }
    for (int i = 0; i < 20; ++i) {
        TelemetryEvent read;
        read.type = EventType::PageRead;
        read.src = Tier::HostPinned;
        read.cache_hit = false;
        read.page_id = static_cast<uint64_t>(i);
        agg.record(read);
    }
    for (int i = 0; i < 20; ++i) {
        TelemetryEvent read;
        read.type = EventType::PageRead;
        read.src = Tier::Nvme;
        read.cache_hit = false;
        read.page_id = static_cast<uint64_t>(i);
        agg.record(read);
    }
    for (int i = 0; i < 50; ++i) {
        TelemetryEvent t;
        t.type = EventType::TransferEnd;
        t.src = Tier::Nvme;
        t.dst = Tier::Vram;
        t.bytes = 1024 * 1024;
        t.duration_us = 1000.0 + i;
        agg.record(t);
    }
    TelemetryEvent df;
    df.type = EventType::DemandFault;
    agg.record(df);
    TelemetryEvent ev;
    ev.type = EventType::Evict;
    ev.reason = "writeback";
    ev.dst = Tier::Nvme;
    agg.record(ev);

    const auto s = agg.summary();
    FT_ASSERT_EQ(s.total_accesses, 100u);
    FT_ASSERT_EQ(s.vram_hits, 60u);
    FT_ASSERT_EQ(s.host_hits, 20u);
    FT_ASSERT_EQ(s.nvme_hits, 20u);
    FT_ASSERT_EQ(s.demand_faults, 1u);
    FT_ASSERT_EQ(s.evictions, 1u);
    FT_ASSERT_EQ(s.writebacks, 1u);
    FT_ASSERT_EQ(s.prefetches_issued, 0u);
    auto it = s.paths.find("nvme->vram");
    FT_ASSERT(it != s.paths.end());
    FT_ASSERT_EQ(it->second.count, 50u);
    FT_ASSERT_EQ(it->second.bytes, 50ull * 1024 * 1024);
    FT_ASSERT(it->second.p50_us >= 1000.0);
    FT_ASSERT(it->second.p99_us > it->second.p50_us);
    FT_ASSERT(it->second.bandwidth_gb_s > 0.0);
}

FT_TEST(aggregator_prefetch_accounting) {
    TelemetryAggregator agg;
    TelemetryEvent issue;
    issue.type = EventType::PrefetchIssue;
    agg.record(issue);
    TelemetryEvent hit;
    hit.type = EventType::PrefetchHit;
    hit.prefetch_hit = true;
    agg.record(hit);
    TelemetryEvent waste;
    waste.type = EventType::PrefetchWaste;
    waste.prefetch_waste = true;
    agg.record(waste);
    const auto s = agg.summary();
    FT_ASSERT_EQ(s.prefetches_issued, 1u);
    FT_ASSERT_EQ(s.prefetch_hits, 1u);
    FT_ASSERT_EQ(s.prefetch_waste, 1u);
    FT_ASSERT_EQ(s.prefetch_hit_rate(), 1.0);
}

int main() { return ft_test::run_all("test_telemetry"); }
