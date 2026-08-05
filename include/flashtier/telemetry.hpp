#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "flashtier/tier.hpp"

namespace flashtier {

constexpr const char* kTelemetrySchemaVersion = "1.0";

enum class EventType : int {
    PageAlloc = 0,
    PageFree = 1,
    PageWrite = 2,
    PageRead = 3,
    TransferBegin = 4,
    TransferEnd = 5,
    Evict = 6,
    PrefetchIssue = 7,
    PrefetchHit = 8,
    PrefetchWaste = 9,
    DemandFault = 10,
    Stall = 11,
    IntegrityCheck = 12,
    ErrorEvent = 13,
    BenchmarkBegin = 14,
    BenchmarkEnd = 15,
    StoreOp = 16,
    ConfigEvent = 17,
};

const char* event_type_name(EventType type) noexcept;

// One structured telemetry event. All fields are optional per event type;
// serialization emits present fields only.
struct TelemetryEvent {
    int64_t timestamp_us = 0;      // wall-clock microseconds
    uint64_t sequence = 0;         // monotonic sequence number
    EventType type = EventType::PageAlloc;
    uint64_t page_id = 0;          // 0 = not applicable
    Tier src = Tier::None;
    Tier dst = Tier::None;
    uint64_t bytes = 0;
    double duration_us = 0.0;
    double bandwidth_gb_s = 0.0;
    std::string policy;
    std::string reason;
    std::string path;              // human-readable transfer path
    uint32_t queue_depth = 0;
    uint64_t vram_used = 0;
    uint64_t vram_free = 0;
    uint64_t host_used = 0;
    uint64_t host_free = 0;
    uint64_t nvme_used = 0;
    uint64_t nvme_free = 0;
    bool cache_hit = false;
    bool prefetch_hit = false;
    bool prefetch_waste = false;
    double stall_us = 0.0;
    bool checksum_ok = true;
    std::string checksum_expected;
    std::string checksum_actual;
    std::string cuda_error;
    std::string error_code;
    int64_t logical_size = 0;
    std::string op;                // store operation name
    double throughput = 0.0;       // ops/s, benchmark events
};

// One JSON line; the only JSON writer in the project (local telemetry only).
std::string telemetry_event_to_jsonl(const TelemetryEvent& ev);

// Thread-safe sink writing human-readable and JSON Lines forms. All output
// is local; nothing is transmitted.
class TelemetrySink {
public:
    TelemetrySink(std::string human_path, std::string jsonl_path, bool to_stdout);
    ~TelemetrySink();

    TelemetrySink(const TelemetrySink&) = delete;
    TelemetrySink& operator=(const TelemetrySink&) = delete;

    void emit(TelemetryEvent ev);  // assigns timestamp + sequence
    uint64_t emit_sequenced(TelemetryEvent ev);  // same, returns sequence
    void flush();
    uint64_t events_emitted() const noexcept { return emitted_; }

private:
    mutable std::mutex mu_;
    std::string human_path_;
    std::string jsonl_path_;
    bool to_stdout_;
    void* human_stream_ = nullptr;  // std::ofstream*
    void* jsonl_stream_ = nullptr;  // std::ofstream*
    uint64_t emitted_ = 0;
    uint64_t sequence_ = 0;
};

// Aggregated metrics across a run.
struct TelemetryAggregates {
    // Per path: source -> destination.
    struct PathStats {
        uint64_t bytes = 0;
        uint64_t count = 0;
        double total_us = 0.0;
        std::vector<double> samples_us;  // for percentiles
        double p50_us = 0.0;
        double p90_us = 0.0;
        double p99_us = 0.0;
        double bandwidth_gb_s = 0.0;
    };

    std::map<std::string, PathStats> paths;

    uint64_t vram_hits = 0;
    uint64_t host_hits = 0;
    uint64_t nvme_hits = 0;
    uint64_t total_accesses = 0;
    uint64_t demand_faults = 0;
    uint64_t evictions = 0;
    uint64_t writebacks = 0;
    uint64_t prefetches_issued = 0;
    uint64_t prefetch_hits = 0;
    uint64_t prefetch_waste = 0;
    double total_stall_us = 0.0;
    uint64_t stall_events = 0;
    uint64_t integrity_checks = 0;
    uint64_t integrity_failures = 0;
    uint64_t errors = 0;
    uint64_t max_vram_resident = 0;
    uint64_t max_host_resident = 0;
    uint64_t max_nvme_resident = 0;
    uint64_t total_promotions = 0;
    uint64_t total_demotions = 0;

    double vram_hit_rate() const;
    double host_hit_rate() const;
    double nvme_hit_rate() const;
    double prefetch_hit_rate() const;
};

// Thread-safe aggregator consuming events.
class TelemetryAggregator {
public:
    void record(const TelemetryEvent& ev);
    TelemetryAggregates summary() const;
    void reset();

private:
    mutable std::mutex mu_;
    TelemetryAggregates agg_;
};

// Format aggregate summary as a human-readable block.
std::string format_aggregates(const TelemetryAggregates& agg);

}  // namespace flashtier
