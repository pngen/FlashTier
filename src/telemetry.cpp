#include "flashtier/telemetry.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "flashtier/error.hpp"

namespace flashtier {

const char* event_type_name(EventType type) noexcept {
    switch (type) {
        case EventType::PageAlloc: return "page_alloc";
        case EventType::PageFree: return "page_free";
        case EventType::PageWrite: return "page_write";
        case EventType::PageRead: return "page_read";
        case EventType::TransferBegin: return "transfer_begin";
        case EventType::TransferEnd: return "transfer_end";
        case EventType::Evict: return "evict";
        case EventType::PrefetchIssue: return "prefetch_issue";
        case EventType::PrefetchHit: return "prefetch_hit";
        case EventType::PrefetchWaste: return "prefetch_waste";
        case EventType::DemandFault: return "demand_fault";
        case EventType::Stall: return "stall";
        case EventType::IntegrityCheck: return "integrity_check";
        case EventType::ErrorEvent: return "error";
        case EventType::BenchmarkBegin: return "benchmark_begin";
        case EventType::BenchmarkEnd: return "benchmark_end";
        case EventType::StoreOp: return "store_op";
        case EventType::ConfigEvent: return "config";
    }
    return "unknown";
}

namespace {

void json_escape(std::string& out, const std::string& s) {
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
}

void append_str(std::string& out, const char* key, const std::string& val, bool& first) {
    if (val.empty()) return;
    if (!first) out += ",";
    first = false;
    out += "\"";
    out += key;
    out += "\":\"";
    json_escape(out, val);
    out += "\"";
}

void append_u64(std::string& out, const char* key, uint64_t val, bool& first) {
    if (val == 0) return;
    if (!first) out += ",";
    first = false;
    out += "\"";
    out += key;
    out += "\":";
    out += std::to_string(val);
}

void append_i64(std::string& out, const char* key, int64_t val, bool& first) {
    if (val == 0) return;
    append_u64(out, key, static_cast<uint64_t>(val), first);
}

void append_dbl(std::string& out, const char* key, double val, bool& first) {
    if (val == 0.0) return;
    if (!first) out += ",";
    first = false;
    out += "\"";
    out += key;
    out += "\":";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.6g", val);
    out += buf;
}

void append_bool(std::string& out, const char* key, bool val, bool& first) {
    if (!first) out += ",";
    first = false;
    out += "\"";
    out += key;
    out += "\":";
    out += val ? "true" : "false";
}

void append_tier(std::string& out, const char* key, Tier t, bool& first) {
    if (t == Tier::None) return;
    append_str(out, key, tier_name(t), first);
}

int64_t wall_clock_us() {
    using namespace std::chrono;
    return duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
}

}  // namespace

std::string telemetry_event_to_jsonl(const TelemetryEvent& ev) {
    std::string out = "{";
    bool first = true;

    out += "\"schema\":\""; out += kTelemetrySchemaVersion; out += "\""; first = false;

    append_i64(out, "timestamp_us", ev.timestamp_us, first);
    append_u64(out, "sequence", ev.sequence, first);
    append_str(out, "event", event_type_name(ev.type), first);
    append_u64(out, "page_id", ev.page_id, first);
    append_tier(out, "src_tier", ev.src, first);
    append_tier(out, "dst_tier", ev.dst, first);
    append_u64(out, "bytes", ev.bytes, first);
    append_dbl(out, "duration_us", ev.duration_us, first);
    append_dbl(out, "bandwidth_gb_s", ev.bandwidth_gb_s, first);
    append_str(out, "policy", ev.policy, first);
    append_str(out, "reason", ev.reason, first);
    append_str(out, "path", ev.path, first);
    append_u64(out, "queue_depth", ev.queue_depth, first);
    append_u64(out, "vram_used", ev.vram_used, first);
    append_u64(out, "vram_free", ev.vram_free, first);
    append_u64(out, "host_used", ev.host_used, first);
    append_u64(out, "host_free", ev.host_free, first);
    append_u64(out, "nvme_used", ev.nvme_used, first);
    append_u64(out, "nvme_free", ev.nvme_free, first);

    if (ev.type == EventType::PageRead || ev.type == EventType::PageWrite ||
        ev.type == EventType::DemandFault) {
        append_bool(out, "cache_hit", ev.cache_hit, first);
    }
    if (ev.type == EventType::PrefetchHit || ev.prefetch_hit) {
        append_bool(out, "prefetch_hit", ev.prefetch_hit, first);
    }
    if (ev.type == EventType::PrefetchWaste || ev.prefetch_waste) {
        append_bool(out, "prefetch_waste", ev.prefetch_waste, first);
    }
    append_dbl(out, "stall_us", ev.stall_us, first);
    if (ev.type == EventType::IntegrityCheck) {
        append_bool(out, "checksum_ok", ev.checksum_ok, first);
        append_str(out, "checksum_expected", ev.checksum_expected, first);
        append_str(out, "checksum_actual", ev.checksum_actual, first);
    }
    append_str(out, "cuda_error", ev.cuda_error, first);
    append_str(out, "error_code", ev.error_code, first);
    append_i64(out, "logical_size", ev.logical_size, first);
    append_str(out, "op", ev.op, first);
    append_dbl(out, "throughput", ev.throughput, first);

    out += "}";
    return out;
}

namespace {

std::string human_line(const TelemetryEvent& ev) {
    std::ostringstream os;
    os << "[" << ev.sequence << "] " << event_type_name(ev.type);
    if (ev.page_id != 0) os << " page=" << ev.page_id;
    if (ev.src != Tier::None && ev.dst != Tier::None) {
        os << " " << tier_name(ev.src) << "->" << tier_name(ev.dst);
    }
    if (ev.bytes != 0) os << " bytes=" << ev.bytes;
    if (ev.duration_us != 0.0) os << " " << ev.duration_us << "us";
    if (ev.bandwidth_gb_s != 0.0) os << " " << ev.bandwidth_gb_s << " GB/s";
    if (!ev.reason.empty()) os << " reason=" << ev.reason;
    if (!ev.error_code.empty()) os << " error=" << ev.error_code;
    if (ev.type == EventType::IntegrityCheck) {
        os << (ev.checksum_ok ? " checksum=ok" : " checksum=MISMATCH");
    }
    return os.str();
}

}  // namespace

TelemetrySink::TelemetrySink(std::string human_path, std::string jsonl_path, bool to_stdout)
    : human_path_(std::move(human_path)),
      jsonl_path_(std::move(jsonl_path)),
      to_stdout_(to_stdout) {
    if (!human_path_.empty()) {
        auto* s = new std::ofstream(human_path_);
        if (!s->is_open()) {
            delete s;
            throw Error(ErrorCode::Io, "cannot open human telemetry file", human_path_);
        }
        human_stream_ = s;
    }
    if (!jsonl_path_.empty()) {
        auto* s = new std::ofstream(jsonl_path_);
        if (!s->is_open()) {
            delete s;
            throw Error(ErrorCode::Io, "cannot open JSONL telemetry file", jsonl_path_);
        }
        jsonl_stream_ = s;
    }
}

TelemetrySink::~TelemetrySink() {
    flush();
    if (human_stream_) delete static_cast<std::ofstream*>(human_stream_);
    if (jsonl_stream_) delete static_cast<std::ofstream*>(jsonl_stream_);
}

void TelemetrySink::emit(TelemetryEvent ev) {
    (void)emit_sequenced(std::move(ev));
}

uint64_t TelemetrySink::emit_sequenced(TelemetryEvent ev) {
    std::lock_guard lock(mu_);
    ev.sequence = ++sequence_;
    ev.timestamp_us = wall_clock_us();
    ++emitted_;
    if (human_stream_) {
        *static_cast<std::ofstream*>(human_stream_) << human_line(ev) << '\n';
    }
    if (jsonl_stream_) {
        *static_cast<std::ofstream*>(jsonl_stream_) << telemetry_event_to_jsonl(ev) << '\n';
    }
    if (to_stdout_) {
        const std::string line = human_line(ev) + "\n";
        std::fwrite(line.c_str(), 1, line.size(), stdout);
        std::fflush(stdout);
    }
    return ev.sequence;
}

void TelemetrySink::flush() {
    std::lock_guard lock(mu_);
    if (human_stream_) static_cast<std::ofstream*>(human_stream_)->flush();
    if (jsonl_stream_) static_cast<std::ofstream*>(jsonl_stream_)->flush();
}

// ---------------------------------------------------------------------------
// Aggregator
// ---------------------------------------------------------------------------

namespace {

std::string path_key(Tier src, Tier dst) {
    return std::string(tier_name(src)) + "->" + tier_name(dst);
}

double percentile(std::vector<double>& samples, double p) {
    if (samples.empty()) return 0.0;
    std::sort(samples.begin(), samples.end());
    const std::size_t idx = static_cast<std::size_t>(p * static_cast<double>(samples.size() - 1));
    return samples[idx];
}

}  // namespace

void TelemetryAggregator::record(const TelemetryEvent& ev) {
    std::lock_guard lock(mu_);

    if (ev.type == EventType::TransferEnd && ev.bytes != 0 && ev.duration_us > 0.0) {
        auto& ps = agg_.paths[path_key(ev.src, ev.dst)];
        ps.bytes += ev.bytes;
        ++ps.count;
        ps.total_us += ev.duration_us;
        if (ps.samples_us.size() < 1000000) ps.samples_us.push_back(ev.duration_us);
    }

    switch (ev.type) {
        case EventType::PageRead:
        case EventType::PageWrite:
            ++agg_.total_accesses;
            if (ev.cache_hit || ev.src == Tier::Vram) {
                ++agg_.vram_hits;
            } else if (ev.src == Tier::HostPinned || ev.src == Tier::HostPageable) {
                ++agg_.host_hits;
            } else if (ev.src == Tier::Nvme) {
                ++agg_.nvme_hits;
            }
            break;
        case EventType::DemandFault:
            ++agg_.demand_faults;
            break;
        case EventType::Evict:
            ++agg_.evictions;
            if (ev.reason == "writeback" || ev.dst == Tier::Nvme) ++agg_.writebacks;
            break;
        case EventType::PrefetchIssue:
            ++agg_.prefetches_issued;
            break;
        case EventType::PrefetchHit:
            ++agg_.prefetch_hits;
            break;
        case EventType::PrefetchWaste:
            ++agg_.prefetch_waste;
            break;
        case EventType::Stall:
            agg_.total_stall_us += ev.stall_us;
            ++agg_.stall_events;
            break;
        case EventType::IntegrityCheck:
            ++agg_.integrity_checks;
            if (!ev.checksum_ok) ++agg_.integrity_failures;
            break;
        case EventType::ErrorEvent:
            ++agg_.errors;
            break;
        default:
            break;
    }

    if (ev.type == EventType::PageAlloc) {
        if (ev.dst == Tier::Vram) agg_.max_vram_resident = std::max(agg_.max_vram_resident, ev.vram_used);
        if (ev.dst == Tier::HostPinned) agg_.max_host_resident = std::max(agg_.max_host_resident, ev.host_used);
        if (ev.dst == Tier::Nvme) agg_.max_nvme_resident = std::max(agg_.max_nvme_resident, ev.nvme_used);
    }
}

TelemetryAggregates TelemetryAggregator::summary() const {
    std::lock_guard lock(mu_);
    TelemetryAggregates out = agg_;
    for (auto& [k, ps] : out.paths) {
        (void)k;
        ps.p50_us = percentile(ps.samples_us, 0.50);
        ps.p90_us = percentile(ps.samples_us, 0.90);
        ps.p99_us = percentile(ps.samples_us, 0.99);
        if (ps.total_us > 0.0) {
            ps.bandwidth_gb_s = static_cast<double>(ps.bytes) / 1e9 / (ps.total_us / 1e6);
        }
        ps.samples_us.clear();
    }
    return out;
}

void TelemetryAggregator::reset() {
    std::lock_guard lock(mu_);
    agg_ = TelemetryAggregates{};
}

double TelemetryAggregates::vram_hit_rate() const {
    return total_accesses == 0 ? 0.0 : static_cast<double>(vram_hits) / static_cast<double>(total_accesses);
}
double TelemetryAggregates::host_hit_rate() const {
    return total_accesses == 0 ? 0.0 : static_cast<double>(host_hits) / static_cast<double>(total_accesses);
}
double TelemetryAggregates::nvme_hit_rate() const {
    return total_accesses == 0 ? 0.0 : static_cast<double>(nvme_hits) / static_cast<double>(total_accesses);
}
double TelemetryAggregates::prefetch_hit_rate() const {
    return prefetches_issued == 0 ? 0.0
                                  : static_cast<double>(prefetch_hits) / static_cast<double>(prefetches_issued);
}

std::string format_aggregates(const TelemetryAggregates& agg) {
    std::ostringstream os;
    os << "  transfers per path:\n";
    if (agg.paths.empty()) {
        os << "    (none)\n";
    } else {
        for (const auto& [k, ps] : agg.paths) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                          "    %-20s count=%7llu bytes=%10llu avg=%8.1fus p50=%8.1fus p90=%8.1fus p99=%8.1fus bw=%6.2f GB/s\n",
                          k.c_str(),
                          static_cast<unsigned long long>(ps.count),
                          static_cast<unsigned long long>(ps.bytes),
                          ps.count ? ps.total_us / static_cast<double>(ps.count) : 0.0,
                          ps.p50_us, ps.p90_us, ps.p99_us, ps.bandwidth_gb_s);
            os << buf;
        }
    }
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "  accesses=%llu vram_hits=%llu(%.1f%%) host_hits=%llu(%.1f%%) nvme_hits=%llu(%.1f%%)\n"
                  "  demand_faults=%llu evictions=%llu writebacks=%llu\n"
                  "  prefetch issued=%llu hits=%llu waste=%llu (hit rate %.1f%%)\n"
                  "  stalls=%llu total_stalled=%.1fus\n"
                  "  integrity_checks=%llu failures=%llu errors=%llu\n"
                  "  max_resident vram=%llu host=%llu nvme=%llu\n"
                  "  promotions=%llu demotions=%llu\n",
                  static_cast<unsigned long long>(agg.total_accesses),
                  static_cast<unsigned long long>(agg.vram_hits), agg.vram_hit_rate() * 100.0,
                  static_cast<unsigned long long>(agg.host_hits), agg.host_hit_rate() * 100.0,
                  static_cast<unsigned long long>(agg.nvme_hits), agg.nvme_hit_rate() * 100.0,
                  static_cast<unsigned long long>(agg.demand_faults),
                  static_cast<unsigned long long>(agg.evictions),
                  static_cast<unsigned long long>(agg.writebacks),
                  static_cast<unsigned long long>(agg.prefetches_issued),
                  static_cast<unsigned long long>(agg.prefetch_hits),
                  static_cast<unsigned long long>(agg.prefetch_waste),
                  agg.prefetch_hit_rate() * 100.0,
                  static_cast<unsigned long long>(agg.stall_events), agg.total_stall_us,
                  static_cast<unsigned long long>(agg.integrity_checks),
                  static_cast<unsigned long long>(agg.integrity_failures),
                  static_cast<unsigned long long>(agg.errors),
                  static_cast<unsigned long long>(agg.max_vram_resident),
                  static_cast<unsigned long long>(agg.max_host_resident),
                  static_cast<unsigned long long>(agg.max_nvme_resident),
                  static_cast<unsigned long long>(agg.total_promotions),
                  static_cast<unsigned long long>(agg.total_demotions));
    os << buf;
    return os.str();
}

}  // namespace flashtier
