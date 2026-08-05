#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

#include "flashtier/allocator.hpp"
#include "flashtier/backends/device_backend.hpp"
#include "flashtier/backends/host_backend.hpp"
#include "flashtier/backends/nvme_backend.hpp"
#include "flashtier/backends/storage_backend.hpp"
#include "flashtier/config.hpp"
#include "flashtier/device_info.hpp"
#include "flashtier/page.hpp"
#include "flashtier/page_table.hpp"
#include "flashtier/planner.hpp"
#include "flashtier/policy.hpp"
#include "flashtier/telemetry.hpp"

namespace flashtier {

// The governed three-tier runtime. Owns the page table, budgets, backends,
// policy, planner, transfer engine, and telemetry.
//
// The device-memory tier (user-facing name Tier::Vram) is driven
// exclusively through the vendor-neutral DeviceBackend contract; the
// runtime contains no vendor API calls. See
// include/flashtier/backends/device_backend.hpp.
//
// Concurrency model (bounded and explicit):
//  - one state lock guards the page table, budgets, and queues;
//  - a bounded worker pool (queue_depth threads) executes page-level
//    transfer chains; chains never wait on other chains;
//  - device copies are async on a backend stream with events;
//  - NVMe ops are async (IOCP / threaded);
//  - deterministic shutdown: stop flag, drain, join, close backends.
class Runtime {
public:
    explicit Runtime(const Config& cfg);
    ~Runtime();

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    // Probe the machine, select the device backend, open backends and
    // stores. Throws ErrorCode::Unsupported when required capabilities are
    // missing; throws ErrorCode::Config for invalid backend selection.
    void start();
    void shutdown();

    // ---- page management --------------------------------------------------
    PageId allocate_page(uint64_t logical_bytes,
                         SemanticClass cls = SemanticClass::Generic,
                         bool pinned = false,
                         int64_t execution_order_hint = -1,
                         int64_t reuse_distance_estimate = -1);
    void free_page(PageId id);

    // ---- access (synchronous; full-page copies) ---------------------------
    void write_page(PageId id, const void* data);
    void read_page(PageId id, void* out);

    // ---- residency control -------------------------------------------------
    void promote(PageId id);                 // ensure resident in device memory
    void demote_to_host(PageId id);          // ensure resident in host
    void demote_to_nvme(PageId id);          // ensure resident in NVMe
    void evict(PageId id);                   // demote to the cheapest tier
    void pin(PageId id, bool pinned);

    // ---- prefetch ----------------------------------------------------------
    void prefetch(const std::vector<PageId>& ids);
    void prefetch_sequential(PageId start, uint64_t count);

    // ---- integrity ---------------------------------------------------------
    void fill_page(PageId id);               // deterministic pattern write
    void verify_page(PageId id);             // full deterministic verify
    uint64_t page_checksum(PageId id) const; // stored FNV-1a 64

    // ---- introspection -----------------------------------------------------
    PageMetadata metadata(PageId id) const;
    const Config& config() const noexcept { return cfg_; }
    SystemInfo system_info() const noexcept { return info_; }
    TelemetryAggregates aggregates() const;
    uint64_t access_sequence() const noexcept { return access_seq_.load(); }

    // Selected device backend (null when the runtime runs host+NVMe only).
    DeviceBackend* device_backend() const { return device_.get(); }
    std::string selected_backend() const;
    std::string backend_selection_reason() const { return backend_reason_; }

    uint64_t vram_used() const;
    uint64_t host_used() const;
    uint64_t nvme_used() const;
    uint64_t pages_resident_vram() const;
    uint64_t pages_resident_host() const;
    uint64_t pages_resident_nvme() const;

    // Benchmarks emit their own events through this channel.
    void emit_telemetry(TelemetryEvent ev);
    void flush_telemetry();

    // Device-memory handle of a VRAM-resident page (for diagnostics).
    void* device_memory_handle(PageId id);
    // Historical alias kept for compatibility.
    void* gpu_memory_handle(PageId id) { return device_memory_handle(id); }

private:
    // ---- transfer chain steps (worker-executed) ---------------------------
    void ensure_vram_headroom(uint64_t need_bytes);
    void* claim_host_buffer(uint64_t bytes, PageId for_page);
    void evict_vram_to_host(PageId id);
    void evict_host_to_nvme(PageId id);
    void load_nvme_to_host(PageId id);
    void promote_host_to_vram(PageId id);
    void load_page(PageId id, Tier target);  // demand or prefetch entry

    // ---- helpers -----------------------------------------------------------
    void emit_event(TelemetryEvent ev);
    PageMetadata wait_settled(PageId id);
    void mark_prefetched(PageId id);
    bool consume_prefetch_mark(PageId id);
    void remove_queued_prefetch(PageId id);
    void record_transfer(EventType type, PageId id, Tier src, Tier dst,
                         uint64_t bytes, double us, const std::string& reason,
                         const std::string& path);
    void record_error(const Error& e);
    void worker_loop();
    void submit_demand(PageId id, Tier target);
    void submit_prefetch(PageId id, Tier target);
    std::string transfer_path_str(Tier src, Tier dst) const;
    bool vram_tier_exists() const;
    void* device_alloc(std::size_t bytes);
    void device_free(void* ptr);
    void device_copy_h2d(void* dst, const void* src, std::size_t bytes);
    void device_copy_d2h(void* dst, const void* src, std::size_t bytes);
    void device_sync();

    // ---- members -----------------------------------------------------------
    Config cfg_;
    SystemInfo info_;
    PageTable table_;
    mutable std::mutex state_mu_;
    std::condition_variable state_cv_;
    std::mutex headroom_mu_;  // serializes host-buffer claims + evictions

    TierBudget vram_budget_;
    TierBudget host_budget_;
    TierBudget nvme_budget_;
    HostBackend host_;
    std::unique_ptr<Planner> planner_;
    std::unique_ptr<Policy> policy_;

    std::unique_ptr<DeviceBackend> device_;  // vendor-neutral accelerator
    DeviceStream* device_stream_ = nullptr;  // runtime transfer stream
    DeviceEvent* device_event_a_ = nullptr;  // timing events
    DeviceEvent* device_event_b_ = nullptr;
    std::string backend_name_;
    std::string backend_reason_;

    std::unique_ptr<StorageBackend> nvme_;
    bool store_opened_ = false;
    std::string store_path_;

    // transfer engine
    struct Request {
        enum class Kind { Demand, Prefetch };
        Kind kind = Kind::Demand;
        PageId page;
        Tier target = Tier::Vram;
        std::shared_ptr<void> completion;  // shared_ptr<promise<void>> marker
        std::string reason;
    };
    std::deque<Request> queue_;
    std::mutex queue_mu_;
    std::condition_variable queue_cv_;
    std::vector<std::thread> workers_;
    bool stopping_ = false;

    std::atomic<uint64_t> access_seq_{0};
    std::atomic<uint64_t> next_page_id_{1};
    std::unordered_set<uint64_t> prefetched_pages_;
    std::unordered_map<uint64_t, std::weak_ptr<void>> in_flight_ops_;

    std::shared_ptr<TelemetrySink> sink_;
    std::unique_ptr<TelemetryAggregator> agg_;
    uint64_t last_seq_ = 0;
};

}  // namespace flashtier
