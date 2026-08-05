#include "flashtier/runtime.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <future>
#include <memory>
#include <optional>
#include <vector>

#include "flashtier/backends/backend_registry.hpp"
#include "flashtier/error.hpp"
#include "flashtier/integrity.hpp"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <process.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace flashtier {

namespace {

using Clock = std::chrono::steady_clock;

uint64_t round_down(uint64_t v, uint64_t page_size) {
    return v - (v % page_size);
}

std::string hex_u64(uint64_t v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%016llX", static_cast<unsigned long long>(v));
    return buf;
}

int process_id() {
#if defined(_WIN32)
    return _getpid();
#else
    return static_cast<int>(::getpid());
#endif
}

}  // namespace

Runtime::Runtime(const Config& cfg) : cfg_(cfg) {}

Runtime::~Runtime() {
    shutdown();
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void Runtime::start() {
    // ---- device backend selection ----------------------------------------
    BackendRegistry& registry = BackendRegistry::instance();
    std::string backend_name = cfg_.backend.empty() ? "auto" : cfg_.backend;
    if (!cfg_.cuda_enabled) {
        // --no-cuda requests CPU-only mode; contradicting explicit
        // non-cpu backends are a configuration error.
        if (!cfg_.backend.empty() && cfg_.backend != "auto" && cfg_.backend != "cpu") {
            throw Error(ErrorCode::Config,
                        "--no-cuda conflicts with explicit backend '" + cfg_.backend + "'");
        }
        backend_name = "cpu";
        backend_reason_ = "cpu backend: CPU-only mode requested (--no-cuda)";
    } else if (cfg_.backend.empty() || cfg_.backend == "auto") {
        backend_name = registry.select_automatic(backend_reason_);
    } else {
        if (!registry.has(backend_name)) {
            throw Error(ErrorCode::Config,
                        "unknown device backend '" + backend_name + "'");
        }
        backend_reason_ = "explicit --backend " + backend_name;
    }
    backend_name_ = backend_name;

    info_ = probe_system(cfg_.device_id, cfg_.nvme_path);

    // CPU-only mode (backend "cpu") runs the host + NVMe tiers without an
    // accelerator tier, exactly as before; no GPU execution is claimed.
    const bool cpu_only_mode = backend_name == "cpu";

    uint64_t vram_limit = 0;
    DeviceInfo dev_info;
    if (!cpu_only_mode) {
        device_ = registry.create(backend_name);
        device_->open(cfg_.device_id);
        dev_info = device_->device_info();
        const uint64_t free_device = device_->free_memory();
        if (free_device == 0) {
            throw Error(ErrorCode::Unsupported,
                        "cannot determine free device memory; refusing to guess budgets");
        }
        if (cfg_.vram_budget_bytes != 0) {
            vram_limit = round_down(cfg_.vram_budget_bytes, cfg_.page_size);
        } else {
            vram_limit = round_down(
                static_cast<uint64_t>(static_cast<double>(free_device) * cfg_.auto_vram_fraction),
                cfg_.page_size);
        }
        if (vram_limit > free_device) {
            throw Error(ErrorCode::Budget,
                        "device-memory budget exceeds currently free device memory",
                        "budget " + std::to_string(vram_limit) +
                            " free " + std::to_string(free_device));
        }
        device_stream_ = device_->create_stream();
        device_event_a_ = device_->create_event();
        device_event_b_ = device_->create_event();
    }

    const uint64_t host_limit =
        cfg_.host_budget_bytes != 0
            ? round_down(cfg_.host_budget_bytes, cfg_.page_size)
            : round_down(
                  static_cast<uint64_t>(static_cast<double>(info_.free_ram_bytes) * cfg_.auto_host_fraction),
                  cfg_.page_size);
    if (host_limit == 0) {
        throw Error(ErrorCode::Budget, "effective host budget is zero");
    }

    // Integrated / shared-memory accelerators share physical RAM with the
    // host: never double-count the same memory, keep an OS reserve, and
    // clamp the device budget so device + host budgets fit free RAM.
    if (dev_info.memory_shared && vram_limit != 0) {
        const uint64_t os_reserve = static_cast<uint64_t>(
            static_cast<double>(info_.free_ram_bytes) * 0.10);
        const uint64_t ram_room =
            info_.free_ram_bytes > host_limit + os_reserve
                ? info_.free_ram_bytes - host_limit - os_reserve
                : 0;
        vram_limit = std::min(vram_limit, round_down(ram_room, cfg_.page_size));
        if (vram_limit == 0) {
            // No room for a meaningful device tier; the runtime degrades to
            // host + NVMe and says so explicitly.
            backend_reason_ += "; integrated GPU: no RAM headroom for a device tier, "
                               "device tier disabled (host+NVMe)";
        }
    }

    const uint64_t nvme_limit =
        cfg_.nvme_budget_bytes != 0
            ? round_down(cfg_.nvme_budget_bytes, cfg_.page_size)
            : std::min(
                  round_down(
                      static_cast<uint64_t>(static_cast<double>(info_.disk_free_bytes) * cfg_.auto_nvme_fraction),
                      cfg_.page_size),
                  cfg_.auto_nvme_cap);
    if (nvme_limit == 0) {
        throw Error(ErrorCode::Budget, "effective NVMe budget is zero");
    }

    vram_budget_.set_limit(vram_limit, cfg_.vram_reserve_margin);
    host_budget_.set_limit(host_limit, 0.0);
    host_.set_budget(host_limit, 0.0);
    host_.set_device_backend(device_.get());  // pinned alloc via the device backend
    nvme_budget_.set_limit(nvme_limit, 0.0);

    planner_ = std::make_unique<Planner>(Planner::Budgets{});
    policy_ = make_policy(cfg_.policy);

    // NVMe store.
    std::string store_path = cfg_.nvme_path;
    const bool auto_store_path = cfg_.nvme_path.empty();
    if (auto_store_path) {
        std::error_code ec;
        store_path = (std::filesystem::temp_directory_path(ec) /
                      ("flashtier-store-" + std::to_string(process_id()) + ".bin"))
                         .string();
        if (ec) {
            throw Error(ErrorCode::Io, "cannot resolve temporary directory", ec.message());
        }
        // Auto-generated temp stores are ephemeral: a leftover from an
        // interrupted run (or a failed deletion) with an incompatible
        // geometry must not block this run. User-specified --nvme-path
        // stores keep strict header validation below.
        if (!cfg_.retain_store && std::filesystem::exists(store_path) &&
            !store_header_valid(store_path, cfg_.page_size, nvme_limit + cfg_.page_size)) {
            NvmeBackend::destroy_file(store_path);
        }
    }
    store_path_ = store_path;
    nvme_ = std::make_unique<NvmeBackend>();
    StorageBackend::Info store =
        nvme_->open(store_path_, nvme_limit + cfg_.page_size, cfg_.page_size);
    store_opened_ = true;

    // Telemetry.
    const std::string dir = cfg_.output_dir.empty() ? "." : cfg_.output_dir;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const std::string base =
        dir + "/flashtier-telemetry-" + std::to_string(::time(nullptr)) + "-" +
        std::to_string(process_id());
    sink_ = std::make_shared<TelemetrySink>(base + ".human.txt", base + ".jsonl", false);
    agg_ = std::make_unique<TelemetryAggregator>();

    TelemetryEvent cfg_ev;
    cfg_ev.type = EventType::ConfigEvent;
    cfg_ev.reason = cfg_.describe() + "\n  selected backend: " + backend_name_ +
                    "\n  selection reason: " + backend_reason_;
    emit_event(cfg_ev);

    // Workers.
    const unsigned n_workers =
        cfg_.worker_threads != 0 ? cfg_.worker_threads : cfg_.queue_depth;
    stopping_ = false;
    for (unsigned i = 0; i < n_workers; ++i) {
        workers_.emplace_back([this] { worker_loop(); });
    }
}

void Runtime::shutdown() {
    {
        std::lock_guard lock(queue_mu_);
        stopping_ = true;
    }
    queue_cv_.notify_all();
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
    workers_.clear();

    if (nvme_ && store_opened_) {
        nvme_->flush();
        nvme_->close();
        store_opened_ = false;
        if (!cfg_.retain_store) {
            NvmeBackend::destroy_file(store_path_);
        }
    }
    // Release the device backend last: host buffers allocated through the
    // backend's pinned allocator must be freed before the backend closes.
    if (device_) {
        if (device_stream_) {
            device_->destroy_stream(device_stream_);
            device_stream_ = nullptr;
        }
        if (device_event_a_) {
            device_->destroy_event(device_event_a_);
            device_event_a_ = nullptr;
        }
        if (device_event_b_) {
            device_->destroy_event(device_event_b_);
            device_event_b_ = nullptr;
        }
        device_->close();
        device_.reset();
        host_.set_device_backend(nullptr);
    }
    if (sink_) {
        sink_->flush();
    }
}

// ---------------------------------------------------------------------------
// Page management
// ---------------------------------------------------------------------------

PageId Runtime::allocate_page(uint64_t logical_bytes, SemanticClass cls,
                              bool pinned, int64_t execution_order_hint,
                              int64_t reuse_distance_estimate) {
    if (logical_bytes == 0) {
        throw Error(ErrorCode::InvalidArgument, "page size must be nonzero");
    }
    const uint64_t alloc_size =
        (logical_bytes + cfg_.page_size - 1) / cfg_.page_size * cfg_.page_size;

    PageId id;
    {
        std::lock_guard lock(state_mu_);
        id.value = next_page_id_++;
    }

    Tier tier;
    {
        std::lock_guard lock(state_mu_);
        planner_->set_budgets(Planner::Budgets{
            vram_budget_.limit(), vram_budget_.reserve_bytes(), vram_budget_.used(),
            host_.limit(), host_.used(), nvme_budget_.limit(), nvme_budget_.used()});
        tier = planner_->choose_allocation_tier(alloc_size, cfg_.use_pageable_fallback);
    }

    PageMetadata meta;
    meta.id = id;
    meta.logical_size = logical_bytes;
    meta.allocation_size = alloc_size;
    meta.semantic_class = cls;
    meta.pinned = pinned;
    meta.execution_order_hint = execution_order_hint;
    meta.reuse_distance_estimate = reuse_distance_estimate;

    switch (tier) {
        case Tier::Vram: {
            // Planned eviction before allocation: never a blind device
            // allocation after budget exhaustion.
            ensure_vram_headroom(alloc_size);
            bool reserved = false;
            {
                std::lock_guard lock(state_mu_);
                reserved = vram_budget_.try_reserve(alloc_size);
            }
            if (!reserved) {
                throw Error(ErrorCode::Budget,
                            "VRAM budget could not be reserved after planned eviction",
                            id.to_string());
            }
            void* ptr = nullptr;
            try {
                ptr = device_alloc(alloc_size);
            } catch (...) {
                vram_budget_.release(alloc_size);
                throw;
            }
            {
                std::lock_guard lock(state_mu_);
                meta.vram_ptr = ptr;
                table_.insert(meta);
                table_.with(id, [&](PageMetadata& m) { m.transition(PageState::ResidentVram); });
            }
            break;
        }
        case Tier::HostPinned:
        case Tier::HostPageable: {
            void* ptr = nullptr;
            {
                std::lock_guard lock(state_mu_);
                ptr = host_.allocate(alloc_size, cfg_.use_pageable_fallback);
            }
            if (ptr == nullptr) {
                throw Error(ErrorCode::Budget, "host budget could not be reserved",
                            id.to_string());
            }
            {
                std::lock_guard lock(state_mu_);
                meta.host_ptr = ptr;
                table_.insert(meta);
                table_.with(id, [&](PageMetadata& m) { m.transition(PageState::ResidentHost); });
            }
            break;
        }
        case Tier::Nvme: {
            uint64_t offset = 0;
            {
                std::lock_guard lock(state_mu_);
                if (!nvme_->allocate_extent(offset)) {
                    throw Error(ErrorCode::Budget, "NVMe store full", id.to_string());
                }
                if (!nvme_budget_.try_reserve(alloc_size)) {
                    nvme_->free_extent(offset);
                    throw Error(ErrorCode::Budget,
                                "NVMe budget could not be reserved", id.to_string());
                }
                meta.nvme_offset = offset;
                meta.has_nvme_copy = false;
                table_.insert(meta);
                table_.with(id, [&](PageMetadata& m) { m.transition(PageState::ResidentNvme); });
            }
            break;
        }
        default:
            throw Error(ErrorCode::Internal, "unexpected allocation tier");
    }

    TelemetryEvent ev;
    ev.type = EventType::PageAlloc;
    ev.page_id = id.value;
    ev.dst = tier;
    ev.bytes = alloc_size;
    ev.logical_size = static_cast<int64_t>(logical_bytes);
    ev.reason = "alloc";
    {
        std::lock_guard lock(state_mu_);
        ev.vram_used = vram_budget_.used();
        ev.host_used = host_.used();
        ev.nvme_used = nvme_budget_.used();
    }
    emit_event(ev);
    return id;
}

void Runtime::free_page(PageId id) {
    // Wait for any in-flight transfer so the page has settled before we
    // release its resources; a queued (not yet started) prefetch for the
    // page is cancelled below.
    PageMetadata m = wait_settled(id);
    {
        std::lock_guard lock(state_mu_);
        if (m.state == PageState::ResidentVram) {
            device_free(m.vram_ptr);
            vram_budget_.release(m.allocation_size);
        } else if (m.state == PageState::ResidentHost) {
            host_.free(m.host_ptr, m.allocation_size);
        } else if (m.state == PageState::ResidentNvme) {
            nvme_->free_extent(m.nvme_offset);
            nvme_budget_.release(m.allocation_size);
        }
        table_.with(id, [](PageMetadata& p) { p.transition(PageState::Released); });
        table_.erase(id);
        prefetched_pages_.erase(id.value);
    }
    remove_queued_prefetch(id);

    TelemetryEvent ev;
    ev.type = EventType::PageFree;
    ev.page_id = id.value;
    emit_event(ev);
}

// ---------------------------------------------------------------------------
// Access
// ---------------------------------------------------------------------------

void Runtime::write_page(PageId id, const void* data) {
    const auto t0 = Clock::now();
    PageMetadata m = wait_settled(id);
    if (m.state == PageState::ResidentNvme) {
        submit_demand(id, Tier::HostPinned);
        m = wait_settled(id);
    }

    const uint64_t bytes = m.allocation_size;
    const double stall_us =
        std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
    const bool resident_in_vram = m.state == PageState::ResidentVram;

    // The data copy runs under the state lock so a concurrent eviction
    // claim (which also needs the lock) can never free the buffer we are
    // copying from or into. If the page was evicted to NVMe between the
    // settle above and the lock, re-demand it and retry.
    const uint64_t checksum = fnv1a64(data, bytes);
    for (int attempt = 0; attempt < 8; ++attempt) {
        bool done = false;
        {
            std::lock_guard lock(state_mu_);
            const PageMetadata cur = table_.copy_of(id);
            if (cur.state == PageState::ResidentHost) {
                std::memcpy(cur.host_ptr, data, bytes);
                done = true;
            } else if (cur.state == PageState::ResidentVram) {
                std::vector<uint8_t> staging(bytes);
                std::memcpy(staging.data(), data, bytes);
                device_copy_h2d(cur.vram_ptr, staging.data(), bytes);
                done = true;
            } else {
                throw Error(ErrorCode::State,
                            "page not writable in current state",
                            id.to_string() + " " + page_state_name(cur.state));
            }
            if (done) {
                table_.with(id, [&](PageMetadata& p) {
                    p.dirty = true;
                    p.has_nvme_copy = false;
                    p.checksum = checksum;
                    p.last_access_sequence = ++access_seq_;
                    ++p.access_count;
                });
            }
        }
        if (done) break;
        m = wait_settled(id);
        if (m.state == PageState::ResidentNvme) {
            submit_demand(id, Tier::HostPinned);
            m = wait_settled(id);
        }
    }

    const bool prefetch_hit = consume_prefetch_mark(id);

    TelemetryEvent ev;
    ev.type = EventType::PageWrite;
    ev.page_id = id.value;
    ev.src = resident_in_vram ? Tier::Vram : Tier::HostPinned;
    ev.bytes = bytes;
    ev.cache_hit = resident_in_vram;
    ev.stall_us = stall_us;
    ev.prefetch_hit = prefetch_hit;
    ev.reason = resident_in_vram ? "staged_h2d" : "direct";
    emit_event(ev);
}

void Runtime::read_page(PageId id, void* out) {
    const auto t0 = Clock::now();
    PageMetadata m = wait_settled(id);
    bool demand_fault = false;
    if (m.state == PageState::ResidentNvme) {
        demand_fault = true;
        Tier target;
        {
            std::lock_guard lock(state_mu_);
            target = vram_budget_.headroom() >= m.allocation_size ? Tier::Vram
                                                                  : Tier::HostPinned;
        }
        submit_demand(id, target);
        m = wait_settled(id);
    }

    const uint64_t bytes = m.allocation_size;
    const double stall_us =
        std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
    const bool resident_in_vram = m.state == PageState::ResidentVram;

    // The data copy runs under the state lock so a concurrent eviction
    // claim can never free the buffer we copy from (see write_page). If the
    // page was evicted to NVMe between the settle and the lock, re-demand
    // it and retry.
    for (int attempt = 0; attempt < 8; ++attempt) {
        bool done = false;
        {
            std::lock_guard lock(state_mu_);
            const PageMetadata cur = table_.copy_of(id);
            if (cur.state == PageState::ResidentHost) {
                std::memcpy(out, cur.host_ptr, bytes);
                done = true;
            } else if (cur.state == PageState::ResidentVram) {
                std::vector<uint8_t> staging(bytes);
                device_copy_d2h(staging.data(), cur.vram_ptr, bytes);
                std::memcpy(out, staging.data(), bytes);
                done = true;
            }
            if (done) {
                table_.with(id, [&](PageMetadata& p) {
                    p.last_access_sequence = ++access_seq_;
                    ++p.access_count;
                });
            }
        }
        if (done) break;
        m = wait_settled(id);
        if (m.state == PageState::ResidentNvme) {
            submit_demand(id, Tier::HostPinned);
            m = wait_settled(id);
        }
    }

    const bool prefetch_hit = consume_prefetch_mark(id);

    TelemetryEvent ev;
    ev.type = EventType::PageRead;
    ev.page_id = id.value;
    ev.src = resident_in_vram ? Tier::Vram : Tier::HostPinned;
    ev.bytes = bytes;
    ev.cache_hit = resident_in_vram;
    ev.stall_us = demand_fault ? stall_us : 0.0;
    ev.prefetch_hit = prefetch_hit;
    ev.reason = demand_fault ? "demand" : "resident";
    emit_event(ev);

    if (demand_fault) {
        TelemetryEvent df;
        df.type = EventType::DemandFault;
        df.page_id = id.value;
        df.stall_us = stall_us;
        df.reason = "nvme";
        emit_event(df);

        TelemetryEvent st;
        st.type = EventType::Stall;
        st.page_id = id.value;
        st.stall_us = stall_us;
        st.reason = "demand_load";
        emit_event(st);
    }
}

// ---------------------------------------------------------------------------
// Residency control
// ---------------------------------------------------------------------------

void Runtime::promote(PageId id) {
    PageMetadata m;
    {
        std::lock_guard lock(state_mu_);
        m = table_.copy_of(id);
    }
    if (m.state == PageState::ResidentVram) return;
    // In CPU-only mode (or a zero VRAM budget) the best available tier is
    // the host tier; promoting to a nonexistent VRAM tier must degrade
    // gracefully instead of failing.
    Tier target;
    {
        std::lock_guard lock(state_mu_);
        target = vram_budget_.limit() != 0 ? Tier::Vram : Tier::HostPinned;
    }
    submit_demand(id, target);
}

void Runtime::demote_to_host(PageId id) {
    for (int attempt = 0; attempt < 16; ++attempt) {
        PageMetadata m = wait_settled(id);
        if (m.state == PageState::ResidentHost) return;
        if (m.state == PageState::ResidentVram) {
            evict_vram_to_host(id);  // claim-atomic; retry if busy
            m = wait_settled(id);
            if (m.state == PageState::ResidentHost) return;
            continue;
        }
        if (m.state == PageState::ResidentNvme) {
            load_nvme_to_host(id);  // claim-atomic; retry if busy
            m = wait_settled(id);
            if (m.state == PageState::ResidentHost) return;
            continue;
        }
        throw Error(ErrorCode::State, "cannot demote page in current state",
                    id.to_string());
    }
    throw Error(ErrorCode::State, "page did not settle during demotion", id.to_string());
}

void Runtime::demote_to_nvme(PageId id) {
    for (int attempt = 0; attempt < 16; ++attempt) {
        PageMetadata m = wait_settled(id);
        if (m.state == PageState::ResidentNvme) return;
        if (m.state == PageState::ResidentVram) {
            evict_vram_to_host(id);
            m = wait_settled(id);
            if (m.state == PageState::ResidentNvme) return;
            continue;
        }
        if (m.state == PageState::ResidentHost) {
            evict_host_to_nvme(id);
            m = wait_settled(id);
            if (m.state == PageState::ResidentNvme) return;
            continue;
        }
        throw Error(ErrorCode::State, "cannot demote page in current state",
                    id.to_string());
    }
    throw Error(ErrorCode::State, "page did not settle during demotion", id.to_string());
}

void Runtime::evict(PageId id) {
    PageMetadata m = wait_settled(id);
    if (m.state == PageState::ResidentNvme) return;
    if (m.state == PageState::ResidentVram && !m.dirty && m.has_nvme_copy) {
        // Clean page with a valid NVMe copy: drop without any transfer.
        // Re-checked under the lock so it cannot race another claim.
        bool dropped = false;
        {
            std::lock_guard lock(state_mu_);
            const PageMetadata cur = table_.copy_of(id);
            if (!cur.dirty && cur.has_nvme_copy && !cur.in_flight &&
                cur.state == PageState::ResidentVram) {
                device_free(cur.vram_ptr);
                vram_budget_.release(cur.allocation_size);
                table_.with(id, [&](PageMetadata& p) { p.transition(PageState::ResidentNvme); });
                prefetched_pages_.erase(id.value);
                dropped = true;
            }
        }
        if (dropped) {
            TelemetryEvent ev;
            ev.type = EventType::Evict;
            ev.page_id = id.value;
            ev.src = Tier::Vram;
            ev.dst = Tier::Nvme;
            ev.bytes = m.allocation_size;
            ev.reason = "drop_clean";
            emit_event(ev);
            return;
        }
    }
    demote_to_host(id);
}

void Runtime::pin(PageId id, bool pinned) {
    std::lock_guard lock(state_mu_);
    table_.with(id, [&](PageMetadata& p) { p.pinned = pinned; });
}

// ---------------------------------------------------------------------------
// Prefetch
// ---------------------------------------------------------------------------

void Runtime::prefetch(const std::vector<PageId>& ids) {
    for (PageId id : ids) {
        // Wait for any in-flight transfer so the residency decision below
        // is authoritative; a page already being loaded needs no new
        // prefetch (the in-flight load serves the access).
        PageMetadata m = wait_settled(id);
        if (m.state == PageState::ResidentVram || m.state == PageState::ResidentHost) {
            mark_prefetched(id);
            TelemetryEvent ev;
            ev.type = EventType::PrefetchIssue;
            ev.page_id = id.value;
            ev.reason = "already_resident";
            emit_event(ev);
            continue;
        }
        Tier target = Tier::HostPinned;
        {
            std::lock_guard lock(state_mu_);
            if (vram_budget_.headroom() >= m.allocation_size) target = Tier::Vram;
        }
        submit_prefetch(id, target);
    }
}

void Runtime::prefetch_sequential(PageId start, uint64_t count) {
    std::vector<PageId> ids;
    ids.reserve(count);
    for (uint64_t i = 0; i < count; ++i) {
        ids.push_back(PageId{start.value + i});
    }
    prefetch(ids);
}

// ---------------------------------------------------------------------------
// Integrity
// ---------------------------------------------------------------------------

void Runtime::fill_page(PageId id) {
    PageMetadata m;
    {
        std::lock_guard lock(state_mu_);
        m = table_.copy_of(id);
    }
    std::vector<uint8_t> buf(m.allocation_size);
    fill_pattern(buf.data(), buf.size(), cfg_.seed, id.value);
    write_page(id, buf.data());
}

void Runtime::verify_page(PageId id) {
    PageMetadata m = wait_settled(id);
    if (m.state == PageState::ResidentNvme) {
        submit_demand(id, Tier::HostPinned);
        m = wait_settled(id);
    }

    // The content copy runs under the state lock so a concurrent eviction
    // claim can never free the buffer we copy from (see write_page). If the
    // page was evicted to NVMe between the settle and the lock, re-demand
    // it and retry.
    std::vector<uint8_t> buf(m.allocation_size);
    for (int attempt = 0; attempt < 8; ++attempt) {
        bool done = false;
        {
            std::lock_guard lock(state_mu_);
            const PageMetadata cur = table_.copy_of(id);
            if (cur.state == PageState::ResidentHost) {
                std::memcpy(buf.data(), cur.host_ptr, buf.size());
                done = true;
            } else if (cur.state == PageState::ResidentVram) {
                std::vector<uint8_t> staging(buf.size());
                device_copy_d2h(staging.data(), cur.vram_ptr, buf.size());
                std::memcpy(buf.data(), staging.data(), buf.size());
                done = true;
            }
            if (done) {
                m = cur;  // keep the settled snapshot for telemetry below
            }
        }
        if (done) break;
        m = wait_settled(id);
        if (m.state == PageState::ResidentNvme) {
            submit_demand(id, Tier::HostPinned);
            m = wait_settled(id);
        }
    }

    const uint64_t actual_checksum = fnv1a64(buf.data(), buf.size());
    const bool checksum_ok = (actual_checksum == m.checksum);
    const auto mismatch = verify_pattern(buf.data(), buf.size(), cfg_.seed, id.value);
    const bool content_ok = !mismatch.has_value();

    TelemetryEvent ev;
    ev.type = EventType::IntegrityCheck;
    ev.page_id = id.value;
    ev.checksum_ok = checksum_ok && content_ok;
    ev.checksum_expected = hex_u64(m.checksum);
    ev.checksum_actual = hex_u64(actual_checksum);
    ev.bytes = buf.size();
    ev.reason = "verify";
    if (mismatch.has_value()) {
        ev.reason = "content_mismatch@" + std::to_string(mismatch->offset);
        ev.error_code = "integrity";
    }
    emit_event(ev);

    if (!checksum_ok || !content_ok) {
        throw Error(ErrorCode::Integrity,
                    "page integrity check failed",
                    "page " + id.to_string() +
                        " expected_checksum=" + hex_u64(m.checksum) +
                        " actual_checksum=" + hex_u64(actual_checksum) +
                        (mismatch.has_value()
                             ? " mismatch_at=" + std::to_string(mismatch->offset)
                             : "") +
                        " state=" + page_state_name(m.state) +
                        " path=" + transfer_path_str(m.current_tier, m.current_tier));
    }
}

uint64_t Runtime::page_checksum(PageId id) const {
    std::lock_guard lock(state_mu_);
    return table_.copy_of(id).checksum;
}

// ---------------------------------------------------------------------------
// Introspection
// ---------------------------------------------------------------------------

PageMetadata Runtime::metadata(PageId id) const {
    std::lock_guard lock(state_mu_);
    return table_.copy_of(id);
}

TelemetryAggregates Runtime::aggregates() const {
    return agg_ ? agg_->summary() : TelemetryAggregates{};
}

uint64_t Runtime::vram_used() const { return vram_budget_.used(); }
uint64_t Runtime::host_used() const { return host_.used(); }
uint64_t Runtime::nvme_used() const { return nvme_budget_.used(); }

uint64_t Runtime::pages_resident_vram() const {
    std::lock_guard lock(state_mu_);
    uint64_t n = 0;
    table_.for_each([&](const PageMetadata& p) {
        if (p.state == PageState::ResidentVram) ++n;
    });
    return n;
}

uint64_t Runtime::pages_resident_host() const {
    std::lock_guard lock(state_mu_);
    uint64_t n = 0;
    table_.for_each([&](const PageMetadata& p) {
        if (p.state == PageState::ResidentHost) ++n;
    });
    return n;
}

uint64_t Runtime::pages_resident_nvme() const {
    std::lock_guard lock(state_mu_);
    uint64_t n = 0;
    table_.for_each([&](const PageMetadata& p) {
        if (p.state == PageState::ResidentNvme) ++n;
    });
    return n;
}

void Runtime::emit_telemetry(TelemetryEvent ev) {
    emit_event(ev);
}

void Runtime::emit_event(TelemetryEvent ev) {
    if (sink_) {
        sink_->emit(ev);
    }
    if (agg_) {
        agg_->record(ev);
    }
}

void Runtime::flush_telemetry() {
    if (sink_) sink_->flush();
}

void* Runtime::device_memory_handle(PageId id) {
    std::lock_guard lock(state_mu_);
    PageMetadata m = table_.copy_of(id);
    return m.state == PageState::ResidentVram ? m.vram_ptr : nullptr;
}

// ---------------------------------------------------------------------------
// Transfer engine
// ---------------------------------------------------------------------------

void Runtime::worker_loop() {
    while (true) {
        Request req;
        {
            std::unique_lock lock(queue_mu_);
            queue_cv_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
            if (stopping_ && queue_.empty()) return;
            req = std::move(queue_.front());
            queue_.pop_front();
        }

        std::shared_ptr<std::promise<void>> promise =
            std::static_pointer_cast<std::promise<void>>(req.completion);
        try {
            load_page(req.page, req.target);
            if (promise) promise->set_value();
        } catch (const Error& e) {
            record_error(e);
            if (promise) promise->set_exception(std::current_exception());
        } catch (const std::exception& e) {
            Error err(ErrorCode::Internal,
                      "worker exception: " + std::string(e.what()));
            record_error(err);
            if (promise) promise->set_exception(std::current_exception());
        }
    }
}

void Runtime::submit_demand(PageId id, Tier target) {
    auto promise = std::make_shared<std::promise<void>>();
    {
        std::unique_lock lock(queue_mu_);
        queue_cv_.wait(lock, [this] { return queue_.size() < cfg_.queue_depth + 1; });
        queue_.push_back(Request{Request::Kind::Demand, id, target, promise, "demand"});
    }
    queue_cv_.notify_one();
    promise->get_future().get();  // rethrows worker errors
}

void Runtime::submit_prefetch(PageId id, Tier target) {
    bool queued = false;
    {
        std::unique_lock lock(queue_mu_);
        for (const auto& q : queue_) {
            if (q.page == id && q.kind == Request::Kind::Prefetch) {
                queued = true;
                break;
            }
        }
        if (!queued && queue_.size() < cfg_.queue_depth + cfg_.prefetch_depth) {
            queue_.push_back(
                Request{Request::Kind::Prefetch, id, target, nullptr, "prefetch"});
            queued = true;
        }
    }
    queue_cv_.notify_one();
    if (!queued) {
        TelemetryEvent ev;
        ev.type = EventType::PrefetchWaste;
        ev.page_id = id.value;
        ev.reason = "queue_full";
        emit_event(ev);
        return;
    }
    TelemetryEvent ev;
    ev.type = EventType::PrefetchIssue;
    ev.page_id = id.value;
    ev.reason = "queued";
    emit_event(ev);
}

void Runtime::remove_queued_prefetch(PageId id) {
    std::lock_guard lock(queue_mu_);
    for (auto it = queue_.begin(); it != queue_.end();) {
        if (it->page == id && it->kind == Request::Kind::Prefetch) {
            it = queue_.erase(it);
        } else {
            ++it;
        }
    }
}

void Runtime::load_page(PageId id, Tier target) {
    // Settle-and-retry loop: transfer steps are claim-atomic (they no-op
    // when another worker owns the page), so a busy page is waited for and
    // re-evaluated instead of racing.
    for (int attempt = 0; attempt < 16; ++attempt) {
        PageMetadata m;
        {
            std::lock_guard lock(state_mu_);
            m = table_.copy_of(id);
        }
        switch (m.state) {
            case PageState::ResidentVram:
                return;
            case PageState::ResidentHost: {
                if (target == Tier::Vram) {
                    bool has_vram = false;
                    {
                        std::lock_guard lock(state_mu_);
                        has_vram = vram_budget_.limit() != 0;
                    }
                    if (!has_vram) return;  // CPU-only: host is the top tier
                    // Planned eviction before allocation: promoting into a
                    // full device tier must free room first, never fail
                    // with a raw device OOM or a spurious budget error.
                    ensure_vram_headroom(m.allocation_size);
                    promote_host_to_vram(id);
                    {
                        std::lock_guard lock(state_mu_);
                        m = table_.copy_of(id);
                    }
                    if (m.state == PageState::ResidentVram) return;
                    wait_settled(id);  // busy or failed; settle then retry
                    continue;
                }
                return;
            }
            case PageState::ResidentNvme: {
                    load_nvme_to_host(id);
                {
                    std::lock_guard lock(state_mu_);
                    m = table_.copy_of(id);
                }
                if (m.state != PageState::ResidentHost) {
                    if (m.state == PageState::ResidentVram) return;
                    wait_settled(id);  // busy or failed; settle then retry
                    continue;
                }
                if (target == Tier::Vram) {
                    bool has_vram = false;
                    {
                        std::lock_guard lock(state_mu_);
                        has_vram = vram_budget_.limit() != 0;
                    }
                    if (has_vram) {
                        // Planned eviction before promotion (see above).
                        ensure_vram_headroom(m.allocation_size);
                        promote_host_to_vram(id);
                        {
                            std::lock_guard lock(state_mu_);
                            m = table_.copy_of(id);
                        }
                        if (m.state == PageState::ResidentVram) return;
                        wait_settled(id);
                        continue;
                    }
                }
                return;
            }
            default:
                // Transfer states: wait for settlement; Error/Released
                // states throw via wait_settled.
                wait_settled(id);
                break;
        }
    }
    throw Error(ErrorCode::State, "page did not settle during load", id.to_string());
}

// Claim a host buffer of `bytes` for `for_page`. Serialized on
// headroom_mu_ so the check-then-allocate cannot race another worker.
// When the budget has no room, legal host-resident victims (never the
// requesting page, never pinned, never busy) are evicted to NVMe; if every
// candidate is busy the lock is released and we wait for settlement before
// retrying, which is safe because workers settle pages without needing
// headroom_mu_ themselves.
void* Runtime::claim_host_buffer(uint64_t bytes, PageId for_page) {
    for (int round = 0; round < 64; ++round) {
        {
            std::lock_guard hlock(headroom_mu_);
            void* ptr = nullptr;
            {
                std::lock_guard lock(state_mu_);
                ptr = host_.allocate(bytes, false);
            }
            if (ptr != nullptr) return ptr;

            bool evicted = false;
            {
                std::vector<PageMetadata> candidates;
                {
                    std::lock_guard lock(state_mu_);
                    table_.for_each([&](const PageMetadata& p) {
                        if (p.id != for_page && p.state == PageState::ResidentHost &&
                            !p.pinned && !p.in_flight) {
                            candidates.push_back(p);
                        }
                    });
                }
                const std::vector<PageId> ordered = policy_->rank_victims(candidates);
                for (PageId victim : ordered) {
                    evict_host_to_nvme(victim);  // claim-atomic; no-op if busy
                    PageState st;
                    {
                        std::lock_guard lock(state_mu_);
                        st = table_.copy_of(victim).state;
                    }
                    if (st == PageState::ResidentNvme) {
                        evicted = true;
                        break;
                    }
                }
            }
            if (evicted) continue;  // retry allocation with freed space
            if (host_.headroom() >= bytes) continue;  // a busy page settled
        }
        // All candidates busy: release the headroom lock and wait for the
        // workers holding those pages to settle them.
        std::unique_lock lock(state_mu_);
        state_cv_.wait_for(lock, std::chrono::milliseconds(25));
    }
    throw Error(ErrorCode::Budget,
                "host budget could not be satisfied", for_page.to_string());
}

void Runtime::ensure_vram_headroom(uint64_t need_bytes) {
    while (true) {
        {
            std::lock_guard lock(state_mu_);
            if (vram_budget_.headroom() >= need_bytes) return;
        }

        std::vector<PageMetadata> candidates;
        uint64_t host_free = 0;
        {
            std::lock_guard lock(state_mu_);
            host_free = host_.headroom();
            table_.for_each([&](const PageMetadata& p) {
                if (p.state == PageState::ResidentVram && !p.pinned && !p.in_flight) {
                    candidates.push_back(p);
                }
            });
        }

        Planner::Budgets budgets{
            vram_budget_.limit(), vram_budget_.reserve_bytes(), vram_budget_.used(),
            host_.limit(), host_.used(), nvme_budget_.limit(), nvme_budget_.used()};
        const uint64_t shortfall =
            need_bytes > vram_budget_.headroom() ? need_bytes - vram_budget_.headroom() : 0;
        std::vector<Planner::EvictionDecision> decisions;
        {
            std::lock_guard lock(state_mu_);
            planner_->set_budgets(budgets);
        }
        decisions = planner_->plan_evictions(shortfall, candidates, *policy_, host_free);

        bool made_progress = false;
        for (const auto& d : decisions) {
            // Clean page with a valid NVMe copy: drop with no transfer.
            // The state check happens under the lock so the drop cannot
            // race another worker's claim on the same page.
            uint64_t drop_bytes = 0;
            bool dropped = false;
            {
                std::lock_guard lock(state_mu_);
                const PageMetadata cur = table_.copy_of(d.id);
                if (cur.pinned || cur.in_flight) continue;
                if (d.target == Tier::Nvme && !cur.dirty && cur.has_nvme_copy &&
                    cur.state == PageState::ResidentVram) {
                    device_free(cur.vram_ptr);
                    vram_budget_.release(cur.allocation_size);
                    table_.with(d.id, [&](PageMetadata& p) {
                        p.transition(PageState::ResidentNvme);
                    });
                    prefetched_pages_.erase(d.id.value);
                    drop_bytes = cur.allocation_size;
                    dropped = true;
                }
            }
            if (dropped) {
                TelemetryEvent ev;
                ev.type = EventType::Evict;
                ev.page_id = d.id.value;
                ev.src = Tier::Vram;
                ev.dst = Tier::Nvme;
                ev.bytes = drop_bytes;
                ev.reason = "drop_clean";
                emit_event(ev);
                made_progress = true;
                continue;
            }
            // Demotion path: claim-atomic steps; busy victims are skipped.
            const PageMetadata cur = table_.copy_of(d.id);
            if (cur.pinned || cur.in_flight || cur.state != PageState::ResidentVram) {
                continue;
            }
            evict_vram_to_host(d.id);
            PageState st;
            {
                std::lock_guard lock(state_mu_);
                st = table_.copy_of(d.id).state;
            }
            if (st != PageState::ResidentHost) {
                continue;  // no-op (busy); another worker owns this page
            }
            if (d.target == Tier::Nvme) {
                evict_host_to_nvme(d.id);
            }
            made_progress = true;
        }
        if (!made_progress) {
            // Every candidate was busy; wait for workers to settle and
            // retry instead of spinning or failing spuriously.
            std::unique_lock lock(state_mu_);
            state_cv_.wait_for(lock, std::chrono::milliseconds(100));
        }
    }
}

void Runtime::evict_vram_to_host(PageId id) {
    // Claim-atomic: mark in_flight and transition under one lock so two
    // workers can never evict the same page. No-op when the page is busy
    // or no longer VRAM-resident; callers re-decide. The host staging
    // buffer is claimed first (serialized with other host consumers).
    PageMetadata m;
    void* host_ptr = claim_host_buffer(table_.copy_of(id).allocation_size, id);
    {
        std::lock_guard lock(state_mu_);
        m = table_.copy_of(id);
        if (m.state != PageState::ResidentVram || m.in_flight) {
            host_.free(host_ptr, m.allocation_size);
            return;
        }
        table_.with(id, [&](PageMetadata& p) {
            p.host_ptr = host_ptr;
            p.in_flight = true;
            p.transition(PageState::EvictingToHost);
        });
    }

    const auto t0 = Clock::now();
    const Tier src = Tier::Vram;
    const Tier dst = Tier::HostPinned;
    const uint64_t bytes = m.allocation_size;
    const std::string path = transfer_path_str(src, dst);
    try {
        device_copy_d2h(host_ptr, m.vram_ptr, bytes);
    } catch (...) {
        std::lock_guard lock(state_mu_);
        host_.free(host_ptr, bytes);
        table_.with(id, [&](PageMetadata& p) {
            p.host_ptr = nullptr;
            p.in_flight = false;
            p.transition(PageState::Error);
        });
        state_cv_.notify_all();  // wake callers waiting on this page
        throw;
    }
    const double us =
        std::chrono::duration<double, std::micro>(Clock::now() - t0).count();

    {
        std::lock_guard lock(state_mu_);
        device_free(m.vram_ptr);
        vram_budget_.release(bytes);
        table_.with(id, [&](PageMetadata& p) {
            p.vram_ptr = nullptr;
            p.in_flight = false;
            ++p.demotion_count;
            p.transition(PageState::ResidentHost);
        });
    }
    state_cv_.notify_all();
    record_transfer(EventType::TransferEnd, id, src, dst, bytes, us, "demote", path);
    TelemetryEvent ev;
    ev.type = EventType::Evict;
    ev.page_id = id.value;
    ev.src = src;
    ev.dst = dst;
    ev.bytes = bytes;
    ev.reason = "demote_to_host";
    emit_event(ev);
}

void Runtime::evict_host_to_nvme(PageId id) {
    // Claim-atomic (see evict_vram_to_host). Extent allocation happens
    // inside the claim so a busy page never gets an extent allocated for
    // it by two workers. No-op when busy or not host-resident.
    PageMetadata m;
    uint64_t offset = 0;
    bool need_write = false;
    {
        std::lock_guard lock(state_mu_);
        m = table_.copy_of(id);
        if (m.state != PageState::ResidentHost || m.in_flight) return;

        need_write = m.dirty || !m.has_nvme_copy;
        offset = m.nvme_offset;
        if (need_write && offset == PageMetadata::kInvalidOffset) {
            if (!nvme_->allocate_extent(offset)) {
                throw Error(ErrorCode::Budget, "NVMe store full during writeback",
                            id.to_string());
            }
            nvme_budget_.try_reserve(m.allocation_size);
        }
        table_.with(id, [&](PageMetadata& p) {
            p.in_flight = true;
            p.nvme_offset = offset;
            p.transition(PageState::EvictingToNvme);
        });
    }

    const auto t0 = Clock::now();
    const uint64_t bytes = m.allocation_size;
    const std::string path = "host_pinned->nvme";
    try {
        if (need_write) {
            AsyncOpPtr op = nvme_->write_async(offset, m.host_ptr, bytes);
            op->wait();
        }
    } catch (...) {
        std::lock_guard lock(state_mu_);
        table_.with(id, [&](PageMetadata& p) {
            p.in_flight = false;
            p.transition(PageState::Error);
        });
        state_cv_.notify_all();  // wake callers waiting on this page
        throw;
    }
    const double us =
        std::chrono::duration<double, std::micro>(Clock::now() - t0).count();

    {
        std::lock_guard lock(state_mu_);
        host_.free(m.host_ptr, bytes);
        table_.with(id, [&](PageMetadata& p) {
            p.host_ptr = nullptr;
            p.in_flight = false;
            p.dirty = false;
            p.has_nvme_copy = true;
            ++p.demotion_count;
            p.transition(PageState::ResidentNvme);
        });
    }
    state_cv_.notify_all();
    record_transfer(EventType::TransferEnd, id, Tier::HostPinned, Tier::Nvme, bytes, us,
                    need_write ? "writeback" : "skip_write", path);
    TelemetryEvent ev;
    ev.type = EventType::Evict;
    ev.page_id = id.value;
    ev.src = Tier::HostPinned;
    ev.dst = Tier::Nvme;
    ev.bytes = bytes;
    ev.reason = need_write ? "writeback" : "drop_clean";
    emit_event(ev);
}

void Runtime::load_nvme_to_host(PageId id) {
    // Claim-atomic (see evict_vram_to_host). No-op when busy.
    PageMetadata m;
    void* host_ptr = claim_host_buffer(table_.copy_of(id).allocation_size, id);
    {
        std::lock_guard lock(state_mu_);
        m = table_.copy_of(id);
        if (m.state != PageState::ResidentNvme || m.in_flight) {
            host_.free(host_ptr, m.allocation_size);
            return;
        }
        table_.with(id, [&](PageMetadata& p) {
            p.host_ptr = host_ptr;
            p.in_flight = true;
            p.transition(PageState::LoadingToHost);
        });
    }

    const auto t0 = Clock::now();
    const uint64_t bytes = m.allocation_size;
    const std::string path = "nvme->host_pinned";
    try {
        AsyncOpPtr op = nvme_->read_async(m.nvme_offset, host_ptr, bytes);
        op->wait();
    } catch (...) {
        std::lock_guard lock(state_mu_);
        host_.free(host_ptr, bytes);
        table_.with(id, [&](PageMetadata& p) {
            p.host_ptr = nullptr;
            p.in_flight = false;
            p.transition(PageState::Error);
        });
        state_cv_.notify_all();  // wake callers waiting on this page
        throw;
    }
    const double us =
        std::chrono::duration<double, std::micro>(Clock::now() - t0).count();

    {
        std::lock_guard lock(state_mu_);
        table_.with(id, [&](PageMetadata& p) {
            p.in_flight = false;
            ++p.promotion_count;
            p.transition(PageState::ResidentHost);
        });
    }
    state_cv_.notify_all();
    record_transfer(EventType::TransferEnd, id, Tier::Nvme, Tier::HostPinned, bytes, us,
                    "promote", path);
}

void Runtime::promote_host_to_vram(PageId id) {
    // Claim-atomic (see evict_vram_to_host). VRAM budget is reserved and
    // the device allocation happens inside the claim; on failure the
    // reservation is released and the page stays host-resident (Error
    // state on transfer failure only).
    PageMetadata m;
    {
        std::lock_guard lock(state_mu_);
        m = table_.copy_of(id);
        if (m.state != PageState::ResidentHost || m.in_flight) return;
    }
    if (!vram_tier_exists()) {
        return;  // no device tier (CPU-only): host residency is the top tier
    }
    void* vram_ptr = nullptr;
    {
        std::lock_guard lock(state_mu_);
        if (!vram_budget_.try_reserve(m.allocation_size)) {
            // No headroom at reserve time (another worker consumed it).
            // This is a no-op: load_page's retry loop re-runs
            // ensure_vram_headroom and retries the promotion.
            return;
        }
        try {
            vram_ptr = device_alloc(m.allocation_size);
        } catch (...) {
            vram_budget_.release(m.allocation_size);
            throw;
        }
        table_.with(id, [&](PageMetadata& p) {
            p.vram_ptr = vram_ptr;
            p.in_flight = true;
            p.transition(PageState::LoadingToVram);
        });
    }

    const auto t0 = Clock::now();
    const uint64_t bytes = m.allocation_size;
    const std::string path = transfer_path_str(Tier::HostPinned, Tier::Vram);
    try {
        device_copy_h2d(vram_ptr, m.host_ptr, bytes);
    } catch (...) {
        std::lock_guard lock(state_mu_);
        vram_budget_.release(bytes);
        device_free(vram_ptr);
        table_.with(id, [&](PageMetadata& p) {
            p.vram_ptr = nullptr;
            p.in_flight = false;
            p.transition(PageState::Error);
        });
        state_cv_.notify_all();  // wake callers waiting on this page
        throw;
    }
    const double us =
        std::chrono::duration<double, std::micro>(Clock::now() - t0).count();

    {
        std::lock_guard lock(state_mu_);
        host_.free(m.host_ptr, bytes);
        table_.with(id, [&](PageMetadata& p) {
            p.host_ptr = nullptr;
            p.in_flight = false;
            ++p.promotion_count;
            p.transition(PageState::ResidentVram);
        });
    }
    state_cv_.notify_all();
    record_transfer(EventType::TransferEnd, id, Tier::HostPinned, Tier::Vram, bytes, us,
                    "promote", path);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

void Runtime::mark_prefetched(PageId id) {
    std::lock_guard lock(state_mu_);
    prefetched_pages_.insert(id.value);
}

bool Runtime::consume_prefetch_mark(PageId id) {
    bool hit = false;
    {
        std::lock_guard lock(state_mu_);
        hit = prefetched_pages_.erase(id.value) != 0;
    }
    if (hit) {
        TelemetryEvent ev;
        ev.type = EventType::PrefetchHit;
        ev.page_id = id.value;
        ev.prefetch_hit = true;
        ev.reason = "accessed";
        emit_event(ev);
    }
    return hit;
}

void Runtime::record_transfer(EventType type, PageId id, Tier src, Tier dst,
                              uint64_t bytes, double us, const std::string& reason,
                              const std::string& path) {
    TelemetryEvent ev;
    ev.type = type;
    ev.page_id = id.value;
    ev.src = src;
    ev.dst = dst;
    ev.bytes = bytes;
    ev.duration_us = us;
    ev.bandwidth_gb_s =
        us > 0.0 ? static_cast<double>(bytes) / 1e9 / (us / 1e6) : 0.0;
    ev.reason = reason;
    ev.path = path;
    ev.policy = policy_kind_name(cfg_.policy);
    {
        std::lock_guard lock(state_mu_);
        ev.queue_depth = static_cast<uint32_t>(queue_.size());
        ev.vram_used = vram_budget_.used();
        ev.vram_free = vram_budget_.headroom();
        ev.host_used = host_.used();
        ev.host_free = host_.headroom();
        ev.nvme_used = nvme_budget_.used();
        ev.nvme_free = nvme_budget_.headroom();
    }
    emit_event(ev);
}

void Runtime::record_error(const Error& e) {
    TelemetryEvent ev;
    ev.type = EventType::ErrorEvent;
    ev.error_code = error_code_name(e.code());
    ev.reason = e.what();
    if (!e.detail().empty()) {
        ev.cuda_error = e.detail();
    }
    emit_event(ev);
}

// Wait until the page is in a settled (non-transfer) state, then return a
// snapshot. Concurrent transfers on the page complete first; the page's
// resources are only touched after it has settled, which keeps the
// single-authoritative-copy invariant.
PageMetadata Runtime::wait_settled(PageId id) {
    std::unique_lock lock(state_mu_);
    while (true) {
        PageMetadata m = table_.copy_of(id);
        switch (m.state) {
            case PageState::ResidentVram:
            case PageState::ResidentHost:
            case PageState::ResidentNvme:
                return m;
            case PageState::Error:
                throw Error(ErrorCode::State,
                            "page is in failed state", id.to_string());
            case PageState::Released:
            case PageState::Unallocated:
                throw Error(ErrorCode::State,
                            "page is not allocated", id.to_string());
            default:
                break;  // in-flight transfer; wait for it to settle
        }
        state_cv_.wait(lock);
    }
}

std::string Runtime::transfer_path_str(Tier src, Tier dst) const {
    // The device tier is reported with the actual backend name so
    // telemetry paths record the real transfer path, e.g.
    // "cuda_device->host_pinned", "nvme->host_pinned->cuda_device".
    auto tier_label = [this](Tier t) -> std::string {
        if (t == Tier::Vram) {
            return backend_name_.empty() ? std::string("vram")
                                         : backend_name_ + "_device";
        }
        return tier_name(t);
    };
    return tier_label(src) + "->" + tier_label(dst);
}

bool Runtime::vram_tier_exists() const {
    std::lock_guard lock(state_mu_);
    return vram_budget_.limit() != 0 && device_ != nullptr;
}

void* Runtime::device_alloc(std::size_t bytes) {
    if (device_ == nullptr) {
        throw Error(ErrorCode::State,
                    "device-memory allocation requested with no device backend");
    }
    return device_->allocate(bytes);
}

void Runtime::device_free(void* ptr) {
    if (ptr == nullptr) return;
    if (device_ == nullptr) {
        throw Error(ErrorCode::State,
                    "device-memory release requested with no device backend");
    }
    device_->free(ptr);
}

void Runtime::device_copy_h2d(void* dst, const void* src, std::size_t bytes) {
    if (device_ == nullptr) {
        throw Error(ErrorCode::State,
                    "device transfer requested with no device backend");
    }
    device_->async_copy_host_to_device(dst, src, bytes, device_stream_);
    device_->sync_stream(device_stream_);
}

void Runtime::device_copy_d2h(void* dst, const void* src, std::size_t bytes) {
    if (device_ == nullptr) {
        throw Error(ErrorCode::State,
                    "device transfer requested with no device backend");
    }
    device_->async_copy_device_to_host(dst, src, bytes, device_stream_);
    device_->sync_stream(device_stream_);
}

void Runtime::device_sync() {
    if (device_ != nullptr) {
        device_->sync_all();
    }
}

std::string Runtime::selected_backend() const {
    return backend_name_.empty() ? "auto" : backend_name_;
}

}  // namespace flashtier
