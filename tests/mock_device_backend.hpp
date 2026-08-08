#pragma once

// Test-only mock device backend: in-memory "device memory" with failure
// injection. Used by the generic backend conformance suite so the battery
// runs on every platform without hardware. A mock passing never counts as
// validating a vendor backend.

#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "flashtier/backends/device_backend.hpp"
#include "flashtier/error.hpp"

namespace flashtier {

class MockDeviceBackend final : public DeviceBackend {
public:
    struct Config {
        uint64_t device_memory = 64ull * 1024 * 1024;  // 64 MiB fake VRAM
        uint64_t max_allocation = 16ull * 1024 * 1024; // 16 MiB per alloc
        uint64_t alignment = 64;
        bool fail_allocate = false;   // inject allocation failures
        bool fail_open = false;       // inject open failure
        bool fail_transfer = false;   // inject transfer failures
        int device_count = 2;         // mock "multi-device" backend
    };

    explicit MockDeviceBackend(Config config = Config{}) : cfg_(std::move(config)) {}

    std::string backend_name() const override { return "mock"; }
    BackendVendor vendor() const override { return BackendVendor::Virtual; }
    std::string vendor_display_name() const override { return "Virtual (test mock)"; }

    std::vector<DeviceInfo> enumerate_devices() const override {
        std::vector<DeviceInfo> out;
        for (int i = 0; i < cfg_.device_count; ++i) {
            DeviceInfo info;
            info.id = "mock:" + std::to_string(i);
            info.name = "Mock device " + std::to_string(i);
            info.architecture = "mock-1.0";
            info.total_memory = cfg_.device_memory;
            info.free_memory = cfg_.device_memory;
            info.discrete = i == 0;  // device 0 mimics a discrete GPU
            info.memory_shared = i != 0;
            info.index = i;
            out.push_back(std::move(info));
        }
        return out;
    }

    DeviceCapabilities probe_capabilities() const override {
        DeviceCapabilities caps;
        caps.backend_available = true;
        caps.device_available = true;
        caps.explicit_allocation = true;
        caps.async_host_to_device = true;
        caps.async_device_to_host = true;
        caps.device_to_device = true;
        caps.pinned_host_allocation = true;
        caps.unified_memory = true;
        caps.host_coherent_memory = true;
        caps.memory_prefetch = true;  // no-op: heap memory is host-coherent
        caps.memory_advice = true;    // no-op: heap memory is host-coherent
        caps.event_timing = true;
        caps.max_allocation_size = cfg_.max_allocation;
        caps.alignment_bytes = cfg_.alignment;
        caps.transfer_granularity = 1;
        caps.queue_count = 4;
        caps.multi_device = cfg_.device_count > 1;
        caps.note = "test mock; failure injection available";
        return caps;
    }

    void open(int device_index) override {
        if (cfg_.fail_open) {
            throw Error(ErrorCode::Device, "mock: injected open failure");
        }
        const std::vector<DeviceInfo> devices = enumerate_devices();
        if (device_index < 0 || device_index >= static_cast<int>(devices.size())) {
            throw Error(ErrorCode::Config,
                        "mock: requested device index out of range");
        }
        std::lock_guard lock(mu_);
        info_ = devices[static_cast<std::size_t>(device_index)];
        opened_ = true;
    }

    void close() override {
        std::lock_guard lock(mu_);
        opened_ = false;
    }

    bool is_open() const override { return opened_; }

    DeviceInfo device_info() const override {
        if (!opened_) throw Error(ErrorCode::State, "mock backend is not open");
        return info_;
    }

    DeviceCapabilities capabilities() const override {
        if (!opened_) throw Error(ErrorCode::State, "mock backend is not open");
        return probe_capabilities();
    }

    uint64_t total_memory() const override {
        return opened_ ? info_.total_memory : 0;
    }

    uint64_t free_memory() const override {
        std::lock_guard lock(mu_);
        return opened_ ? info_.total_memory - used_bytes_ : 0;
    }

    void* allocate(std::size_t bytes) override {
        if (bytes == 0) {
            throw Error(ErrorCode::InvalidArgument, "mock: zero-byte allocation");
        }
        if (bytes > cfg_.max_allocation) {
            throw Error(ErrorCode::Budget,
                        "mock: allocation exceeds mock device limit");
        }
        if (cfg_.fail_allocate) {
            throw Error(ErrorCode::Budget, "mock: injected allocation failure");
        }
        std::lock_guard lock(mu_);
        if (used_bytes_ + bytes > cfg_.device_memory) {
            throw Error(ErrorCode::Budget, "mock: fake device memory exhausted");
        }
        if (used_bytes_ + bytes > cfg_.max_allocation) {
            // Outstanding allocations are capped like a real device limit,
            // so budget exhaustion is exercised by the conformance suite.
            throw Error(ErrorCode::Budget,
                        "mock: outstanding allocations exceed mock device limit");
        }
        // Over-allocate so the device pointer can satisfy the mock's
        // advertised alignment guarantee.
        auto* block = new std::vector<uint8_t>(bytes + cfg_.alignment, 0x00);
        uintptr_t raw = reinterpret_cast<std::uintptr_t>(block->data());
        const uintptr_t aligned =
            (raw + cfg_.alignment - 1) & ~(static_cast<std::uintptr_t>(cfg_.alignment) - 1);
        void* data = reinterpret_cast<void*>(aligned);
        used_bytes_ += bytes;
        blocks_[data] = {block, bytes};
        return data;
    }

    void free(void* ptr) override {
        if (ptr == nullptr) return;
        std::lock_guard lock(mu_);
        auto it = blocks_.find(ptr);
        if (it == blocks_.end()) {
            throw Error(ErrorCode::State, "mock: free of unknown pointer");
        }
        used_bytes_ -= it->second.second;
        delete it->second.first;
        blocks_.erase(it);
    }

    void* allocate_host_pinned(std::size_t bytes) override { return allocate(bytes); }
    void free_host_pinned(void* ptr) override { free(ptr); }
    void* allocate_unified(std::size_t bytes) override { return allocate(bytes); }
    void free_unified(void* ptr) override { free(ptr); }
    void prefetch_to_device(void*, std::size_t) override {}
    void advise_preferred_location(void*, std::size_t) override {}

    DeviceStream* create_stream() override {
        auto* stream = new MockStream();
        std::lock_guard lock(mu_);
        ++active_streams_;
        return stream;
    }
    void destroy_stream(DeviceStream* s) override {
        if (s == nullptr) return;
        {
            std::lock_guard lock(mu_);
            if (active_streams_ == 0) {
                throw Error(ErrorCode::State,
                            "mock: destroy_stream without an active stream");
            }
            --active_streams_;
        }
        delete s;
    }
    DeviceEvent* create_event() override { return new MockEvent(); }
    void destroy_event(DeviceEvent* e) override { delete e; }

    void async_copy_host_to_device(void* dst_device, const void* src_host,
                                   std::size_t bytes, DeviceStream*) override {
        transfer_guard();
        std::memcpy(dst_device, src_host, bytes);
    }

    void async_copy_device_to_host(void* dst_host, const void* src_device,
                                   std::size_t bytes, DeviceStream*) override {
        transfer_guard();
        std::memcpy(dst_host, src_device, bytes);
    }

    void async_copy_device_to_device(void* dst_device, const void* src_device,
                                     std::size_t bytes, DeviceStream*) override {
        transfer_guard();
        std::memcpy(dst_device, src_device, bytes);
    }

    void sync_stream(DeviceStream*) override {}
    void sync_all() override {}

    void record_event(DeviceEvent* event, DeviceStream*) override {
        static_cast<MockEvent*>(event)->recorded = std::chrono::steady_clock::now();
    }

    void wait_event(DeviceEvent*) override {}

    double event_elapsed_us(DeviceEvent* start, DeviceEvent* end) override {
        return std::chrono::duration<double, std::micro>(
                   static_cast<MockEvent*>(end)->recorded -
                   static_cast<MockEvent*>(start)->recorded)
            .count();
    }

    bool healthy() const override { return opened_; }
    std::string diagnostics() const override {
        return "mock backend: " + std::to_string(cfg_.device_count) + " devices, " +
               std::to_string(cfg_.device_memory) + " bytes fake memory";
    }
    std::string last_error() const override { return last_error_; }

    // Failure-injection controls (for conformance tests).
    void set_fail_allocate(bool v) { cfg_.fail_allocate = v; }

    uint64_t used_bytes() const {
        std::lock_guard lock(mu_);
        return used_bytes_;
    }

    std::size_t active_streams() const {
        std::lock_guard lock(mu_);
        return active_streams_;
    }

private:
    class MockStream final : public DeviceStream {};
    class MockEvent final : public DeviceEvent {
    public:
        std::chrono::steady_clock::time_point recorded{};
    };

    void transfer_guard() {
        if (cfg_.fail_transfer) {
            throw Error(ErrorCode::Device, "mock: injected transfer failure");
        }
    }

    Config cfg_;
    mutable std::mutex mu_;
    DeviceInfo info_;
    bool opened_ = false;
    uint64_t used_bytes_ = 0;
    std::size_t active_streams_ = 0;
    std::unordered_map<void*, std::pair<std::vector<uint8_t>*, uint64_t>> blocks_;
    std::string last_error_;
};

}  // namespace flashtier
