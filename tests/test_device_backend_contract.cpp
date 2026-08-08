#include "test_harness.hpp"

#include <atomic>
#include <filesystem>
#include <limits>
#include <mutex>
#include <thread>

#include "conformance.hpp"
#include "flashtier/backends/backend_registry.hpp"
#include "flashtier/backends/cpu_backend.hpp"
#include "flashtier/backends/host_backend.hpp"
#if FLASHTIER_HAVE_LEVEL_ZERO
#include "flashtier/backends/level_zero_backend.hpp"
#endif
#if FLASHTIER_HAVE_VULKAN
#include "flashtier/backends/vulkan_backend.hpp"
#endif
#if FLASHTIER_HAVE_METAL
#include "flashtier/backends/metal_backend.hpp"
#endif
#include "flashtier/device_info.hpp"
#include "flashtier/error.hpp"
#include "mock_device_backend.hpp"

using namespace flashtier;

// Generic backend conformance: runs the full battery against the mock and
// CPU backends on every platform. A mock passing never validates a vendor
// backend; vendor backends must pass the same battery on real hardware.

FT_TEST(mock_backend_conformance_battery) {
    MockDeviceBackend backend;
    conformance::run_battery(backend, "mock", /*allow_exhaustion=*/true);
}

FT_TEST(mock_backend_failure_injection) {
    MockDeviceBackend::Config cfg;
    cfg.fail_allocate = true;
    MockDeviceBackend backend(cfg);
    backend.open(0);
    bool threw = false;
    try {
        backend.allocate(4096);
    } catch (const Error& e) {
        threw = true;
        FT_ASSERT_EQ(e.code(), ErrorCode::Budget);
    }
    FT_ASSERT(threw);
    backend.close();
}

FT_TEST(mock_backend_injected_open_failure) {
    MockDeviceBackend::Config cfg;
    cfg.fail_open = true;
    MockDeviceBackend backend(cfg);
    FT_ASSERT_THROWS(backend.open(0), ErrorCode::Device);
    FT_ASSERT(!backend.is_open());
    cfg.fail_open = false;
    MockDeviceBackend ok;
    ok.open(0);
    FT_ASSERT(ok.is_open());
    ok.close();
}

FT_TEST(mock_backend_injected_transfer_failure) {
    MockDeviceBackend::Config cfg;
    cfg.fail_transfer = true;
    MockDeviceBackend backend(cfg);
    backend.open(0);
    void* d = backend.allocate(4096);
    std::vector<uint8_t> host(4096, 0x11);
    FT_ASSERT_THROWS(backend.copy_host_to_device_sync(d, host.data(), 4096),
                     ErrorCode::Device);
    FT_ASSERT_EQ(backend.active_streams(), 0u);
    backend.free(d);
    backend.close();
}

FT_TEST(mock_backend_budget_accounting) {
    MockDeviceBackend backend;
    backend.open(0);
    FT_ASSERT_EQ(backend.used_bytes(), 0u);
    void* a = backend.allocate(4096);
    FT_ASSERT_EQ(backend.used_bytes(), 4096u);
    backend.free(a);
    FT_ASSERT_EQ(backend.used_bytes(), 0u);
    FT_ASSERT_THROWS(backend.free(a), ErrorCode::State);  // double free rejected
    backend.close();
}

#if FLASHTIER_HAVE_CPU_BACKEND

FT_TEST(cpu_backend_conformance_battery) {
    CpuBackend backend;
    conformance::run_battery(backend, "cpu", /*allow_exhaustion=*/true);
}

FT_TEST(cpu_backend_identity) {
    CpuBackend backend;
    FT_ASSERT_EQ(backend.backend_name(), std::string("cpu"));
    FT_ASSERT_EQ(backend.vendor(), BackendVendor::Portable);
    auto devices = backend.enumerate_devices();
    FT_ASSERT_EQ(devices.size(), 1u);
    FT_ASSERT(devices[0].memory_shared);
    FT_ASSERT(!devices[0].discrete);
}

FT_TEST(cpu_backend_rejects_operations_while_closed) {
    CpuBackend backend;
    FT_ASSERT_THROWS(backend.capabilities(), ErrorCode::State);
    FT_ASSERT_THROWS(backend.allocate(1), ErrorCode::State);
    FT_ASSERT_THROWS(backend.create_stream(), ErrorCode::State);
    FT_ASSERT_THROWS(backend.create_event(), ErrorCode::State);
    FT_ASSERT_THROWS(backend.sync_all(), ErrorCode::State);
}

FT_TEST(cpu_backend_close_reclaims_owned_resources) {
    CpuBackend backend;
    backend.open(0);
    const uint64_t free_before = backend.free_memory();
    void* allocation = backend.allocate(4097);  // exercises non-aligned sizes
    DeviceStream* stream = backend.create_stream();
    DeviceEvent* event = backend.create_event();
    FT_ASSERT(backend.free_memory() < free_before);
    backend.close();

    backend.open(0);
    FT_ASSERT_EQ(backend.free_memory(), free_before);
    FT_ASSERT_THROWS(backend.free(allocation), ErrorCode::State);
    FT_ASSERT_THROWS(backend.destroy_stream(stream), ErrorCode::State);
    FT_ASSERT_THROWS(backend.destroy_event(event), ErrorCode::State);
    backend.close();
}

FT_TEST(cpu_backend_rejects_invalid_transfer_ranges) {
    CpuBackend backend;
    backend.open(0);
    void* allocation = backend.allocate(16);
    std::vector<uint8_t> host(17, 0xA5);
    FT_ASSERT_THROWS(
        backend.async_copy_host_to_device(allocation, host.data(), host.size(), nullptr),
        ErrorCode::InvalidArgument);
    FT_ASSERT_THROWS(
        backend.async_copy_device_to_host(host.data(), allocation, host.size(), nullptr),
        ErrorCode::InvalidArgument);
    backend.free(allocation);
    backend.close();
}

FT_TEST(host_backend_rejects_malformed_ownership_operations) {
    CpuBackend device;
    device.open(0);
    HostBackend host;
    host.set_budget(256 * 1024, 0.0);
    host.set_device_backend(&device);

    FT_ASSERT_THROWS(host.allocate(0, false), ErrorCode::InvalidArgument);
    void* ptr = host.allocate(64 * 1024, false);
    FT_ASSERT(ptr != nullptr);
    FT_ASSERT_THROWS(host.free(ptr, 32 * 1024), ErrorCode::InvalidArgument);
    FT_ASSERT_EQ(host.used(), 64u * 1024u);
    FT_ASSERT_THROWS(host.set_device_backend(nullptr), ErrorCode::State);
    host.free(ptr, 64 * 1024);
    FT_ASSERT_EQ(host.used(), 0u);
    FT_ASSERT_THROWS(host.free(ptr, 64 * 1024), ErrorCode::State);
    host.set_device_backend(nullptr);
    device.close();

    FT_ASSERT_THROWS(host.set_budget(1024, -0.01), ErrorCode::InvalidArgument);
    FT_ASSERT_THROWS(host.set_budget(1024, 1.01), ErrorCode::InvalidArgument);
    FT_ASSERT_THROWS(
        host.set_budget(1024, std::numeric_limits<double>::quiet_NaN()),
        ErrorCode::InvalidArgument);
}

FT_TEST(host_backend_never_silently_substitutes_pageable_memory) {
    HostBackend host;
    host.set_budget(64 * 1024, 0.0);
    FT_ASSERT_THROWS(host.allocate(4096, false), ErrorCode::Unsupported);
    void* pageable = host.allocate(4096, true);
    FT_ASSERT(pageable != nullptr);
    FT_ASSERT_EQ(host.allocation_tier(pageable), Tier::HostPageable);
    host.free(pageable, 4096);
}

FT_TEST(host_backend_concurrent_budget_claims_never_overshoot) {
    CpuBackend device;
    device.open(0);
    HostBackend host;
    constexpr uint64_t kAllocation = 64 * 1024;
    constexpr uint64_t kLimit = 4 * kAllocation;
    host.set_budget(kLimit, 0.0);
    host.set_device_backend(&device);

    std::atomic<bool> start{false};
    std::mutex results_mu;
    std::vector<void*> allocations;
    std::vector<std::thread> threads;
    for (int i = 0; i < 16; ++i) {
        threads.emplace_back([&] {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            void* ptr = host.allocate(kAllocation, false);
            if (ptr != nullptr) {
                std::lock_guard lock(results_mu);
                allocations.push_back(ptr);
            }
        });
    }
    start.store(true, std::memory_order_release);
    for (std::thread& thread : threads) thread.join();

    FT_ASSERT(allocations.size() <= 4);
    FT_ASSERT(host.used() <= kLimit);
    FT_ASSERT_EQ(host.used(), allocations.size() * kAllocation);
    for (void* ptr : allocations) host.free(ptr, kAllocation);
    host.set_device_backend(nullptr);
    device.close();
}

#endif

FT_TEST(registry_matches_cpu_backend_feature_gate) {
#if FLASHTIER_HAVE_CPU_BACKEND
    FT_ASSERT(BackendRegistry::instance().has("cpu"));
#else
    FT_ASSERT(!BackendRegistry::instance().has("cpu"));
#endif
}

FT_TEST(system_probe_resolves_default_and_planned_store_volumes) {
    const SystemInfo default_path = probe_system(0, "");
    FT_ASSERT(default_path.total_ram_bytes > 0);
    FT_ASSERT(default_path.free_ram_bytes > 0);
    FT_ASSERT(default_path.disk_total_bytes > 0);
    FT_ASSERT(default_path.disk_free_bytes > 0);
    FT_ASSERT(!default_path.cpu.empty());

    const std::filesystem::path planned =
        std::filesystem::temp_directory_path() /
        "flashtier-capacity-probe-nonexistent" / "store.bin";
    const SystemInfo planned_path = probe_system(0, planned.string());
    FT_ASSERT(planned_path.disk_total_bytes > 0);
    FT_ASSERT(planned_path.disk_free_bytes > 0);
}

#if FLASHTIER_HAVE_LEVEL_ZERO

FT_TEST(level_zero_backend_fails_closed_without_loader_calls) {
    LevelZeroBackend backend;
    FT_ASSERT(backend.enumerate_devices().empty());
    const DeviceCapabilities caps = backend.probe_capabilities();
    FT_ASSERT(!caps.backend_available);
    FT_ASSERT(!caps.device_available);
    FT_ASSERT(caps.note.find("disabled") != std::string::npos);
    FT_ASSERT_THROWS(backend.open(0), ErrorCode::Unsupported);
    FT_ASSERT(!backend.is_open());
    FT_ASSERT(!backend.healthy());
    FT_ASSERT_THROWS(backend.allocate(4096), ErrorCode::Unsupported);
    backend.free(nullptr);
    backend.free_host_pinned(nullptr);
    backend.free_unified(nullptr);
    backend.destroy_stream(nullptr);
    backend.destroy_event(nullptr);
    backend.close();
}

#endif

#if FLASHTIER_HAVE_VULKAN

FT_TEST(vulkan_backend_fails_closed_without_loader_calls) {
    VulkanBackend backend;
    FT_ASSERT(backend.enumerate_devices().empty());
    const DeviceCapabilities caps = backend.probe_capabilities();
    FT_ASSERT(!caps.backend_available);
    FT_ASSERT(!caps.device_available);
    FT_ASSERT(caps.note.find("disabled") != std::string::npos);
    FT_ASSERT_THROWS(backend.open(0), ErrorCode::Unsupported);
    FT_ASSERT(!backend.is_open());
    FT_ASSERT(!backend.healthy());
    FT_ASSERT_THROWS(backend.allocate(4096), ErrorCode::Unsupported);
    backend.free(nullptr);
    backend.free_host_pinned(nullptr);
    backend.free_unified(nullptr);
    backend.destroy_stream(nullptr);
    backend.destroy_event(nullptr);
    backend.close();
}

#endif

#if FLASHTIER_HAVE_METAL

FT_TEST(metal_backend_fails_closed_without_framework_calls) {
    MetalBackend backend;
    FT_ASSERT(backend.enumerate_devices().empty());
    const DeviceCapabilities caps = backend.probe_capabilities();
    FT_ASSERT(!caps.backend_available);
    FT_ASSERT(!caps.device_available);
    FT_ASSERT(caps.note.find("disabled") != std::string::npos);
    FT_ASSERT_THROWS(backend.open(0), ErrorCode::Unsupported);
    FT_ASSERT(!backend.is_open());
    FT_ASSERT(!backend.healthy());
    FT_ASSERT_THROWS(backend.allocate(4096), ErrorCode::Unsupported);
    backend.free(nullptr);
    backend.free_host_pinned(nullptr);
    backend.free_unified(nullptr);
    backend.destroy_stream(nullptr);
    backend.destroy_event(nullptr);
    backend.close();
}

#endif

#if !FLASHTIER_HAVE_CPU_BACKEND && !FLASHTIER_HAVE_CUDA && \
    !FLASHTIER_HAVE_HIP && !FLASHTIER_HAVE_LEVEL_ZERO && \
    !FLASHTIER_HAVE_VULKAN && !FLASHTIER_HAVE_METAL

FT_TEST(registry_auto_without_any_backend_is_typed_unsupported) {
    std::string reason;
    FT_ASSERT_THROWS(BackendRegistry::instance().select_automatic(reason),
                     ErrorCode::Unsupported);
    FT_ASSERT(reason.find("no usable device backend") != std::string::npos);
    FT_ASSERT(reason.find("cpu: backend not compiled") != std::string::npos);
}

#endif

int main() { return ft_test::run_all("test_device_backend_contract"); }
