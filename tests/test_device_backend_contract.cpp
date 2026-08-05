#include "test_harness.hpp"

#include "conformance.hpp"
#include "flashtier/backends/cpu_backend.hpp"
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

int main() { return ft_test::run_all("test_device_backend_contract"); }
