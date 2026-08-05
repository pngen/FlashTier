#include "test_harness.hpp"

#include <cstring>
#include <vector>

#include "conformance.hpp"
#include "flashtier/backends/cuda_backend.hpp"
#include "flashtier/error.hpp"
#include "flashtier/integrity.hpp"

using namespace flashtier;

#if FLASHTIER_HAVE_CUDA

// CUDA backend conformance + allocation/round-trip checks. Runs on the
// installed NVIDIA GPU (LABELS gpu); skipped cleanly when no CUDA device
// is available.

FT_TEST(cuda_backend_conformance_battery) {
    CudaBackend backend;
    if (backend.enumerate_devices().empty()) {
        std::printf("SKIP: no CUDA device available\n");
        return;
    }
    conformance::run_battery(backend, "cuda", /*allow_exhaustion=*/false);
}

FT_TEST(cuda_backend_device_discovery) {
    CudaBackend backend;
    const auto devices = backend.enumerate_devices();
    if (devices.empty()) {
        std::printf("SKIP: no CUDA device available\n");
        return;
    }
    backend.open(0);
    const DeviceInfo info = backend.device_info();
    FT_ASSERT(!info.name.empty());
    FT_ASSERT(info.total_memory > 0);
    FT_ASSERT(info.free_memory > 0);
    const DeviceCapabilities caps = backend.capabilities();
    FT_ASSERT(caps.explicit_allocation);
    FT_ASSERT(caps.async_host_to_device);
    FT_ASSERT(caps.pinned_host_allocation);
    FT_ASSERT(caps.alignment_bytes >= 256);
    FT_ASSERT(caps.max_allocation_size == info.total_memory);
    backend.close();
}

FT_TEST(cuda_backend_tiny_round_trip) {
    CudaBackend backend;
    if (backend.enumerate_devices().empty()) {
        std::printf("SKIP: no CUDA device available\n");
        return;
    }
    backend.open(0);

    // Tiny allocation (16 MiB or less) with full byte-for-byte integrity.
    const std::size_t bytes = 16ull * 1024 * 1024;
    void* dev = backend.allocate(bytes);
    FT_ASSERT(dev != nullptr);
    void* pinned = backend.allocate_host_pinned(bytes);
    FT_ASSERT(pinned != nullptr);

    std::vector<uint8_t> pattern(bytes);
    for (std::size_t i = 0; i < bytes; ++i) {
        pattern[i] = static_cast<uint8_t>((i * 131 + 7) & 0xFF);
    }

    backend.copy_host_to_device_sync(dev, pattern.data(), bytes);
    std::vector<uint8_t> out(bytes, 0);
    backend.copy_device_to_host_sync(out.data(), dev, bytes);
    FT_ASSERT(std::memcmp(pattern.data(), out.data(), bytes) == 0);

    // Deterministic integrity pattern round-trip (matches runtime fill).
    fill_pattern(pinned, bytes, 0xC0FFEE, 7);
    backend.copy_host_to_device_sync(dev, pinned, bytes);
    backend.copy_device_to_host_sync(out.data(), dev, bytes);
    FT_ASSERT(!verify_pattern(out.data(), bytes, 0xC0FFEE, 7).has_value());

    backend.free_host_pinned(pinned);
    backend.free(dev);
    backend.close();
}

FT_TEST(cuda_backend_repeated_open_close) {
    CudaBackend backend;
    if (backend.enumerate_devices().empty()) {
        std::printf("SKIP: no CUDA device available\n");
        return;
    }
    for (int i = 0; i < 3; ++i) {
        backend.open(0);
        void* p = backend.allocate(4096);
        backend.free(p);
        backend.close();
    }
    backend.close();  // idempotent
    FT_ASSERT(!backend.is_open());
}

FT_TEST(cuda_backend_typed_errors) {
    CudaBackend backend;
    if (backend.enumerate_devices().empty()) {
        std::printf("SKIP: no CUDA device available\n");
        return;
    }
    backend.open(0);
    FT_ASSERT_THROWS(backend.allocate(0), ErrorCode::InvalidArgument);
    const DeviceCapabilities caps = backend.capabilities();
    FT_ASSERT_THROWS(backend.allocate(caps.max_allocation_size + 1), ErrorCode::Budget);
    FT_ASSERT_THROWS(backend.open(999), ErrorCode::Config);
    backend.close();
}

#else

FT_TEST(cuda_backend_not_compiled) {
    CudaBackend backend;
    FT_ASSERT(backend.enumerate_devices().empty());
    FT_ASSERT_THROWS(backend.open(0), ErrorCode::Unsupported);
}

#endif

int main() { return ft_test::run_all("test_cuda_device_backend"); }
