#include "conformance.hpp"

#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "flashtier/error.hpp"

namespace flashtier {
namespace conformance {

namespace {

void check(bool cond, const char* label, const char* what) {
    if (!cond) {
        throw Error(ErrorCode::Invariant,
                    std::string("conformance[") + label + "] failed: " + what);
    }
}

void expect_error(const std::function<void()>& fn, ErrorCode code, const char* label,
                  const char* what) {
    try {
        fn();
    } catch (const Error& e) {
        check(e.code() == code, label,
              (std::string(what) + ": expected " + error_code_name(code) + ", got " +
               error_code_name(e.code()))
                  .c_str());
        return;
    }
    check(false, label, (std::string(what) + ": expected typed error " +
                         error_code_name(code))
                            .c_str());
}

constexpr std::size_t kTiny = 4096;
constexpr std::size_t kSmall = 65536;

}  // namespace

void run_battery(DeviceBackend& backend, const char* label, bool allow_exhaustion) {
    const std::string suite = std::string(label);

    // ---- creation -----------------------------------------------------------
    check(backend.backend_name() == backend.backend_name(), label, "backend identity");
    check(backend.vendor() != BackendVendor::Unknown, label, "vendor identity");

    // ---- device enumeration --------------------------------------------------
    const std::vector<DeviceInfo> devices = backend.enumerate_devices();
    check(!devices.empty(), label, "enumerate_devices returned no devices");
    check(devices.size() <= 64, label, "enumerate_devices returned absurd count");
    for (std::size_t i = 0; i < devices.size(); ++i) {
        check(!devices[i].name.empty(), label, "device name empty");
        check(!devices[i].id.empty(), label, "device id empty");
        check(devices[i].total_memory > 0, label, "device total_memory is zero");
        check(devices[i].index == static_cast<int>(i), label, "device index ordering");
    }
    // Deterministic enumeration: two calls return the same devices.
    const std::vector<DeviceInfo> again = backend.enumerate_devices();
    check(again.size() == devices.size(), label, "enumeration not deterministic");

    // ---- capability structure validity ----------------------------------------
    const DeviceCapabilities caps = backend.probe_capabilities();
    check(caps.backend_available, label, "backend_available not set");
    check(caps.device_available, label, "device_available not set");
    check(caps.alignment_bytes > 0, label, "alignment_bytes is zero");
    check(caps.transfer_granularity > 0, label, "transfer_granularity is zero");
    check(caps.max_allocation_size > 0, label, "max_allocation_size is zero");
    if (caps.explicit_allocation) {
        check(caps.alignment_bytes >= 16, label, "alignment below 16 bytes");
    }

    // ---- invalid-device rejection ---------------------------------------------
    expect_error([&] { backend.open(static_cast<int>(devices.size())); },
                 ErrorCode::Config, label, "open with out-of-range index");
    expect_error([&] { backend.open(-1); }, ErrorCode::Config, label,
                 "open with negative index");

    // ---- open ---------------------------------------------------------------
    backend.open(0);
    check(backend.is_open(), label, "is_open false after open");
    const DeviceInfo dev0 = backend.device_info();
    check(dev0.id == devices[0].id, label, "device_info mismatch with enumeration");
    check(backend.total_memory() == devices[0].total_memory, label,
          "total_memory mismatch");

    // ---- zero-byte allocation rejection --------------------------------------
    expect_error([&] { backend.allocate(0); }, ErrorCode::InvalidArgument, label,
                 "zero-byte allocation");

    // ---- oversized allocation rejection (limit guard, no real allocation) ----
    expect_error([&] { backend.allocate(caps.max_allocation_size + 1); },
                 ErrorCode::Budget, label, "allocation beyond max_allocation_size");

    // ---- aligned allocation ----------------------------------------------------
    void* a = backend.allocate(kTiny);
    check(reinterpret_cast<std::uintptr_t>(a) % caps.alignment_bytes == 0, label,
          "allocation not aligned");
    backend.free(a);

    // ---- allocation and release cycles -----------------------------------------
    // Content is written through host->device copies: device memory must
    // never be touched directly by host code (mock/cpu backends allow it,
    // real device backends do not).
    for (int i = 0; i < 64; ++i) {
        const std::size_t bytes = kSmall + static_cast<std::size_t>(i) * 4096;
        void* p = backend.allocate(bytes);
        check(p != nullptr, label, "allocation returned null");
        std::vector<uint8_t> host(bytes, static_cast<uint8_t>(i & 0xFF));
        backend.copy_host_to_device_sync(p, host.data(), bytes);
        std::vector<uint8_t> back(bytes, 0);
        backend.copy_device_to_host_sync(back.data(), p, bytes);
        check(std::memcmp(host.data(), back.data(), bytes) == 0, label,
              "allocation/release cycle content mismatch");
        backend.free(p);
    }

    // ---- pinned host allocation (where supported) ------------------------------
    if (caps.pinned_host_allocation) {
        void* pinned = backend.allocate_host_pinned(kSmall);
        check(pinned != nullptr, label, "pinned allocation returned null");
        std::memset(pinned, 0x5A, kSmall);
        std::vector<uint8_t> check_back(kSmall, 0);
        std::memcpy(check_back.data(), pinned, kSmall);
        for (std::size_t i = 0; i < kSmall; ++i) {
            check(check_back[i] == 0x5A, label, "pinned memory content corrupted");
        }
        backend.free_host_pinned(pinned);
    }

    // ---- H2D / D2H round-trip ----------------------------------------------------
    {
        void* d = backend.allocate(kTiny);
        std::vector<uint8_t> host_in(kTiny);
        for (std::size_t i = 0; i < kTiny; ++i) host_in[i] = static_cast<uint8_t>(i * 31);
        backend.copy_host_to_device_sync(d, host_in.data(), kTiny);
        std::vector<uint8_t> host_out(kTiny, 0);
        backend.copy_device_to_host_sync(host_out.data(), d, kTiny);
        check(std::memcmp(host_in.data(), host_out.data(), kTiny) == 0, label,
              "round-trip integrity failed");
        backend.free(d);
    }

    // ---- async queue submission + sync -------------------------------------------
    {
        DeviceStream* stream = backend.create_stream();
        constexpr int kOps = 8;
        std::vector<void*> bufs(kOps);
        std::vector<std::vector<uint8_t>> patterns(kOps);
        for (int i = 0; i < kOps; ++i) {
            bufs[static_cast<std::size_t>(i)] = backend.allocate(kSmall);
            patterns[static_cast<std::size_t>(i)].resize(kSmall);
            for (std::size_t j = 0; j < kSmall; ++j) {
                patterns[static_cast<std::size_t>(i)][j] =
                    static_cast<uint8_t>((i * 53 + j) & 0xFF);
            }
            backend.async_copy_host_to_device(bufs[static_cast<std::size_t>(i)],
                                              patterns[static_cast<std::size_t>(i)].data(),
                                              kSmall, stream);
        }
        backend.sync_stream(stream);
        for (int i = 0; i < kOps; ++i) {
            std::vector<uint8_t> out(kSmall, 0);
            backend.async_copy_device_to_host(out.data(),
                                              bufs[static_cast<std::size_t>(i)], kSmall,
                                              stream);
            backend.sync_stream(stream);
            check(std::memcmp(patterns[static_cast<std::size_t>(i)].data(), out.data(),
                              kSmall) == 0,
                  label, "async copy content mismatch");
        }
        for (int i = 0; i < kOps; ++i) {
            backend.free(bufs[static_cast<std::size_t>(i)]);
        }
        backend.destroy_stream(stream);
    }

    // ---- event timing (where supported) ------------------------------------------
    if (caps.event_timing) {
        DeviceStream* stream = backend.create_stream();
        DeviceEvent* start = backend.create_event();
        DeviceEvent* end = backend.create_event();
        void* d = backend.allocate(kSmall);
        std::vector<uint8_t> host(kSmall, 0x77);
        backend.record_event(start, stream);
        backend.async_copy_host_to_device(d, host.data(), kSmall, stream);
        backend.record_event(end, stream);
        backend.sync_stream(stream);
        const double elapsed = backend.event_elapsed_us(start, end);
        check(elapsed >= 0.0, label, "negative event elapsed time");
        check(elapsed < 10.0 * 1000.0 * 1000.0, label, "implausible event elapsed time");
        backend.free(d);
        backend.destroy_event(end);
        backend.destroy_event(start);
        backend.destroy_stream(stream);
    }

    // ---- typed error propagation ------------------------------------------------
    if (!caps.unified_memory) {
        expect_error([&] { backend.allocate_unified(kTiny); }, ErrorCode::Unsupported,
                     label, "unified allocation without capability");
    }
    if (!caps.memory_prefetch) {
        expect_error([&] { backend.prefetch_to_device(nullptr, 0); },
                     ErrorCode::Unsupported, label, "prefetch without capability");
    }

    // ---- budget exhaustion (bounded fake backends only) ----------------------------
    if (allow_exhaustion) {
        // Exhaust the mock's device memory, then a further allocation must
        // fail with a typed error.
        void* big = backend.allocate(caps.max_allocation_size);
        check(big != nullptr, label, "big allocation failed unexpectedly");
        expect_error([&] { backend.allocate(caps.max_allocation_size); },
                     ErrorCode::Budget, label, "exhausted device memory did not fail");
        backend.free(big);
    }

    // ---- concurrent transfers on two streams ---------------------------------------
    {
        DeviceStream* s1 = backend.create_stream();
        DeviceStream* s2 = backend.create_stream();
        void* d1 = backend.allocate(kSmall);
        void* d2 = backend.allocate(kSmall);
        std::vector<uint8_t> p1(kSmall, 0x11), p2(kSmall, 0x22);
        backend.async_copy_host_to_device(d1, p1.data(), kSmall, s1);
        backend.async_copy_host_to_device(d2, p2.data(), kSmall, s2);
        backend.sync_all();
        std::vector<uint8_t> o1(kSmall, 0), o2(kSmall, 0);
        backend.async_copy_device_to_host(o1.data(), d1, kSmall, s1);
        backend.async_copy_device_to_host(o2.data(), d2, kSmall, s2);
        backend.sync_all();
        check(std::memcmp(p1.data(), o1.data(), kSmall) == 0, label,
              "stream1 copy mismatch");
        check(std::memcmp(p2.data(), o2.data(), kSmall) == 0, label,
              "stream2 copy mismatch");
        backend.free(d2);
        backend.free(d1);
        backend.destroy_stream(s2);
        backend.destroy_stream(s1);
    }

    // ---- no use-after-free during transfer lifecycle --------------------------------
    {
        DeviceStream* stream = backend.create_stream();
        void* d = backend.allocate(kSmall);
        std::vector<uint8_t> host(kSmall, 0xAB);
        backend.async_copy_host_to_device(d, host.data(), kSmall, stream);
        backend.sync_stream(stream);
        backend.free(d);  // sync'd before free: safe
        d = backend.allocate(kSmall);  // reallocate the same budget
        backend.async_copy_host_to_device(d, host.data(), kSmall, stream);
        backend.sync_stream(stream);
        backend.free(d);
        backend.destroy_stream(stream);
    }

    // ---- repeated open/close ---------------------------------------------------------
    for (int i = 0; i < 3; ++i) {
        backend.close();
        check(!backend.is_open(), label, "is_open true after close");
        backend.close();  // idempotent
        backend.open(0);
        check(backend.is_open(), label, "reopen failed");
        void* p = backend.allocate(kTiny);
        backend.free(p);
        backend.close();
    }

    // ---- partial initialization failure then recovery --------------------------------
    backend.close();
    expect_error([&] { backend.open(static_cast<int>(devices.size())); },
                 ErrorCode::Config, label, "re-open with invalid index");
    backend.open(0);  // recovery works
    check(backend.is_open(), label, "recovery open failed");
    void* p = backend.allocate(kTiny);
    backend.free(p);

    // ---- deterministic shutdown ----------------------------------------------------------
    backend.close();
    backend.close();
    check(!backend.is_open(), label, "shutdown left backend open");
}

}  // namespace conformance
}  // namespace flashtier
