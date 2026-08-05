#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "flashtier/backends/device_backend.hpp"

namespace flashtier {

// Runtime availability status of one compiled backend, from probing only
// (enumeration; no device is opened).
struct BackendStatus {
    std::string name;
    std::string vendor;
    bool available = false;
    std::string reason;  // why unavailable (e.g. "no runtime loader found")
    std::vector<DeviceInfo> devices;
};

// Backend registry: lists compiled backends, probes runtime availability,
// enumerates devices, allows explicit selection, and reports unavailable
// reasons. Deterministic selection order; one backend's failure never
// affects another backend's probe.
class BackendRegistry {
public:
    using Factory = std::function<std::unique_ptr<DeviceBackend>()>;

    static BackendRegistry& instance();

    // Factory registration is done lazily by ensure_registered() via the
    // per-backend register_* functions, so static-library object pruning
    // cannot drop a backend from the registry.
    void register_backend(std::string name, Factory factory, std::string vendor);

    bool has(const std::string& name) const;
    std::vector<std::string> compiled_backends() const;  // sorted, deterministic
    std::vector<BackendStatus> probe_all() const;        // never opens devices
    std::unique_ptr<DeviceBackend> create(const std::string& name) const;

    // Automatic selection: prefer the first compiled backend with a
    // functioning discrete GPU device, then any other accelerator device,
    // then CPU-only. Returns the chosen backend name and sets `reason` to
    // explain the choice (including CPU-only fallback).
    std::string select_automatic(std::string& reason_out) const;

private:
    BackendRegistry() = default;
    std::map<std::string, std::pair<Factory, std::string>> backends_;  // name -> {factory, vendor}
};

// Called by BackendRegistry::instance() on first use. Each backend module
// defines one of these (compiled only when its build option is enabled).
void register_cpu_backend(BackendRegistry& registry);
#if FLASHTIER_HAVE_CUDA
void register_cuda_backend(BackendRegistry& registry);
#endif
#if FLASHTIER_HAVE_HIP
void register_hip_backend(BackendRegistry& registry);
#endif
#if FLASHTIER_HAVE_LEVEL_ZERO
void register_level_zero_backend(BackendRegistry& registry);
#endif
#if FLASHTIER_HAVE_VULKAN
void register_vulkan_backend(BackendRegistry& registry);
#endif
#if FLASHTIER_HAVE_METAL
void register_metal_backend(BackendRegistry& registry);
#endif

}  // namespace flashtier
