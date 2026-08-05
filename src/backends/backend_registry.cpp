#include "flashtier/backends/backend_registry.hpp"

#include <algorithm>

#include "flashtier/error.hpp"

namespace flashtier {

BackendRegistry& BackendRegistry::instance() {
    static BackendRegistry registry;
    static bool initialized = [] {
        register_cpu_backend(registry);
#if FLASHTIER_HAVE_CUDA
        register_cuda_backend(registry);
#endif
#if FLASHTIER_HAVE_HIP
        register_hip_backend(registry);
#endif
#if FLASHTIER_HAVE_LEVEL_ZERO
        register_level_zero_backend(registry);
#endif
#if FLASHTIER_HAVE_VULKAN
        register_vulkan_backend(registry);
#endif
#if FLASHTIER_HAVE_METAL
        register_metal_backend(registry);
#endif
        return true;
    }();
    (void)initialized;
    return registry;
}

void BackendRegistry::register_backend(std::string name, Factory factory,
                                       std::string vendor) {
    backends_[std::move(name)] = {std::move(factory), std::move(vendor)};
}

bool BackendRegistry::has(const std::string& name) const {
    return backends_.count(name) != 0;
}

std::vector<std::string> BackendRegistry::compiled_backends() const {
    std::vector<std::string> names;
    names.reserve(backends_.size());
    for (const auto& [name, entry] : backends_) {
        (void)entry;
        names.push_back(name);
    }
    return names;  // std::map iteration is already sorted and deterministic
}

std::vector<BackendStatus> BackendRegistry::probe_all() const {
    std::vector<BackendStatus> statuses;
    for (const auto& [name, entry] : backends_) {
        BackendStatus status;
        status.name = name;
        status.vendor = entry.second;
        try {
            std::unique_ptr<DeviceBackend> backend = entry.first();
            status.devices = backend->enumerate_devices();
            status.available = !status.devices.empty();
            if (!status.available) {
                status.reason = "no devices discovered";
            }
        } catch (const Error& e) {
            status.available = false;
            status.reason = std::string(e.what());
        } catch (const std::exception& e) {
            status.available = false;
            status.reason = std::string("probe failed: ") + e.what();
        }
        statuses.push_back(std::move(status));
    }
    return statuses;
}

std::unique_ptr<DeviceBackend> BackendRegistry::create(const std::string& name) const {
    auto it = backends_.find(name);
    if (it == backends_.end()) {
        std::string known;
        for (const auto& [n, e] : backends_) {
            (void)e;
            if (!known.empty()) known += ", ";
            known += n;
        }
        throw Error(ErrorCode::Config,
                    "unknown device backend '" + name + "' (compiled backends: " +
                        (known.empty() ? std::string("none") : known) + ")");
    }
    return it->second.first();
}

std::string BackendRegistry::select_automatic(std::string& reason_out) const {
    // Deterministic preference: discrete-GPU backend > any accelerator >
    // CPU-only fallback. Order is fixed; probing isolates failures.
    const std::vector<std::string> preference = {"cuda", "hip", "level_zero",
                                                 "vulkan", "metal", "cpu"};
    for (const std::string& name : preference) {
        auto it = backends_.find(name);
        if (it == backends_.end()) continue;
        try {
            std::unique_ptr<DeviceBackend> backend = it->second.first();
            const std::vector<DeviceInfo> devices = backend->enumerate_devices();
            if (devices.empty()) continue;
            bool any_discrete = false;
            for (const DeviceInfo& d : devices) {
                if (d.discrete) any_discrete = true;
            }
            reason_out = name + " backend selected (" +
                         std::to_string(devices.size()) + " device(s)" +
                         (any_discrete ? ", discrete GPU present" : ", integrated/shared memory") +
                         ")";
            return name;
        } catch (const std::exception& e) {
            reason_out = name + " backend probe failed: " + e.what();
            continue;  // isolate this backend's failure; try the next
        }
    }
    reason_out = "cpu backend selected: no accelerator backend available "
                 "(CPU-only mode; no GPU execution is claimed)";
    return "cpu";
}

}  // namespace flashtier
