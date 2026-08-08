#include "flashtier/backends/backend_registry.hpp"

#include <algorithm>

#include "flashtier/error.hpp"

namespace flashtier {

BackendRegistry& BackendRegistry::instance() {
    static BackendRegistry registry;
    static bool initialized = [] {
#if FLASHTIER_HAVE_CPU_BACKEND
        register_cpu_backend(registry);
#endif
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
    if (name.empty()) {
        throw Error(ErrorCode::InvalidArgument,
                    "cannot register a backend with an empty name");
    }
    if (!factory) {
        throw Error(ErrorCode::InvalidArgument,
                    "cannot register backend '" + name + "' without a factory");
    }
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
            if (!backend) {
                throw Error(ErrorCode::Internal,
                            "backend factory returned null");
            }
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
        } catch (...) {
            status.available = false;
            status.reason = "probe failed with an unknown exception";
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
    std::unique_ptr<DeviceBackend> backend = it->second.first();
    if (!backend) {
        throw Error(ErrorCode::Internal,
                    "backend factory for '" + name + "' returned null");
    }
    return backend;
}

std::string BackendRegistry::select_automatic(std::string& reason_out) const {
    // Probe each accelerator exactly once, then select in two passes so a
    // lower-priority discrete GPU wins over a higher-priority integrated
    // device as promised by the public contract.
    std::vector<std::string> preference = {"cuda", "hip", "level_zero",
                                           "vulkan", "metal"};
    for (const auto& [name, entry] : backends_) {
        (void)entry;
        if (name == "cpu" ||
            std::find(preference.begin(), preference.end(), name) != preference.end()) {
            continue;
        }
        preference.push_back(name);
    }

    struct Candidate {
        std::string name;
        std::vector<DeviceInfo> devices;
    };
    std::vector<Candidate> candidates;
    std::vector<std::string> failures;
    for (const std::string& name : preference) {
        auto it = backends_.find(name);
        if (it == backends_.end()) continue;
        try {
            std::unique_ptr<DeviceBackend> backend = it->second.first();
            if (!backend) {
                throw Error(ErrorCode::Internal,
                            "backend factory returned null");
            }
            std::vector<DeviceInfo> devices = backend->enumerate_devices();
            if (devices.empty()) {
                failures.push_back(name + ": no devices discovered");
                continue;
            }
            candidates.push_back({name, std::move(devices)});
        } catch (const std::exception& e) {
            failures.push_back(name + ": probe failed: " + e.what());
        } catch (...) {
            failures.push_back(name + ": probe failed with an unknown exception");
        }
    }

    for (const Candidate& candidate : candidates) {
        const bool any_discrete = std::any_of(
            candidate.devices.begin(), candidate.devices.end(),
            [](const DeviceInfo& device) { return device.discrete; });
        if (any_discrete) {
            reason_out = candidate.name + " backend selected (" +
                         std::to_string(candidate.devices.size()) +
                         " device(s), discrete GPU present)";
            return candidate.name;
        }
    }
    if (!candidates.empty()) {
        const Candidate& candidate = candidates.front();
        reason_out = candidate.name + " backend selected (" +
                     std::to_string(candidate.devices.size()) +
                     " device(s), integrated/shared memory)";
        return candidate.name;
    }

    auto cpu = backends_.find("cpu");
    if (cpu != backends_.end()) {
        try {
            std::unique_ptr<DeviceBackend> backend = cpu->second.first();
            if (!backend) {
                throw Error(ErrorCode::Internal,
                            "backend factory returned null");
            }
            if (!backend->enumerate_devices().empty()) {
                reason_out = "cpu backend selected: no accelerator backend available "
                             "(CPU-only mode; no GPU execution is claimed)";
                if (!failures.empty()) {
                    reason_out += "; accelerator probes: ";
                    for (std::size_t i = 0; i < failures.size(); ++i) {
                        if (i != 0) reason_out += "; ";
                        reason_out += failures[i];
                    }
                }
                return "cpu";
            }
            failures.push_back("cpu: no emulation device discovered");
        } catch (const std::exception& e) {
            failures.push_back(std::string("cpu: probe failed: ") + e.what());
        } catch (...) {
            failures.push_back("cpu: probe failed with an unknown exception");
        }
    } else {
        failures.push_back("cpu: backend not compiled");
    }

    reason_out = "no usable device backend";
    if (!failures.empty()) {
        reason_out += ": ";
        for (std::size_t i = 0; i < failures.size(); ++i) {
            if (i != 0) reason_out += "; ";
            reason_out += failures[i];
        }
    }
    throw Error(ErrorCode::Unsupported, reason_out);
}

}  // namespace flashtier
