#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "flashtier/tier.hpp"

namespace flashtier {

struct GpuInfo {
    std::string name;
    int major = 0;
    int minor = 0;
    uint64_t total_bytes = 0;
    uint64_t free_bytes = 0;
    bool unified_memory = false;
    bool concurrent_managed_access = false;
    bool cuda_available = false;
};

struct SystemInfo {
    std::string os;
    std::string cpu;
    uint64_t total_ram_bytes = 0;
    uint64_t free_ram_bytes = 0;
    std::vector<GpuInfo> gpus;
    bool cuda_enabled = false;
    int cuda_runtime_version = 0;  // 0 = unavailable
    int cuda_driver_version = 0;
    std::string nvme_path;
    uint64_t disk_free_bytes = 0;
    uint64_t disk_total_bytes = 0;
    bool directstorage_detected = false;
    bool gds_detected = false;
    std::string build_version;
    std::string git_revision;
    int device_id = 0;

    // GPU that will be used, or an empty record when CUDA is unavailable.
    GpuInfo primary_gpu() const;
};

// Full machine probe. Safe without CUDA; GPU fields report cuda_available
// = false when the CUDA build is absent or no device can be opened.
SystemInfo probe_system(int device_id, const std::string& nvme_path);

}  // namespace flashtier
