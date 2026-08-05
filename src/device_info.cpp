#include "flashtier/device_info.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#include <sys/statvfs.h>
#include <sys/utsname.h>
#include <fstream>
#endif

#if FLASHTIER_HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace flashtier {

namespace {

uint64_t ram_total_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    GlobalMemoryStatusEx(&ms);
    return ms.ullTotalPhys;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return pages > 0 && page_size > 0
               ? static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size)
               : 0;
#endif
}

uint64_t ram_free_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    GlobalMemoryStatusEx(&ms);
    return ms.ullAvailPhys;
#else
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return pages > 0 && page_size > 0
               ? static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size)
               : 0;
#endif
}

std::string os_name() {
#if defined(_WIN32)
    return "Windows";
#elif defined(__linux__)
    return "Linux";
#elif defined(__APPLE__)
    return "macOS";
#else
    return "Unknown";
#endif
}

std::string cpu_name() {
#if defined(_WIN32)
    return "x86-64";
#else
    return "x86-64/ARM64";
#endif
}

void disk_capacity(const std::string& path, uint64_t& free_bytes, uint64_t& total_bytes) {
#if defined(_WIN32)
    std::string root = path.empty() ? "." : path;
    if (root.size() >= 3 && root[1] == ':') {
        root = root.substr(0, 3);  // "C:\"
    } else if (root.size() < 3) {
        root = "C:\\";
    }
    ULARGE_INTEGER free_, total_;
    free_.QuadPart = 0;
    total_.QuadPart = 0;
    if (GetDiskFreeSpaceExA(root.c_str(), &free_, &total_, nullptr)) {
        free_bytes = free_.QuadPart;
        total_bytes = total_.QuadPart;
    }
#else
    struct statvfs vfs;
    if (statvfs(path.c_str(), &vfs) == 0) {
        const uint64_t frag = static_cast<uint64_t>(vfs.f_frsize);
        free_bytes = static_cast<uint64_t>(vfs.f_bavail) * frag;
        total_bytes = static_cast<uint64_t>(vfs.f_blocks) * frag;
    }
#endif
}

bool directstorage_probe() {
#if defined(_WIN32) && FLASHTIER_ENABLE_DIRECTSTORAGE
    // Capability probe only; no integration in v0.1. Windows 10 1903+
    // ships DirectStorage; we report presence of the platform DLL.
    static const char* kDlls[] = {"dstorage.dll", "dstoragecore.dll"};
    for (const char* dll : kDlls) {
        if (LoadLibraryA(dll) != nullptr) return true;
    }
    return false;
#else
    return false;
#endif
}

bool gds_probe() {
#if defined(__linux__) && FLASHTIER_ENABLE_GDS
    // cuFile (libcufile.so) presence check; future backend.
    FILE* f = std::fopen("libcufile.so", "rb");
    if (f != nullptr) {
        std::fclose(f);
        return true;
    }
#endif
    return false;
}

}  // namespace

GpuInfo SystemInfo::primary_gpu() const {
    if (gpus.empty()) return GpuInfo{};
    std::size_t idx = 0;
    if (device_id >= 0 && static_cast<std::size_t>(device_id) < gpus.size()) {
        idx = static_cast<std::size_t>(device_id);
    }
    return gpus[idx];
}

SystemInfo probe_system(int device_id, const std::string& nvme_path) {
    SystemInfo info;
    info.os = os_name();
    info.cpu = cpu_name();
    info.total_ram_bytes = ram_total_bytes();
    info.free_ram_bytes = ram_free_bytes();
    info.device_id = device_id;
    info.nvme_path = nvme_path;
    disk_capacity(nvme_path, info.disk_free_bytes, info.disk_total_bytes);
    info.directstorage_detected = directstorage_probe();
    info.gds_detected = gds_probe();
    info.build_version = FLASHTIER_VERSION;
    info.git_revision = FLASHTIER_GIT_REVISION;

#if FLASHTIER_HAVE_CUDA
    info.cuda_enabled = true;
    int rt = 0;
    int drv = 0;
    cudaError_t e = cudaRuntimeGetVersion(&rt);
    if (e == cudaSuccess) info.cuda_runtime_version = rt;
    e = cudaDriverGetVersion(&drv);
    if (e == cudaSuccess) info.cuda_driver_version = drv;

    int count = 0;
    e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess || count <= 0) {
        return info;
    }
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};
        cudaError_t pe = cudaGetDeviceProperties(&prop, i);
        GpuInfo g;
        if (pe != cudaSuccess) {
            g.name = "cuda device " + std::to_string(i);
            g.cuda_available = false;
            info.gpus.push_back(g);
            continue;
        }
        g.name = prop.name;
        g.major = prop.major;
        g.minor = prop.minor;
        g.total_bytes = static_cast<uint64_t>(prop.totalGlobalMem);
        g.unified_memory = prop.managedMemory != 0;
        g.concurrent_managed_access = prop.concurrentManagedAccess != 0;
        g.cuda_available = true;
        cudaSetDevice(i);
        std::size_t free_ = 0;
        std::size_t total_ = 0;
        cudaError_t me = cudaMemGetInfo(&free_, &total_);
        if (me == cudaSuccess) {
            g.free_bytes = static_cast<uint64_t>(free_);
        } else {
            g.free_bytes = g.total_bytes;  // unknown; report conservatively
        }
        info.gpus.push_back(g);
    }
    cudaSetDevice(device_id >= 0 && device_id < count ? device_id : 0);
#endif
    return info;
}

}  // namespace flashtier
