#include "flashtier/device_info.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <system_error>
#include <thread>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#if defined(_M_IX86) || defined(_M_X64)
#include <intrin.h>
#endif
#else
#if FLASHTIER_ENABLE_GDS
#include <dlfcn.h>
#endif
#include <unistd.h>
#include <sys/statvfs.h>
#include <sys/utsname.h>
#include <fstream>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#endif

#if FLASHTIER_HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace flashtier {

namespace {

uint64_t ram_total_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    return GlobalMemoryStatusEx(&ms) ? ms.ullTotalPhys : 0;
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
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    return GlobalMemoryStatusEx(&ms) ? ms.ullAvailPhys : 0;
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

std::string cpu_architecture() {
#if defined(_M_X64) || defined(__x86_64__)
    return "x86-64";
#elif defined(_M_ARM64) || defined(__aarch64__)
    return "ARM64";
#elif defined(_M_IX86) || defined(__i386__)
    return "x86";
#elif defined(__arm__)
    return "ARM";
#else
    return "unknown architecture";
#endif
}

std::string trim_cpu_name(std::string value) {
    const auto whitespace = [](char ch) {
        return ch == '\0' || ch == ' ' || ch == '\t' || ch == '\r' ||
               ch == '\n';
    };
    while (!value.empty() && whitespace(value.back())) value.pop_back();
    std::size_t first = 0;
    while (first < value.size() && whitespace(value[first])) ++first;
    return value.substr(first);
}

std::string cpu_model() {
#if defined(_WIN32) && (defined(_M_IX86) || defined(_M_X64))
    std::array<int, 4> registers{};
    __cpuid(registers.data(), static_cast<int>(0x80000000u));
    if (static_cast<unsigned>(registers[0]) >= 0x80000004u) {
        std::array<char, 49> brand{};
        for (unsigned leaf = 0; leaf < 3; ++leaf) {
            __cpuid(registers.data(), static_cast<int>(0x80000002u + leaf));
            std::memcpy(brand.data() + leaf * 16, registers.data(), 16);
        }
        const std::string model = trim_cpu_name(brand.data());
        if (!model.empty()) return model;
    }
#elif defined(__linux__)
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        const std::size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        const std::string key = trim_cpu_name(line.substr(0, colon));
        if (key == "model name" || key == "Hardware") {
            const std::string model = trim_cpu_name(line.substr(colon + 1));
            if (!model.empty()) return model;
        }
    }
#elif defined(__APPLE__)
    std::size_t size = 0;
    if (sysctlbyname("machdep.cpu.brand_string", nullptr, &size, nullptr, 0) == 0 &&
        size > 1) {
        std::string model(size, '\0');
        if (sysctlbyname("machdep.cpu.brand_string", model.data(), &size,
                         nullptr, 0) == 0) {
            model.resize(size);
            model = trim_cpu_name(model);
            if (!model.empty()) return model;
        }
    }
#endif
    return cpu_architecture();
}

std::string cpu_name() {
    std::string name = cpu_model();
    const std::string architecture = cpu_architecture();
    if (name != architecture) name += " (" + architecture + ")";
    const unsigned logical_threads = std::thread::hardware_concurrency();
    if (logical_threads != 0) {
        name += ", " + std::to_string(logical_threads) + " logical threads";
    }
    return name;
}

std::filesystem::path capacity_probe_directory(const std::string& path) {
    std::error_code ec;
    std::filesystem::path candidate;
    if (path.empty()) {
        candidate = std::filesystem::temp_directory_path(ec);
        if (ec) {
            ec.clear();
            candidate = std::filesystem::current_path(ec);
        }
    } else {
        candidate = std::filesystem::path(path);
        if (candidate.is_relative()) {
            const std::filesystem::path absolute =
                std::filesystem::absolute(candidate, ec);
            if (!ec) candidate = absolute;
            ec.clear();
        }
    }

    while (!candidate.empty()) {
        const std::filesystem::file_status status =
            std::filesystem::status(candidate, ec);
        if (!ec && std::filesystem::exists(status)) {
            if (std::filesystem::is_directory(status)) return candidate;
            const std::filesystem::path parent = candidate.parent_path();
            return parent.empty() ? candidate : parent;
        }
        ec.clear();
        const std::filesystem::path parent = candidate.parent_path();
        if (parent.empty() || parent == candidate) break;
        candidate = parent;
    }
    return {};
}

void disk_capacity(const std::string& path, uint64_t& free_bytes, uint64_t& total_bytes) {
    free_bytes = 0;
    total_bytes = 0;
    const std::filesystem::path directory = capacity_probe_directory(path);
    if (directory.empty()) return;
#if defined(_WIN32)
    ULARGE_INTEGER free_, total_;
    free_.QuadPart = 0;
    total_.QuadPart = 0;
    if (GetDiskFreeSpaceExW(directory.c_str(), &free_, &total_, nullptr)) {
        free_bytes = free_.QuadPart;
        total_bytes = total_.QuadPart;
    }
#else
    struct statvfs vfs{};
    if (statvfs(directory.c_str(), &vfs) == 0) {
        const uint64_t frag = static_cast<uint64_t>(vfs.f_frsize);
        if (frag != 0 &&
            static_cast<uint64_t>(vfs.f_bavail) <=
                std::numeric_limits<uint64_t>::max() / frag &&
            static_cast<uint64_t>(vfs.f_blocks) <=
                std::numeric_limits<uint64_t>::max() / frag) {
            free_bytes = static_cast<uint64_t>(vfs.f_bavail) * frag;
            total_bytes = static_cast<uint64_t>(vfs.f_blocks) * frag;
        }
    }
#endif
}

bool directstorage_probe() {
#if defined(_WIN32) && FLASHTIER_ENABLE_DIRECTSTORAGE
    // Capability probe only; no integration in v0.1. Load from the normal
    // safe DLL search set, and always close the probe handle.
    static const wchar_t* kDlls[] = {L"dstorage.dll", L"dstoragecore.dll"};
    for (const wchar_t* dll : kDlls) {
        HMODULE module = LoadLibraryExW(dll, nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (module != nullptr) {
            FreeLibrary(module);
            return true;
        }
    }
    return false;
#else
    return false;
#endif
}

bool gds_probe() {
#if defined(__linux__) && FLASHTIER_ENABLE_GDS
    // cuFile loader presence check; future backend. Use the platform loader
    // search path rather than looking for a file named libcufile.so in cwd.
    void* module = dlopen("libcufile.so", RTLD_LAZY | RTLD_LOCAL);
    if (module != nullptr) {
        dlclose(module);
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
    int original_device = -1;
    const bool restore_device = cudaGetDevice(&original_device) == cudaSuccess;
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
        if (cudaSetDevice(i) != cudaSuccess) {
            g.cuda_available = false;
            info.gpus.push_back(g);
            continue;
        }
        std::size_t free_ = 0;
        std::size_t total_ = 0;
        cudaError_t me = cudaMemGetInfo(&free_, &total_);
        if (me == cudaSuccess) {
            g.free_bytes = static_cast<uint64_t>(free_);
        } else {
            g.free_bytes = 0;  // unknown; never overclaim available capacity
        }
        info.gpus.push_back(g);
    }
    if (restore_device) {
        (void)cudaSetDevice(original_device);
    }
#endif
    return info;
}

}  // namespace flashtier
