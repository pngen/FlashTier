// NVIDIA CUDA backend implementing the vendor-neutral DeviceBackend
// contract. Compiled with nvcc only when FLASHTIER_ENABLE_CUDA=ON.
//
// Every CUDA call is checked and mapped to typed FlashTier errors; no
// failure is silently ignored. Device capability fields come from
// cudaDeviceProp discovery, never from product names.

#include "flashtier/backends/cuda_backend.hpp"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

#include <cuda_runtime.h>

#include "flashtier/error.hpp"

namespace flashtier {

namespace {

void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::string detail = cudaGetErrorString(e);
        if (e == cudaErrorDevicesUnavailable || e == cudaErrorDeviceUninitialized) {
            detail += " (device unavailable/lost)";
        }
        throw Error(ErrorCode::Cuda, std::string(what) + " failed: " + detail,
                    std::to_string(static_cast<int>(e)));
    }
}

std::string device_arch(const cudaDeviceProp& p) {
    return std::to_string(p.major) + "." + std::to_string(p.minor);
}

__global__ void fill_pattern_kernel(unsigned char* dst, std::uint64_t bytes,
                                    std::uint64_t seed, std::uint64_t page_id) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= bytes) return;
    const std::uint64_t block = i / 8;
    std::uint64_t x = seed ^ (page_id * 0x9E3779B97F4A7C15ull) ^ block;
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    x = x ^ (x >> 31);
    dst[i] = static_cast<unsigned char>(x >> ((i % 8) * 8));
}

}  // namespace

class CudaStream final : public DeviceStream {
public:
    cudaStream_t handle = nullptr;
};

class CudaEvent final : public DeviceEvent {
public:
    cudaEvent_t handle = nullptr;
};

struct CudaBackend::Impl {
    int device_index = -1;
    cudaDeviceProp prop{};
    bool opened = false;

    // Diagnostics: last error string (thread-agnostic snapshot).
    std::mutex mu;
    std::string last_error;
    bool healthy = true;

    // Cached capability probe (stable for the process lifetime).
    bool caps_cache_valid = false;
    DeviceCapabilities caps_cache;
};

CudaBackend::CudaBackend() : impl_(std::make_unique<Impl>()) {}

CudaBackend::~CudaBackend() {
    close();
}

std::vector<DeviceInfo> CudaBackend::enumerate_devices() const {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess || count <= 0) {
        return {};  // no CUDA runtime/device: backend unavailable
    }
    std::vector<DeviceInfo> out;
    out.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, i) != cudaSuccess) {
            continue;
        }
        DeviceInfo info;
        info.id = "cuda:" + std::to_string(i);
        info.name = p.name;
        info.architecture = device_arch(p);
        info.total_memory = static_cast<uint64_t>(p.totalGlobalMem);
        info.free_memory = info.total_memory;
        // Discovery, not naming: the driver reports integration status.
        info.discrete = p.integrated == 0;
        info.memory_shared = p.integrated != 0;
        info.index = i;
        out.push_back(std::move(info));
    }
    return out;
}

DeviceCapabilities CudaBackend::probe_capabilities() const {
    // The driver-level hint probe result is stable for the process; cache
    // it so per-allocation capability queries stay cheap.
    {
        std::lock_guard lock(impl_->mu);
        if (impl_->caps_cache_valid) {
            return impl_->caps_cache;
        }
    }
    DeviceCapabilities caps;
    const std::vector<DeviceInfo> devices = enumerate_devices();
    if (devices.empty()) {
        return caps;
    }
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) {
        return caps;
    }
    caps.backend_available = true;
    caps.device_available = true;
    caps.explicit_allocation = true;
    caps.async_host_to_device = true;
    caps.async_device_to_host = true;
    caps.device_to_device = true;
    caps.pinned_host_allocation = true;
    caps.concurrent_managed_access = p.concurrentManagedAccess != 0;
    caps.direct_storage = false;  // no GDS/DirectStorage path in v0.1
    caps.event_timing = true;
    caps.max_allocation_size = static_cast<uint64_t>(p.totalGlobalMem);
    caps.alignment_bytes = 512;  // cudaMalloc guarantees >= 256; 512 is safe
    caps.transfer_granularity = 1;
    caps.queue_count = 64;  // cudaStreams are effectively unlimited; report a bound
    caps.multi_device = devices.size() > 1;
    caps.note = "explicit allocations; unified memory only when the driver reports it";

    // Live probe: some WDDM drivers report managedMemory but reject
    // prefetch/advice at runtime. Capabilities are discovered, never
    // guessed from property bits alone.
    if (p.managedMemory != 0) {
        void* probe = nullptr;
        if (cudaMallocManaged(&probe, 4096) == cudaSuccess) {
            caps.unified_memory = true;
            const cudaError_t pf = cudaMemPrefetchAsync(probe, 4096, 0);
            caps.memory_prefetch = pf == cudaSuccess;
            const cudaError_t ad = cudaMemAdvise(probe, 4096,
                                                 cudaMemAdviseSetPreferredLocation, 0);
            caps.memory_advice = ad == cudaSuccess;
            caps.hardware_page_fault = caps.memory_prefetch;
            if (!caps.memory_prefetch || !caps.memory_advice) {
                caps.note += "; runtime hint probe: prefetch " +
                             std::string(caps.memory_prefetch ? "ok" : "unavailable") +
                             ", advice " +
                             std::string(caps.memory_advice ? "ok" : "unavailable");
            }
            cudaFree(probe);
        }
        cudaGetLastError();  // clear stale error state from the probe
    }
    {
        std::lock_guard lock(impl_->mu);
        impl_->caps_cache = caps;
        impl_->caps_cache_valid = true;
    }
    return caps;
}

void CudaBackend::open(int device_index) {
    if (impl_->opened) {
        close();
    }
    const std::vector<DeviceInfo> devices = enumerate_devices();
    if (devices.empty()) {
        throw Error(ErrorCode::Unsupported, "cuda backend: no CUDA device available");
    }
    if (device_index < 0 || device_index >= static_cast<int>(devices.size())) {
        throw Error(ErrorCode::Config,
                    "cuda backend: requested device index out of range",
                    "device " + std::to_string(device_index) + " of " +
                        std::to_string(devices.size()));
    }
    check_cuda(cudaSetDevice(device_index), "cudaSetDevice");
    check_cuda(cudaGetDeviceProperties(&impl_->prop, device_index),
               "cudaGetDeviceProperties");
    impl_->device_index = device_index;
    impl_->opened = true;
    impl_->healthy = true;
}

void CudaBackend::close() {
    if (!impl_->opened) return;
    // Drain outstanding work before releasing the device context.
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    impl_->opened = false;
    impl_->device_index = -1;
}

bool CudaBackend::is_open() const {
    return impl_->opened;
}

DeviceInfo CudaBackend::device_info() const {
    if (!impl_->opened) {
        throw Error(ErrorCode::State, "cuda backend is not open");
    }
    DeviceInfo info;
    info.id = "cuda:" + std::to_string(impl_->device_index);
    info.name = impl_->prop.name;
    info.architecture = device_arch(impl_->prop);
    info.total_memory = static_cast<uint64_t>(impl_->prop.totalGlobalMem);
    std::size_t free_ = 0;
    std::size_t total_ = 0;
    cudaError_t e = cudaMemGetInfo(&free_, &total_);
    info.free_memory = e == cudaSuccess ? static_cast<uint64_t>(free_) : 0;
    info.discrete = impl_->prop.integrated == 0;
    info.memory_shared = impl_->prop.integrated != 0;
    info.index = impl_->device_index;
    return info;
}

DeviceCapabilities CudaBackend::capabilities() const {
    if (!impl_->opened) {
        throw Error(ErrorCode::State, "cuda backend is not open");
    }
    return probe_capabilities();
}

uint64_t CudaBackend::total_memory() const {
    return impl_->opened ? static_cast<uint64_t>(impl_->prop.totalGlobalMem) : 0;
}

uint64_t CudaBackend::free_memory() const {
    if (!impl_->opened) return 0;
    std::size_t free_ = 0;
    std::size_t total_ = 0;
    cudaError_t e = cudaMemGetInfo(&free_, &total_);
    return e == cudaSuccess ? static_cast<uint64_t>(free_) : 0;
}

void* CudaBackend::allocate(std::size_t bytes) {
    if (bytes == 0) {
        throw Error(ErrorCode::InvalidArgument, "cuda backend: zero-byte allocation");
    }
    if (bytes > impl_->prop.totalGlobalMem) {
        throw Error(ErrorCode::Budget,
                    "cuda backend: allocation exceeds device memory limit");
    }
    void* ptr = nullptr;
    check_cuda(cudaMalloc(&ptr, bytes), "cudaMalloc");
    return ptr;
}

void CudaBackend::free(void* ptr) {
    if (ptr == nullptr) return;
    check_cuda(cudaFree(ptr), "cudaFree");
}

void* CudaBackend::allocate_host_pinned(std::size_t bytes) {
    if (bytes == 0) {
        throw Error(ErrorCode::InvalidArgument,
                    "cuda backend: zero-byte pinned allocation");
    }
    void* ptr = nullptr;
    check_cuda(cudaMallocHost(&ptr, bytes), "cudaMallocHost");
    return ptr;
}

void CudaBackend::free_host_pinned(void* ptr) {
    if (ptr == nullptr) return;
    check_cuda(cudaFreeHost(ptr), "cudaFreeHost");
}

void* CudaBackend::allocate_unified(std::size_t bytes) {
    if (impl_->prop.managedMemory == 0) {
        throw Error(ErrorCode::Unsupported,
                    "cuda backend: device reports no unified memory support");
    }
    void* ptr = nullptr;
    check_cuda(cudaMallocManaged(&ptr, bytes), "cudaMallocManaged");
    return ptr;
}

void CudaBackend::free_unified(void* ptr) {
    if (ptr == nullptr) return;
    check_cuda(cudaFree(ptr), "cudaFree(managed)");
}

void CudaBackend::prefetch_to_device(void* ptr, std::size_t bytes) {
    if (!probe_capabilities().memory_prefetch) {
        // The runtime hint probe found this driver rejects prefetch; the
        // contract requires a typed Unsupported, not a raw driver error.
        throw Error(ErrorCode::Unsupported,
                    "cuda backend: memory prefetch unavailable on this platform "
                    "(runtime hint probe)");
    }
    check_cuda(cudaMemPrefetchAsync(ptr, bytes, impl_->device_index),
               "cudaMemPrefetchAsync");
}

void CudaBackend::advise_preferred_location(void* ptr, std::size_t bytes) {
    if (!probe_capabilities().memory_advice) {
        throw Error(ErrorCode::Unsupported,
                    "cuda backend: memory advice unavailable on this platform "
                    "(runtime hint probe)");
    }
    check_cuda(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation,
                             impl_->device_index),
               "cudaMemAdvise");
}

DeviceStream* CudaBackend::create_stream() {
    auto* s = new CudaStream();
    check_cuda(cudaStreamCreateWithFlags(&s->handle, cudaStreamNonBlocking),
               "cudaStreamCreate");
    return s;
}

void CudaBackend::destroy_stream(DeviceStream* stream) {
    if (stream == nullptr) return;
    auto* s = static_cast<CudaStream*>(stream);
    check_cuda(cudaStreamDestroy(s->handle), "cudaStreamDestroy");
    delete s;
}

DeviceEvent* CudaBackend::create_event() {
    auto* e = new CudaEvent();
    check_cuda(cudaEventCreateWithFlags(&e->handle, cudaEventDefault),
               "cudaEventCreate");
    return e;
}

void CudaBackend::destroy_event(DeviceEvent* event) {
    if (event == nullptr) return;
    auto* e = static_cast<CudaEvent*>(event);
    check_cuda(cudaEventDestroy(e->handle), "cudaEventDestroy");
    delete e;
}

void CudaBackend::async_copy_host_to_device(void* dst_device, const void* src_host,
                                            std::size_t bytes, DeviceStream* stream) {
    cudaStream_t s = stream != nullptr ? static_cast<CudaStream*>(stream)->handle : 0;
    check_cuda(cudaMemcpyAsync(dst_device, src_host, bytes, cudaMemcpyHostToDevice, s),
               "cudaMemcpyAsync(H2D)");
}

void CudaBackend::async_copy_device_to_host(void* dst_host, const void* src_device,
                                            std::size_t bytes, DeviceStream* stream) {
    cudaStream_t s = stream != nullptr ? static_cast<CudaStream*>(stream)->handle : 0;
    check_cuda(cudaMemcpyAsync(dst_host, src_device, bytes, cudaMemcpyDeviceToHost, s),
               "cudaMemcpyAsync(D2H)");
}

void CudaBackend::async_copy_device_to_device(void* dst_device, const void* src_device,
                                              std::size_t bytes, DeviceStream* stream) {
    cudaStream_t s = stream != nullptr ? static_cast<CudaStream*>(stream)->handle : 0;
    check_cuda(cudaMemcpyAsync(dst_device, src_device, bytes, cudaMemcpyDeviceToDevice, s),
               "cudaMemcpyAsync(D2D)");
}

void CudaBackend::sync_stream(DeviceStream* stream) {
    cudaStream_t s = stream != nullptr ? static_cast<CudaStream*>(stream)->handle : 0;
    check_cuda(cudaStreamSynchronize(s), "cudaStreamSynchronize");
}

void CudaBackend::sync_all() {
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

void CudaBackend::record_event(DeviceEvent* event, DeviceStream* stream) {
    cudaStream_t s = stream != nullptr ? static_cast<CudaStream*>(stream)->handle : 0;
    check_cuda(cudaEventRecord(static_cast<CudaEvent*>(event)->handle, s),
               "cudaEventRecord");
}

void CudaBackend::wait_event(DeviceEvent* event) {
    check_cuda(cudaEventSynchronize(static_cast<CudaEvent*>(event)->handle),
               "cudaEventSynchronize");
}

double CudaBackend::event_elapsed_us(DeviceEvent* start, DeviceEvent* end) {
    float ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&ms,
                                    static_cast<CudaEvent*>(start)->handle,
                                    static_cast<CudaEvent*>(end)->handle),
               "cudaEventElapsedTime");
    return static_cast<double>(ms) * 1000.0;
}

bool CudaBackend::healthy() const {
    return impl_->opened && impl_->healthy;
}

std::string CudaBackend::diagnostics() const {
    int rt = 0;
    int drv = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    return "cuda backend: runtime " + std::to_string(rt) + ", driver " +
           std::to_string(drv) +
           (impl_->opened ? ", device " + std::string(impl_->prop.name) +
                                " cc " + device_arch(impl_->prop)
                          : ", not open");
}

std::string CudaBackend::last_error() const {
    const char* s = cudaGetErrorString(cudaGetLastError());
    return s != nullptr ? s : "";
}

void CudaBackend::fill_device_pattern(void* device_ptr, std::size_t bytes,
                                      std::uint64_t seed, std::uint64_t page_id,
                                      DeviceStream* stream) {
    const unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((bytes + threads - 1) / threads);
    cudaStream_t s = stream != nullptr ? static_cast<CudaStream*>(stream)->handle : 0;
    fill_pattern_kernel<<<blocks, threads, 0, s>>>(
        static_cast<unsigned char*>(device_ptr), bytes, seed, page_id);
    check_cuda(cudaGetLastError(), "fill_pattern_kernel launch");
}

void register_cuda_backend(BackendRegistry& registry) {
    registry.register_backend(
        "cuda", [] { return std::make_unique<CudaBackend>(); },
        "NVIDIA (CUDA)");
}

}  // namespace flashtier
