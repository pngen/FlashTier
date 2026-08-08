// NVIDIA CUDA backend implementing the vendor-neutral DeviceBackend
// contract. Compiled with nvcc only when FLASHTIER_ENABLE_CUDA=ON.
//
// Every CUDA call is checked and mapped to typed FlashTier errors; no
// failure is silently ignored. Device capability fields come from
// cudaDeviceProp discovery, never from product names.

#include "flashtier/backends/cuda_backend.hpp"

#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
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

class DeviceRestore final {
public:
    explicit DeviceRestore(int device_index) {
        check_cuda(cudaGetDevice(&previous_), "cudaGetDevice");
        if (previous_ != device_index) {
            check_cuda(cudaSetDevice(device_index), "cudaSetDevice");
            restore_ = true;
        }
    }

    ~DeviceRestore() {
        if (restore_) {
            (void)cudaSetDevice(previous_);
        }
    }

    DeviceRestore(const DeviceRestore&) = delete;
    DeviceRestore& operator=(const DeviceRestore&) = delete;

private:
    int previous_ = 0;
    bool restore_ = false;
};

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
    mutable std::shared_mutex lifecycle_mu;
    int device_index = -1;
    cudaDeviceProp prop{};
    bool opened = false;

    // Diagnostics: last error string (thread-agnostic snapshot).
    mutable std::mutex mu;
    std::string last_error;
    bool healthy = true;

    // Cached capability probes, keyed by the actual CUDA device index.
    std::unordered_map<int, DeviceCapabilities> caps_cache;

    std::unordered_map<void*, std::size_t> device_allocations;
    std::unordered_map<void*, std::size_t> unified_allocations;
    std::unordered_map<void*, std::size_t> host_allocations;
    std::unordered_set<DeviceStream*> streams;
    std::unordered_set<DeviceEvent*> events;

    int require_open() const {
        if (!opened) {
            throw Error(ErrorCode::State, "cuda backend is not open");
        }
        return device_index;
    }

    bool owns_range(const std::unordered_map<void*, std::size_t>& allocations,
                    const void* ptr, std::size_t bytes) const {
        if (ptr == nullptr) return false;
        const auto address = reinterpret_cast<std::uintptr_t>(ptr);
        for (const auto& [base_ptr, allocation_bytes] : allocations) {
            const auto base = reinterpret_cast<std::uintptr_t>(base_ptr);
            if (address < base) continue;
            const std::size_t offset = static_cast<std::size_t>(address - base);
            if (offset <= allocation_bytes && bytes <= allocation_bytes - offset) {
                return true;
            }
        }
        return false;
    }

    bool owns_device_range(const void* ptr, std::size_t bytes) const {
        return owns_range(device_allocations, ptr, bytes) ||
               owns_range(unified_allocations, ptr, bytes);
    }

    bool owns_unified_range(const void* ptr, std::size_t bytes) const {
        return owns_range(unified_allocations, ptr, bytes);
    }

    void check(cudaError_t error, const char* operation) {
        if (error == cudaSuccess) return;
        std::string detail = cudaGetErrorString(error);
        if (error == cudaErrorDevicesUnavailable ||
            error == cudaErrorDeviceUninitialized) {
            detail += " (device unavailable/lost)";
        }
        const std::string message =
            std::string(operation) + " failed: " + detail;
        {
            std::lock_guard lock(mu);
            last_error = message;
            healthy = false;
        }
        throw Error(ErrorCode::Cuda, message,
                    std::to_string(static_cast<int>(error)));
    }
};

CudaBackend::CudaBackend() : impl_(std::make_unique<Impl>()) {}

CudaBackend::~CudaBackend() {
    try {
        close();
    } catch (...) {
    }
}

std::vector<DeviceInfo> CudaBackend::enumerate_devices() const {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e == cudaErrorNoDevice) return {};
    impl_->check(e, "cudaGetDeviceCount");
    if (count <= 0) return {};
    std::vector<DeviceInfo> out;
    out.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp p{};
        impl_->check(cudaGetDeviceProperties(&p, i), "cudaGetDeviceProperties");
        DeviceInfo info;
        info.id = "cuda:" + std::to_string(i);
        info.name = p.name;
        info.architecture = device_arch(p);
        info.total_memory = static_cast<uint64_t>(p.totalGlobalMem);
        try {
            DeviceRestore restore(i);
            std::size_t free_bytes = 0;
            std::size_t total_bytes = 0;
            if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
                info.free_memory = static_cast<uint64_t>(free_bytes);
            }
        } catch (const Error&) {
            // Free memory is optional discovery data (0 = unknown). Device
            // properties remain valid even when a context cannot be opened.
            info.free_memory = 0;
        }
        // Discovery, not naming: the driver reports integration status.
        info.discrete = p.integrated == 0;
        info.memory_shared = p.integrated != 0;
        info.index = i;
        out.push_back(std::move(info));
    }
    return out;
}

DeviceCapabilities CudaBackend::probe_capabilities() const {
    return probe_capabilities_for_device(0);
}

DeviceCapabilities CudaBackend::probe_capabilities_for_device(
    int device_index) const {
    {
        std::lock_guard lock(impl_->mu);
        const auto cached = impl_->caps_cache.find(device_index);
        if (cached != impl_->caps_cache.end()) return cached->second;
    }
    DeviceCapabilities caps;
    int device_count = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    if (count_error == cudaErrorNoDevice) return caps;
    impl_->check(count_error, "cudaGetDeviceCount");
    if (device_count <= 0) return caps;
    if (device_index < 0 || device_index >= device_count) return caps;

    cudaDeviceProp p{};
    impl_->check(cudaGetDeviceProperties(&p, device_index),
                 "cudaGetDeviceProperties");
    DeviceRestore restore(device_index);
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
    caps.alignment_bytes = 256;  // documented cudaMalloc minimum alignment
    caps.transfer_granularity = 1;
    caps.queue_count = 1;  // default stream guaranteed; additional streams are dynamic
    caps.multi_device = device_count > 1;
    caps.hardware_page_fault = p.pageableMemoryAccess != 0;
    caps.note = "explicit allocations; CUDA streams are created dynamically; "
                "unified memory only when the runtime probe succeeds";

    // Live probe: some WDDM drivers report managedMemory but reject
    // prefetch/advice at runtime. Capabilities are discovered, never
    // guessed from property bits alone.
    if (p.managedMemory != 0) {
        void* probe = nullptr;
        if (cudaMallocManaged(&probe, 4096) == cudaSuccess) {
            caps.unified_memory = true;
            const cudaError_t pf =
                cudaMemPrefetchAsync(probe, 4096, device_index);
            caps.memory_prefetch = pf == cudaSuccess;
            const cudaError_t ad = cudaMemAdvise(probe, 4096,
                                                 cudaMemAdviseSetPreferredLocation,
                                                 device_index);
            caps.memory_advice = ad == cudaSuccess;
            if (!caps.memory_prefetch || !caps.memory_advice) {
                caps.note += "; runtime hint probe: prefetch " +
                             std::string(caps.memory_prefetch ? "ok" : "unavailable") +
                             ", advice " +
                             std::string(caps.memory_advice ? "ok" : "unavailable");
            }
            impl_->check(cudaFree(probe), "cudaFree(capability probe)");
        }
        cudaGetLastError();  // clear stale error state from the probe
    }
    {
        std::lock_guard lock(impl_->mu);
        impl_->caps_cache[device_index] = caps;
    }
    return caps;
}

void CudaBackend::open(int device_index) {
    std::unique_lock lifecycle_lock(impl_->lifecycle_mu);
    int device_count = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    if (count_error == cudaErrorNoDevice) {
        throw Error(ErrorCode::Unsupported, "cuda backend: no CUDA device available");
    }
    impl_->check(count_error, "cudaGetDeviceCount");
    if (device_count <= 0) {
        throw Error(ErrorCode::Unsupported, "cuda backend: no CUDA device available");
    }
    if (device_index < 0 || device_index >= device_count) {
        throw Error(ErrorCode::Config,
                    "cuda backend: requested device index out of range",
                    "device " + std::to_string(device_index) + " of " +
                        std::to_string(device_count));
    }
    if (impl_->opened) close_locked();
    DeviceRestore restore(device_index);
    impl_->check(cudaGetDeviceProperties(&impl_->prop, device_index),
                 "cudaGetDeviceProperties");
    impl_->device_index = device_index;
    impl_->opened = true;
    {
        std::lock_guard lock(impl_->mu);
        impl_->healthy = true;
        impl_->last_error.clear();
    }
}

void CudaBackend::close() {
    std::unique_lock lifecycle_lock(impl_->lifecycle_mu);
    close_locked();
}

void CudaBackend::close_locked() {
    if (!impl_->opened) return;

    const int device_index = impl_->device_index;
    std::vector<CudaEvent*> events;
    std::vector<CudaStream*> streams;
    std::vector<void*> device_allocations;
    std::vector<void*> unified_allocations;
    std::vector<void*> host_allocations;
    {
        std::lock_guard lock(impl_->mu);
        events.reserve(impl_->events.size());
        for (DeviceEvent* event : impl_->events) {
            events.push_back(static_cast<CudaEvent*>(event));
        }
        streams.reserve(impl_->streams.size());
        for (DeviceStream* stream : impl_->streams) {
            streams.push_back(static_cast<CudaStream*>(stream));
        }
        device_allocations.reserve(impl_->device_allocations.size());
        for (const auto& [ptr, bytes] : impl_->device_allocations) {
            (void)bytes;
            device_allocations.push_back(ptr);
        }
        unified_allocations.reserve(impl_->unified_allocations.size());
        for (const auto& [ptr, bytes] : impl_->unified_allocations) {
            (void)bytes;
            unified_allocations.push_back(ptr);
        }
        host_allocations.reserve(impl_->host_allocations.size());
        for (const auto& [ptr, bytes] : impl_->host_allocations) {
            (void)bytes;
            host_allocations.push_back(ptr);
        }
        impl_->events.clear();
        impl_->streams.clear();
        impl_->device_allocations.clear();
        impl_->unified_allocations.clear();
        impl_->host_allocations.clear();
    }

    std::exception_ptr activation_error;
    std::unique_ptr<DeviceRestore> restore;
    try {
        restore = std::make_unique<DeviceRestore>(device_index);
    } catch (...) {
        activation_error = std::current_exception();
    }

    cudaError_t first_error = cudaSuccess;
    const char* first_operation = nullptr;
    const auto capture = [&](cudaError_t error, const char* operation) {
        if (error != cudaSuccess && first_error == cudaSuccess) {
            first_error = error;
            first_operation = operation;
        }
    };

    if (restore) {
        capture(cudaDeviceSynchronize(), "cudaDeviceSynchronize(close)");
        for (CudaEvent* event : events) {
            capture(cudaEventDestroy(event->handle), "cudaEventDestroy(close)");
        }
        for (CudaStream* stream : streams) {
            capture(cudaStreamDestroy(stream->handle), "cudaStreamDestroy(close)");
        }
        for (void* ptr : device_allocations) {
            capture(cudaFree(ptr), "cudaFree(close)");
        }
        for (void* ptr : unified_allocations) {
            capture(cudaFree(ptr), "cudaFree(managed close)");
        }
    }
    for (void* ptr : host_allocations) {
        capture(cudaFreeHost(ptr), "cudaFreeHost(close)");
    }
    for (CudaEvent* event : events) delete event;
    for (CudaStream* stream : streams) delete stream;

    impl_->opened = false;
    impl_->device_index = -1;
    if (activation_error) std::rethrow_exception(activation_error);
    impl_->check(first_error,
                 first_operation != nullptr ? first_operation : "cuda close");
}

bool CudaBackend::is_open() const {
    std::shared_lock lifecycle_lock(impl_->lifecycle_mu);
    return impl_->opened;
}

DeviceInfo CudaBackend::device_info() const {
    std::shared_lock lifecycle_lock(impl_->lifecycle_mu);
    const int device_index = impl_->require_open();
    DeviceRestore restore(device_index);
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
    std::shared_lock lifecycle_lock(impl_->lifecycle_mu);
    return probe_capabilities_for_device(impl_->require_open());
}

uint64_t CudaBackend::total_memory() const {
    std::shared_lock lifecycle_lock(impl_->lifecycle_mu);
    return impl_->opened ? static_cast<uint64_t>(impl_->prop.totalGlobalMem) : 0;
}

uint64_t CudaBackend::free_memory() const {
    std::shared_lock lifecycle_lock(impl_->lifecycle_mu);
    if (!impl_->opened) return 0;
    DeviceRestore restore(impl_->device_index);
    std::size_t free_ = 0;
    std::size_t total_ = 0;
    cudaError_t e = cudaMemGetInfo(&free_, &total_);
    return e == cudaSuccess ? static_cast<uint64_t>(free_) : 0;
}

void* CudaBackend::allocate(std::size_t bytes) {
    std::shared_lock lifecycle_lock(impl_->lifecycle_mu);
    const int device_index = impl_->require_open();
    if (bytes == 0) {
        throw Error(ErrorCode::InvalidArgument, "cuda backend: zero-byte allocation");
    }
    if (bytes > impl_->prop.totalGlobalMem) {
        throw Error(ErrorCode::Budget,
                    "cuda backend: allocation exceeds device memory limit");
    }
    DeviceRestore restore(device_index);
    void* ptr = nullptr;
    impl_->check(cudaMalloc(&ptr, bytes), "cudaMalloc");
    try {
        std::lock_guard lock(impl_->mu);
        if (!impl_->device_allocations.emplace(ptr, bytes).second) {
            throw Error(ErrorCode::State,
                        "cuda backend: allocator returned an already-live pointer");
        }
    } catch (...) {
        (void)cudaFree(ptr);
        throw;
    }
    return ptr;
}

void CudaBackend::free(void* ptr) {
    if (ptr == nullptr) return;
    std::shared_lock lifecycle_lock(impl_->lifecycle_mu);
    const int device_index = impl_->require_open();
    DeviceRestore restore(device_index);
    cudaError_t result = cudaSuccess;
    {
        std::lock_guard lock(impl_->mu);
        const auto allocation = impl_->device_allocations.find(ptr);
        if (allocation == impl_->device_allocations.end()) {
            throw Error(ErrorCode::State,
                        "cuda backend: free of unknown device pointer");
        }
        result = cudaFree(ptr);
        if (result == cudaSuccess) impl_->device_allocations.erase(allocation);
    }
    impl_->check(result, "cudaFree");
}

void* CudaBackend::allocate_host_pinned(std::size_t bytes) {
    std::shared_lock lifecycle_lock(impl_->lifecycle_mu);
    impl_->require_open();
    if (bytes == 0) {
        throw Error(ErrorCode::InvalidArgument,
                    "cuda backend: zero-byte pinned allocation");
    }
    void* ptr = nullptr;
    impl_->check(cudaMallocHost(&ptr, bytes), "cudaMallocHost");
    try {
        std::lock_guard lock(impl_->mu);
        if (!impl_->host_allocations.emplace(ptr, bytes).second) {
            throw Error(ErrorCode::State,
                        "cuda backend: pinned allocator returned an already-live pointer");
        }
    } catch (...) {
        (void)cudaFreeHost(ptr);
        throw;
    }
    return ptr;
}

void CudaBackend::free_host_pinned(void* ptr) {
    if (ptr == nullptr) return;
    std::shared_lock lifecycle_lock(impl_->lifecycle_mu);
    impl_->require_open();
    cudaError_t result = cudaSuccess;
    {
        std::lock_guard lock(impl_->mu);
        const auto allocation = impl_->host_allocations.find(ptr);
        if (allocation == impl_->host_allocations.end()) {
            throw Error(ErrorCode::State,
                        "cuda backend: free of unknown pinned pointer");
        }
        result = cudaFreeHost(ptr);
        if (result == cudaSuccess) impl_->host_allocations.erase(allocation);
    }
    impl_->check(result, "cudaFreeHost");
}

void* CudaBackend::allocate_unified(std::size_t bytes) {
    std::shared_lock lifecycle_lock(impl_->lifecycle_mu);
    const int device_index = impl_->require_open();
    if (bytes == 0) {
        throw Error(ErrorCode::InvalidArgument,
                    "cuda backend: zero-byte unified allocation");
    }
    if (bytes > impl_->prop.totalGlobalMem) {
        throw Error(ErrorCode::Budget,
                    "cuda backend: unified allocation exceeds device memory limit");
    }
    if (impl_->prop.managedMemory == 0) {
        throw Error(ErrorCode::Unsupported,
                    "cuda backend: device reports no unified memory support");
    }
    if (!probe_capabilities_for_device(device_index).unified_memory) {
        throw Error(ErrorCode::Unsupported,
                    "cuda backend: unified memory failed the runtime capability probe");
    }
    DeviceRestore restore(device_index);
    void* ptr = nullptr;
    impl_->check(cudaMallocManaged(&ptr, bytes), "cudaMallocManaged");
    try {
        std::lock_guard lock(impl_->mu);
        if (!impl_->unified_allocations.emplace(ptr, bytes).second) {
            throw Error(ErrorCode::State,
                        "cuda backend: managed allocator returned an already-live pointer");
        }
    } catch (...) {
        (void)cudaFree(ptr);
        throw;
    }
    return ptr;
}

void CudaBackend::free_unified(void* ptr) {
    if (ptr == nullptr) return;
    std::shared_lock lifecycle_lock(impl_->lifecycle_mu);
    const int device_index = impl_->require_open();
    DeviceRestore restore(device_index);
    cudaError_t result = cudaSuccess;
    {
        std::lock_guard lock(impl_->mu);
        const auto allocation = impl_->unified_allocations.find(ptr);
        if (allocation == impl_->unified_allocations.end()) {
            throw Error(ErrorCode::State,
                        "cuda backend: free of unknown managed pointer");
        }
        result = cudaFree(ptr);
        if (result == cudaSuccess) impl_->unified_allocations.erase(allocation);
    }
    impl_->check(result, "cudaFree(managed)");
}

void CudaBackend::prefetch_to_device(void* ptr, std::size_t bytes) {
    std::shared_lock lifecycle_lock(impl_->lifecycle_mu);
    const int device_index = impl_->require_open();
    if (!probe_capabilities_for_device(device_index).memory_prefetch) {
        // The runtime hint probe found this driver rejects prefetch; the
        // contract requires a typed Unsupported, not a raw driver error.
        throw Error(ErrorCode::Unsupported,
                    "cuda backend: memory prefetch unavailable on this platform "
                    "(runtime hint probe)");
    }
    if (ptr == nullptr || bytes == 0) {
        throw Error(ErrorCode::InvalidArgument,
                    "cuda backend: invalid managed-memory prefetch range");
    }
    {
        std::lock_guard lock(impl_->mu);
        if (!impl_->owns_unified_range(ptr, bytes)) {
            if (impl_->owns_unified_range(ptr, 0)) {
                throw Error(ErrorCode::InvalidArgument,
                            "cuda backend: managed-memory prefetch exceeds allocation");
            }
            throw Error(ErrorCode::State,
                        "cuda backend: prefetch of unknown managed pointer");
        }
    }
    DeviceRestore restore(device_index);
    impl_->check(cudaMemPrefetchAsync(ptr, bytes, device_index),
                 "cudaMemPrefetchAsync");
}

void CudaBackend::advise_preferred_location(void* ptr, std::size_t bytes) {
    std::shared_lock lifecycle_lock(impl_->lifecycle_mu);
    const int device_index = impl_->require_open();
    if (!probe_capabilities_for_device(device_index).memory_advice) {
        throw Error(ErrorCode::Unsupported,
                    "cuda backend: memory advice unavailable on this platform "
                    "(runtime hint probe)");
    }
    if (ptr == nullptr || bytes == 0) {
        throw Error(ErrorCode::InvalidArgument,
                    "cuda backend: invalid managed-memory advice range");
    }
    {
        std::lock_guard lock(impl_->mu);
        if (!impl_->owns_unified_range(ptr, bytes)) {
            if (impl_->owns_unified_range(ptr, 0)) {
                throw Error(ErrorCode::InvalidArgument,
                            "cuda backend: managed-memory advice exceeds allocation");
            }
            throw Error(ErrorCode::State,
                        "cuda backend: advice for unknown managed pointer");
        }
    }
    DeviceRestore restore(device_index);
    impl_->check(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation,
                               device_index),
                 "cudaMemAdvise");
}

DeviceStream* CudaBackend::create_stream() {
    std::shared_lock lifecycle_lock(impl_->lifecycle_mu);
    const int device_index = impl_->require_open();
    DeviceRestore restore(device_index);
    std::unique_ptr<CudaStream> stream = std::make_unique<CudaStream>();
    impl_->check(
        cudaStreamCreateWithFlags(&stream->handle, cudaStreamNonBlocking),
        "cudaStreamCreate");
    try {
        std::lock_guard lock(impl_->mu);
        impl_->streams.insert(stream.get());
    } catch (...) {
        (void)cudaStreamDestroy(stream->handle);
        throw;
    }
    return stream.release();
}

void CudaBackend::destroy_stream(DeviceStream* stream) {
    if (stream == nullptr) return;
    std::shared_lock lifecycle_lock(impl_->lifecycle_mu);
    const int device_index = impl_->require_open();
    DeviceRestore restore(device_index);
    auto* s = static_cast<CudaStream*>(stream);
    cudaError_t result = cudaSuccess;
    {
        std::lock_guard lock(impl_->mu);
        const auto owned = impl_->streams.find(stream);
        if (owned == impl_->streams.end()) {
            throw Error(ErrorCode::State,
                        "cuda backend: destroy of unknown stream");
        }
        result = cudaStreamDestroy(s->handle);
        impl_->streams.erase(owned);
    }
    delete s;
    impl_->check(result, "cudaStreamDestroy");
}

DeviceEvent* CudaBackend::create_event() {
    std::shared_lock lifecycle_lock(impl_->lifecycle_mu);
    const int device_index = impl_->require_open();
    DeviceRestore restore(device_index);
    std::unique_ptr<CudaEvent> event = std::make_unique<CudaEvent>();
    impl_->check(cudaEventCreateWithFlags(&event->handle, cudaEventDefault),
                 "cudaEventCreate");
    try {
        std::lock_guard lock(impl_->mu);
        impl_->events.insert(event.get());
    } catch (...) {
        (void)cudaEventDestroy(event->handle);
        throw;
    }
    return event.release();
}

void CudaBackend::destroy_event(DeviceEvent* event) {
    if (event == nullptr) return;
    std::shared_lock lifecycle_lock(impl_->lifecycle_mu);
    const int device_index = impl_->require_open();
    DeviceRestore restore(device_index);
    auto* e = static_cast<CudaEvent*>(event);
    cudaError_t result = cudaSuccess;
    {
        std::lock_guard lock(impl_->mu);
        const auto owned = impl_->events.find(event);
        if (owned == impl_->events.end()) {
            throw Error(ErrorCode::State,
                        "cuda backend: destroy of unknown event");
        }
        result = cudaEventDestroy(e->handle);
        impl_->events.erase(owned);
    }
    delete e;
    impl_->check(result, "cudaEventDestroy");
}

void CudaBackend::async_copy_host_to_device(void* dst_device, const void* src_host,
                                            std::size_t bytes, DeviceStream* stream) {
    std::shared_lock lifecycle_lock(impl_->lifecycle_mu);
    const int device_index = impl_->require_open();
    if (bytes != 0 && (dst_device == nullptr || src_host == nullptr)) {
        throw Error(ErrorCode::InvalidArgument,
                    "cuda backend: invalid host-to-device copy range");
    }
    DeviceRestore restore(device_index);
    cudaError_t result = cudaSuccess;
    {
        std::lock_guard lock(impl_->mu);
        if (!impl_->owns_device_range(dst_device, bytes)) {
            if (impl_->owns_device_range(dst_device, 0)) {
                throw Error(ErrorCode::InvalidArgument,
                            "cuda backend: copy destination range exceeds allocation");
            }
            throw Error(ErrorCode::State,
                        "cuda backend: copy destination is not owned");
        }
        if (stream != nullptr && impl_->streams.count(stream) == 0) {
            throw Error(ErrorCode::State, "cuda backend: unknown stream");
        }
        const cudaStream_t handle =
            stream != nullptr ? static_cast<CudaStream*>(stream)->handle : 0;
        result = cudaMemcpyAsync(dst_device, src_host, bytes,
                                 cudaMemcpyHostToDevice, handle);
    }
    impl_->check(result, "cudaMemcpyAsync(H2D)");
}

void CudaBackend::async_copy_device_to_host(void* dst_host, const void* src_device,
                                            std::size_t bytes, DeviceStream* stream) {
    std::shared_lock lifecycle_lock(impl_->lifecycle_mu);
    const int device_index = impl_->require_open();
    if (bytes != 0 && (dst_host == nullptr || src_device == nullptr)) {
        throw Error(ErrorCode::InvalidArgument,
                    "cuda backend: invalid device-to-host copy range");
    }
    DeviceRestore restore(device_index);
    cudaError_t result = cudaSuccess;
    {
        std::lock_guard lock(impl_->mu);
        if (!impl_->owns_device_range(src_device, bytes)) {
            if (impl_->owns_device_range(src_device, 0)) {
                throw Error(ErrorCode::InvalidArgument,
                            "cuda backend: copy source range exceeds allocation");
            }
            throw Error(ErrorCode::State,
                        "cuda backend: copy source is not owned");
        }
        if (stream != nullptr && impl_->streams.count(stream) == 0) {
            throw Error(ErrorCode::State, "cuda backend: unknown stream");
        }
        const cudaStream_t handle =
            stream != nullptr ? static_cast<CudaStream*>(stream)->handle : 0;
        result = cudaMemcpyAsync(dst_host, src_device, bytes,
                                 cudaMemcpyDeviceToHost, handle);
    }
    impl_->check(result, "cudaMemcpyAsync(D2H)");
}

void CudaBackend::async_copy_device_to_device(void* dst_device, const void* src_device,
                                              std::size_t bytes, DeviceStream* stream) {
    std::shared_lock lifecycle_lock(impl_->lifecycle_mu);
    const int device_index = impl_->require_open();
    if (bytes != 0 && (dst_device == nullptr || src_device == nullptr)) {
        throw Error(ErrorCode::InvalidArgument,
                    "cuda backend: invalid device-to-device copy range");
    }
    DeviceRestore restore(device_index);
    cudaError_t result = cudaSuccess;
    {
        std::lock_guard lock(impl_->mu);
        const bool destination_owned =
            impl_->owns_device_range(dst_device, bytes);
        const bool source_owned = impl_->owns_device_range(src_device, bytes);
        if (!destination_owned || !source_owned) {
            if ((!destination_owned &&
                 impl_->owns_device_range(dst_device, 0)) ||
                (!source_owned && impl_->owns_device_range(src_device, 0))) {
                throw Error(ErrorCode::InvalidArgument,
                            "cuda backend: device copy range exceeds allocation");
            }
            throw Error(ErrorCode::State,
                        "cuda backend: device copy uses an unknown pointer");
        }
        if (stream != nullptr && impl_->streams.count(stream) == 0) {
            throw Error(ErrorCode::State, "cuda backend: unknown stream");
        }
        const cudaStream_t handle =
            stream != nullptr ? static_cast<CudaStream*>(stream)->handle : 0;
        result = cudaMemcpyAsync(dst_device, src_device, bytes,
                                 cudaMemcpyDeviceToDevice, handle);
    }
    impl_->check(result, "cudaMemcpyAsync(D2D)");
}

void CudaBackend::sync_stream(DeviceStream* stream) {
    std::shared_lock lifecycle_lock(impl_->lifecycle_mu);
    const int device_index = impl_->require_open();
    DeviceRestore restore(device_index);
    cudaError_t result = cudaSuccess;
    {
        std::lock_guard lock(impl_->mu);
        if (stream != nullptr && impl_->streams.count(stream) == 0) {
            throw Error(ErrorCode::State, "cuda backend: unknown stream");
        }
        const cudaStream_t handle =
            stream != nullptr ? static_cast<CudaStream*>(stream)->handle : 0;
        result = cudaStreamSynchronize(handle);
    }
    impl_->check(result, "cudaStreamSynchronize");
}

void CudaBackend::sync_all() {
    std::shared_lock lifecycle_lock(impl_->lifecycle_mu);
    const int device_index = impl_->require_open();
    DeviceRestore restore(device_index);
    impl_->check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

void CudaBackend::record_event(DeviceEvent* event, DeviceStream* stream) {
    std::shared_lock lifecycle_lock(impl_->lifecycle_mu);
    const int device_index = impl_->require_open();
    DeviceRestore restore(device_index);
    cudaError_t result = cudaSuccess;
    {
        std::lock_guard lock(impl_->mu);
        if (event == nullptr || impl_->events.count(event) == 0) {
            throw Error(ErrorCode::State, "cuda backend: unknown event");
        }
        if (stream != nullptr && impl_->streams.count(stream) == 0) {
            throw Error(ErrorCode::State, "cuda backend: unknown stream");
        }
        const cudaStream_t handle =
            stream != nullptr ? static_cast<CudaStream*>(stream)->handle : 0;
        result = cudaEventRecord(static_cast<CudaEvent*>(event)->handle, handle);
    }
    impl_->check(result, "cudaEventRecord");
}

void CudaBackend::wait_event(DeviceEvent* event) {
    std::shared_lock lifecycle_lock(impl_->lifecycle_mu);
    const int device_index = impl_->require_open();
    DeviceRestore restore(device_index);
    cudaError_t result = cudaSuccess;
    {
        std::lock_guard lock(impl_->mu);
        if (event == nullptr || impl_->events.count(event) == 0) {
            throw Error(ErrorCode::State, "cuda backend: unknown event");
        }
        result =
            cudaEventSynchronize(static_cast<CudaEvent*>(event)->handle);
    }
    impl_->check(result, "cudaEventSynchronize");
}

double CudaBackend::event_elapsed_us(DeviceEvent* start, DeviceEvent* end) {
    std::shared_lock lifecycle_lock(impl_->lifecycle_mu);
    const int device_index = impl_->require_open();
    DeviceRestore restore(device_index);
    float ms = 0.0f;
    cudaError_t result = cudaSuccess;
    {
        std::lock_guard lock(impl_->mu);
        if (start == nullptr || end == nullptr ||
            impl_->events.count(start) == 0 || impl_->events.count(end) == 0) {
            throw Error(ErrorCode::State, "cuda backend: unknown timing event");
        }
        result = cudaEventElapsedTime(
            &ms, static_cast<CudaEvent*>(start)->handle,
            static_cast<CudaEvent*>(end)->handle);
    }
    impl_->check(result, "cudaEventElapsedTime");
    return static_cast<double>(ms) * 1000.0;
}

bool CudaBackend::healthy() const {
    std::shared_lock lifecycle_lock(impl_->lifecycle_mu);
    std::lock_guard state_lock(impl_->mu);
    return impl_->opened && impl_->healthy;
}

std::string CudaBackend::diagnostics() const {
    std::shared_lock lifecycle_lock(impl_->lifecycle_mu);
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
    std::lock_guard lock(impl_->mu);
    return impl_->last_error;
}

void CudaBackend::fill_device_pattern(void* device_ptr, std::size_t bytes,
                                      std::uint64_t seed, std::uint64_t page_id,
                                      DeviceStream* stream) {
    std::shared_lock lifecycle_lock(impl_->lifecycle_mu);
    const int device_index = impl_->require_open();
    if (device_ptr == nullptr || bytes == 0) {
        throw Error(ErrorCode::InvalidArgument,
                    "cuda backend: invalid device pattern range");
    }
    if (bytes > static_cast<std::size_t>(std::numeric_limits<unsigned>::max()) *
                    256ull) {
        throw Error(ErrorCode::InvalidArgument,
                    "cuda backend: pattern range exceeds kernel launch geometry");
    }
    DeviceRestore restore(device_index);
    const unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((bytes + threads - 1) / threads);
    cudaStream_t handle = 0;
    {
        std::lock_guard lock(impl_->mu);
        if (!impl_->owns_device_range(device_ptr, bytes)) {
            if (impl_->owns_device_range(device_ptr, 0)) {
                throw Error(ErrorCode::InvalidArgument,
                            "cuda backend: pattern range exceeds allocation");
            }
            throw Error(ErrorCode::State,
                        "cuda backend: pattern destination is not owned");
        }
        if (stream != nullptr && impl_->streams.count(stream) == 0) {
            throw Error(ErrorCode::State, "cuda backend: unknown stream");
        }
        handle = stream != nullptr ? static_cast<CudaStream*>(stream)->handle : 0;
    }
    fill_pattern_kernel<<<blocks, threads, 0, handle>>>(
        static_cast<unsigned char*>(device_ptr), bytes, seed, page_id);
    impl_->check(cudaGetLastError(), "fill_pattern_kernel launch");
}

void register_cuda_backend(BackendRegistry& registry) {
    registry.register_backend(
        "cuda", [] { return std::make_unique<CudaBackend>(); },
        "NVIDIA (CUDA)");
}

}  // namespace flashtier
