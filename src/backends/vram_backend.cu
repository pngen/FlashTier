// CUDA VRAM backend implementation. Compiled with nvcc only when
// FLASHTIER_ENABLE_CUDA=ON.

#include "flashtier/backends/vram_backend.hpp"

#include <algorithm>
#include <vector>

#include "flashtier/error.hpp"

namespace flashtier {

namespace {

void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw Error(ErrorCode::Cuda,
                    std::string(what) + " failed: " + cudaGetErrorString(e));
    }
}

__global__ void fill_pattern_kernel(unsigned char* dst, std::uint64_t bytes,
                                    std::uint64_t seed, std::uint64_t page_id) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= bytes) return;
    // Deterministic per 8-byte block; matches the host-side generator.
    const std::uint64_t block = i / 8;
    std::uint64_t x = seed ^ (page_id * 0x9E3779B97F4A7C15ull) ^ block;
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    x = x ^ (x >> 31);
    dst[i] = static_cast<unsigned char>(x >> ((i % 8) * 8));
}

}  // namespace

VramBackend::VramBackend(int device_id) : device_id_(device_id) {}

VramBackend::~VramBackend() {
    close();
}

void VramBackend::open() {
    int count = 0;
    check_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count <= 0) {
        throw Error(ErrorCode::Unsupported, "no CUDA device available");
    }
    if (device_id_ >= count) {
        throw Error(ErrorCode::Config,
                    "requested device index out of range",
                    "device " + std::to_string(device_id_) + " of " + std::to_string(count));
    }
    check_cuda(cudaSetDevice(device_id_), "cudaSetDevice");
    check_cuda(cudaGetDeviceProperties(&prop_, device_id_),
               "cudaGetDeviceProperties");
    check_cuda(cudaStreamCreateWithFlags(&stream_[0], cudaStreamNonBlocking),
               "cudaStreamCreate(0)");
    check_cuda(cudaStreamCreateWithFlags(&stream_[1], cudaStreamNonBlocking),
               "cudaStreamCreate(1)");
    check_cuda(cudaEventCreateWithFlags(&start_event_, cudaEventDisableTiming),
               "cudaEventCreate(start)");
    check_cuda(cudaEventCreateWithFlags(&end_event_, cudaEventDisableTiming),
               "cudaEventCreate(end)");
    opened_ = true;
}

void VramBackend::close() {
    if (!opened_) return;
    for (auto& s : stream_) {
        if (s) {
            cudaStreamDestroy(s);
            s = nullptr;
        }
    }
    if (start_event_) {
        cudaEventDestroy(start_event_);
        start_event_ = nullptr;
    }
    if (end_event_) {
        cudaEventDestroy(end_event_);
        end_event_ = nullptr;
    }
    opened_ = false;
}

const cudaDeviceProp& VramBackend::prop() const {
    return prop_;
}

uint64_t VramBackend::total_mem_bytes() const {
    return static_cast<uint64_t>(prop_.totalGlobalMem);
}

uint64_t VramBackend::free_mem_bytes() const {
    std::size_t free_ = 0;
    std::size_t total_ = 0;
    cudaError_t e = cudaMemGetInfo(&free_, &total_);
    if (e != cudaSuccess) {
        return 0;  // caller treats 0 as unknown
    }
    return static_cast<uint64_t>(free_);
}

void* VramBackend::alloc(std::size_t bytes) {
    void* ptr = nullptr;
    check_cuda(cudaMalloc(&ptr, bytes), "cudaMalloc");
    return ptr;
}

void VramBackend::free(void* ptr) {
    if (ptr == nullptr) return;
    check_cuda(cudaFree(ptr), "cudaFree");
}

void VramBackend::async_h2d(void* dst_device, const void* src_host, std::size_t bytes, int stream) {
    check_cuda(cudaMemcpyAsync(dst_device, src_host, bytes, cudaMemcpyHostToDevice,
                               stream_[stream]),
               "cudaMemcpyAsync(H2D)");
}

void VramBackend::async_d2h(void* dst_host, const void* src_device, std::size_t bytes, int stream) {
    check_cuda(cudaMemcpyAsync(dst_host, src_device, bytes, cudaMemcpyDeviceToHost,
                               stream_[stream]),
               "cudaMemcpyAsync(D2H)");
}

void VramBackend::sync(int stream) {
    check_cuda(cudaStreamSynchronize(stream_[stream]), "cudaStreamSynchronize");
}

void VramBackend::sync_all() {
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

void VramBackend::record_start(int stream) {
    check_cuda(cudaEventRecord(start_event_, stream_[stream]), "cudaEventRecord(start)");
}

void VramBackend::record_end(int stream) {
    check_cuda(cudaEventRecord(end_event_, stream_[stream]), "cudaEventRecord(end)");
}

double VramBackend::elapsed_us() const {
    float ms = 0.0f;
    check_cuda(cudaEventSynchronize(end_event_), "cudaEventSynchronize");
    check_cuda(cudaEventElapsedTime(&ms, start_event_, end_event_),
               "cudaEventElapsedTime");
    return static_cast<double>(ms) * 1000.0;
}

VramBackend::CopyResult VramBackend::measure_h2d(std::size_t bytes, int iterations) {
    void* d = alloc(bytes);
    std::vector<unsigned char> h(bytes, 0xAB);
    CopyResult best;
    best.bandwidth_gb_s = 0.0;
    for (int i = 0; i < iterations; ++i) {
        record_start(0);
        async_h2d(d, h.data(), bytes, 0);
        record_end(0);
        const double us = elapsed_us();
        if (us > 0.0) {
            const double bw = static_cast<double>(bytes) / 1e9 / (us / 1e6);
            if (best.bandwidth_gb_s < bw) {
                best.bandwidth_gb_s = bw;
                best.duration_us = us;
            }
        }
    }
    free(d);
    return best;
}

VramBackend::CopyResult VramBackend::measure_d2h(std::size_t bytes, int iterations) {
    void* d = alloc(bytes);
    fill_device_pattern(d, bytes, 1, 0, 0);
    sync_all();
    std::vector<unsigned char> h(bytes);
    CopyResult best;
    best.bandwidth_gb_s = 0.0;
    for (int i = 0; i < iterations; ++i) {
        record_start(0);
        async_d2h(h.data(), d, bytes, 0);
        record_end(0);
        const double us = elapsed_us();
        if (us > 0.0) {
            const double bw = static_cast<double>(bytes) / 1e9 / (us / 1e6);
            if (best.bandwidth_gb_s < bw) {
                best.bandwidth_gb_s = bw;
                best.duration_us = us;
            }
        }
    }
    free(d);
    return best;
}

void VramBackend::fill_device_pattern(void* device_ptr, std::size_t bytes,
                                      std::uint64_t seed, std::uint64_t page_id,
                                      int stream) {
    const unsigned threads = 256;
    const unsigned blocks =
        static_cast<unsigned>((bytes + threads - 1) / threads);
    fill_pattern_kernel<<<blocks, threads, 0, stream_[stream]>>>(
        static_cast<unsigned char*>(device_ptr), bytes, seed, page_id);
    check_cuda(cudaGetLastError(), "fill_pattern_kernel launch");
}

bool VramBackend::unified_memory_available() const {
    return prop_.managedMemory != 0;
}

void* VramBackend::alloc_managed(std::size_t bytes) {
    void* ptr = nullptr;
    check_cuda(cudaMallocManaged(&ptr, bytes), "cudaMallocManaged");
    return ptr;
}

void VramBackend::free_managed(void* ptr) {
    if (ptr == nullptr) return;
    check_cuda(cudaFree(ptr), "cudaFree(managed)");
}

void VramBackend::prefetch_managed(void* ptr, std::size_t bytes, int device) {
    check_cuda(cudaMemPrefetchAsync(ptr, bytes, device), "cudaMemPrefetchAsync");
}

void VramBackend::advise_preferred_location(void* ptr, std::size_t bytes, int device) {
    check_cuda(cudaMemAdvise(ptr, bytes, cudaMemAdviseSetPreferredLocation, device),
               "cudaMemAdvise");
}

}  // namespace flashtier
