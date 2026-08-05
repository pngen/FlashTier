// Apple Metal backend (macOS only; compiled only on Apple platforms).
//
// This module is written for the Metal framework and follows the
// vendor-neutral DeviceBackend contract. It was architecture-checked on a
// Windows host and is NOT locally compiled or executed in this pass (no
// macOS build environment). macOS hardware execution is not claimed.
//
// Implementation notes for a macOS maintainer:
//  - MTLDevice is discovered via MTLCreateSystemDefaultDevice (or
//    MTLCopyAllDevices for multi-device Macs).
//  - Device memory is a MTLBuffer with storageModePrivate.
//  - Pinned/shared host memory is a MTLBuffer with storageModeShared
//    (Apple Silicon is unified memory: shared system RAM).
//  - Transfers are blit command encoders submitted on a MTLCommandQueue,
//    synchronized with MTLCommandBuffer waitUntilCompleted (or
//    addCompletedHandler for async completion).
//  - MTLDevice.recommendedMaxWorkingSetSize bounds max_allocation_size.
//  - Apple Silicon reports unified memory; the runtime must not double
//    count shared RAM as separate device + host capacity (handled by the
//    integrated-GPU budget logic in Runtime::start).

#if defined(__APPLE__)

#include "flashtier/backends/metal_backend.hpp"

#include <cstdint>
#include <mutex>
#include <vector>

#include <Metal/Metal.h>
#include <Foundation/Foundation.h>

#include "flashtier/error.hpp"

namespace flashtier {

namespace {

// Metal NSError* -> typed FlashTier error.
void check_metal(bool ok, const char* what, NSError* error = nullptr) {
    if (!ok) {
        const char* detail = error != nullptr
                                  ? [[error localizedDescription] UTF8String]
                                  : "unknown Metal error";
        throw Error(ErrorCode::Device, std::string(what) + " failed: " + detail);
    }
}

}  // namespace

class MetalStream final : public DeviceStream {
public:
    id<MTLCommandQueue> queue = nil;
};

class MetalEvent final : public DeviceEvent {
public:
    id<MTLCommandBuffer> buffer = nil;
};

struct MetalBackend::Impl {
    id<MTLDevice> device = nil;
    int device_index = -1;
    bool opened = false;
    DeviceInfo info;
    DeviceCapabilities caps;
    std::mutex mu;
    std::string last_error;
};

MetalBackend::MetalBackend() : impl_(std::make_unique<Impl>()) {}

MetalBackend::~MetalBackend() {
    close();
}

std::vector<DeviceInfo> MetalBackend::enumerate_devices() const {
    std::vector<DeviceInfo> out;
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    const NSUInteger count = [devices count];
    if (count == 0) {
        return out;
    }
    for (NSUInteger i = 0; i < count; ++i) {
        id<MTLDevice> dev = [devices objectAtIndex:i];
        DeviceInfo info;
        info.id = "metal:" + std::to_string(i);
        info.name = std::string([[dev name] UTF8String]);
        info.architecture = "metal";
        info.total_memory = [dev recommendedMaxWorkingSetSize];
        info.free_memory = info.total_memory;
        // Apple Silicon is unified memory; discrete GPUs are rare and
        // reported through MTLCopyAllDevices ordering/names only. v0.1
        // treats Metal devices as shared-memory until discovery improves.
        info.discrete = false;
        info.memory_shared = true;
        info.index = static_cast<int>(i);
        out.push_back(std::move(info));
    }
    return out;
}

DeviceCapabilities MetalBackend::probe_capabilities() const {
    DeviceCapabilities caps;
    const std::vector<DeviceInfo> devices = enumerate_devices();
    if (devices.empty()) {
        return caps;
    }
    caps.backend_available = true;
    caps.device_available = true;
    caps.explicit_allocation = true;
    caps.async_host_to_device = true;
    caps.async_device_to_host = true;
    caps.device_to_device = true;
    caps.pinned_host_allocation = true;
    caps.unified_memory = true;  // Apple Silicon: shared system memory
    caps.host_coherent_memory = true;
    caps.event_timing = false;  // host-side timing in v0.1
    caps.max_allocation_size = devices[0].total_memory;
    caps.alignment_bytes = 16;  // MTLBuffer minimum alignment
    caps.transfer_granularity = 1;
    caps.queue_count = 1;
    caps.note = "Apple Silicon reports unified memory; integrated-GPU accounting applies";
    return caps;
}

void MetalBackend::open(int device_index) {
    if (impl_->opened) close();
    const std::vector<DeviceInfo> devices = enumerate_devices();
    if (device_index < 0 || device_index >= static_cast<int>(devices.size())) {
        throw Error(ErrorCode::Config, "metal backend: requested device index out of range");
    }
    NSArray<id<MTLDevice>>* all = MTLCopyAllDevices();
    impl_->device = [all objectAtIndex:static_cast<NSUInteger>(device_index)];
    impl_->device_index = device_index;
    impl_->info = devices[static_cast<std::size_t>(device_index)];
    impl_->caps = probe_capabilities();
    impl_->opened = true;
}

void MetalBackend::close() {
    if (!impl_->opened) return;
    // MTLDevice is autoreleased by the framework; no explicit destroy.
    impl_->device = nil;
    impl_->opened = false;
}

bool MetalBackend::is_open() const {
    return impl_->opened;
}

DeviceInfo MetalBackend::device_info() const {
    if (!impl_->opened) {
        throw Error(ErrorCode::State, "metal backend is not open");
    }
    return impl_->info;
}

DeviceCapabilities MetalBackend::capabilities() const {
    if (!impl_->opened) {
        throw Error(ErrorCode::State, "metal backend is not open");
    }
    return impl_->caps;
}

uint64_t MetalBackend::total_memory() const {
    return impl_->opened ? impl_->info.total_memory : 0;
}

uint64_t MetalBackend::free_memory() const {
    return impl_->opened ? impl_->info.total_memory : 0;
}

void* MetalBackend::allocate(std::size_t bytes) {
    if (bytes == 0) {
        throw Error(ErrorCode::InvalidArgument, "metal backend: zero-byte allocation");
    }
    id<MTLBuffer> buffer = [impl_->device newBufferWithLength:bytes
                                                     options:MTLResourceStorageModePrivate];
    check_metal(buffer != nil, "newBufferWithLength(private)");
    return (__bridge_retained void*)buffer;
}

void MetalBackend::free(void* ptr) {
    if (ptr == nullptr) return;
    CFBridgingRelease(ptr);  // releases the MTLBuffer
}

void* MetalBackend::allocate_host_pinned(std::size_t bytes) {
    if (bytes == 0) {
        throw Error(ErrorCode::InvalidArgument,
                    "metal backend: zero-byte host allocation");
    }
    id<MTLBuffer> buffer = [impl_->device newBufferWithLength:bytes
                                                     options:MTLResourceStorageModeShared];
    check_metal(buffer != nil, "newBufferWithLength(shared)");
    return (__bridge_retained void*)buffer;
}

void MetalBackend::free_host_pinned(void* ptr) {
    free(ptr);
}

void* MetalBackend::allocate_unified(std::size_t bytes) {
    return allocate_host_pinned(bytes);  // unified memory == shared buffers on Apple
}

void MetalBackend::free_unified(void* ptr) {
    free(ptr);
}

void MetalBackend::prefetch_to_device(void* ptr, std::size_t bytes) {
    (void)ptr;
    (void)bytes;
    throw Error(ErrorCode::Unsupported,
                "metal backend: prefetch not supported in v0.1 (unified memory)");
}

void MetalBackend::advise_preferred_location(void* ptr, std::size_t bytes) {
    (void)ptr;
    (void)bytes;
    throw Error(ErrorCode::Unsupported,
                "metal backend: memory advice not supported in v0.1");
}

DeviceStream* MetalBackend::create_stream() {
    auto* s = new MetalStream();
    s->queue = [impl_->device newCommandQueue];
    check_metal(s->queue != nil, "newCommandQueue");
    return s;
}

void MetalBackend::destroy_stream(DeviceStream* stream) {
    delete stream;  // MTLCommandQueue is ARC-managed
}

DeviceEvent* MetalBackend::create_event() {
    return new MetalEvent();
}

void MetalBackend::destroy_event(DeviceEvent* event) {
    delete event;
}

namespace {

void submit_blit(MetalBackend::Impl& impl, MetalStream* s, id<MTLBuffer> src,
                 id<MTLBuffer> dst, std::size_t bytes) {
    id<MTLCommandBuffer> buffer = [s->queue commandBuffer];
    check_metal(buffer != nil, "commandBuffer");
    id<MTLBlitCommandEncoder> encoder = [buffer blitCommandEncoder];
    [encoder copyFromBuffer:src
               sourceOffset:0
                   toBuffer:dst
          destinationOffset:0
                      size:bytes];
    [encoder endEncoding];
    [buffer commit];
    [buffer waitUntilCompleted];
}

}  // namespace

void MetalBackend::async_copy_host_to_device(void* dst_device, const void* src_host,
                                             std::size_t bytes, DeviceStream* stream) {
    id<MTLBuffer> dst = (__bridge id<MTLBuffer>)dst_device;
    id<MTLBuffer> src = (__bridge id<MTLBuffer>)src_host;
    submit_blit(*impl_, static_cast<MetalStream*>(stream), src, dst, bytes);
}

void MetalBackend::async_copy_device_to_host(void* dst_host, const void* src_device,
                                             std::size_t bytes, DeviceStream* stream) {
    id<MTLBuffer> dst = (__bridge id<MTLBuffer>)dst_host;
    id<MTLBuffer> src = (__bridge id<MTLBuffer>)src_device;
    submit_blit(*impl_, static_cast<MetalStream*>(stream), src, dst, bytes);
}

void MetalBackend::async_copy_device_to_device(void* dst_device, const void* src_device,
                                               std::size_t bytes, DeviceStream* stream) {
    id<MTLBuffer> dst = (__bridge id<MTLBuffer>)dst_device;
    id<MTLBuffer> src = (__bridge id<MTLBuffer>)src_device;
    submit_blit(*impl_, static_cast<MetalStream*>(stream), src, dst, bytes);
}

void MetalBackend::sync_stream(DeviceStream* stream) {
    (void)stream;  // blits wait until completed
}

void MetalBackend::sync_all() {
    // blits wait until completed
}

void MetalBackend::record_event(DeviceEvent* event, DeviceStream* stream) {
    (void)event;
    (void)stream;
    throw Error(ErrorCode::Unsupported,
                "metal backend: event recording not supported in v0.1 (host timing)");
}

void MetalBackend::wait_event(DeviceEvent* event) {
    (void)event;
    throw Error(ErrorCode::Unsupported,
                "metal backend: event waiting not supported in v0.1");
}

double MetalBackend::event_elapsed_us(DeviceEvent* start, DeviceEvent* end) {
    (void)start;
    (void)end;
    throw Error(ErrorCode::Unsupported,
                "metal backend: event timing not supported in v0.1");
}

bool MetalBackend::healthy() const {
    return impl_->opened;
}

std::string MetalBackend::diagnostics() const {
    return "metal backend: " +
           (impl_->opened ? std::string("device ") + impl_->info.id
                          : "not open") +
           " (not locally validated on this Windows host)";
}

std::string MetalBackend::last_error() const {
    std::lock_guard lock(impl_->mu);
    return impl_->last_error;
}

void register_metal_backend(BackendRegistry& registry) {
    registry.register_backend(
        "metal", [] { return std::make_unique<MetalBackend>(); },
        "Apple (Metal)");
}

}  // namespace flashtier

#endif  // __APPLE__
