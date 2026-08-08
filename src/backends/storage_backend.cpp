#include "flashtier/backends/storage_backend.hpp"

#include <cstdio>
#include <limits>

#include "store_common.hpp"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace flashtier {

namespace {

#if defined(_WIN32)
bool utf8_path_to_wide(const std::string& path, std::wstring& out) {
    if (path.empty() || path.find('\0') != std::string::npos ||
        path.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    const int input_size = static_cast<int>(path.size());
    const int wide_size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                               path.data(), input_size,
                                               nullptr, 0);
    if (wide_size <= 0) return false;
    out.resize(static_cast<std::size_t>(wide_size));
    return MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                               path.data(), input_size,
                               out.data(), wide_size) == wide_size;
}
#endif

}  // namespace

bool StorageBackend::allocate_extents(uint64_t extent_count, uint64_t& offset_out) {
    if (extent_count == 0) {
        throw Error(ErrorCode::InvalidArgument, "extent count must be nonzero");
    }
    if (extent_count != 1) {
        throw Error(ErrorCode::Unsupported,
                    "storage backend does not support contiguous multi-extent allocation",
                    std::to_string(extent_count));
    }
    return allocate_extent(offset_out);
}

void StorageBackend::free_extents(uint64_t offset, uint64_t extent_count) {
    if (extent_count == 0) {
        throw Error(ErrorCode::InvalidArgument, "extent count must be nonzero");
    }
    if (extent_count != 1) {
        throw Error(ErrorCode::Unsupported,
                    "storage backend does not support contiguous multi-extent release",
                    std::to_string(extent_count));
    }
    free_extent(offset);
}

bool store_header_valid(const std::string& path, uint64_t page_size,
                         uint64_t capacity_bytes) {
#if defined(_WIN32)
    std::wstring wide_path;
    if (!utf8_path_to_wide(path, wide_path)) return false;
    std::FILE* f = _wfopen(wide_path.c_str(), L"rb");
#else
    std::FILE* f = std::fopen(path.c_str(), "rb");
#endif
    if (f == nullptr) {
        return false;  // file does not exist -> not an existing store
    }
    detail::StoreHeader h;
    const std::size_t got = std::fread(&h, 1, sizeof(h), f);
    if (got != sizeof(h) ||
        !detail::header_mismatch_reason(h, page_size, capacity_bytes).empty()) {
        std::fclose(f);
        return false;
    }

    if (h.capacity_bytes > std::numeric_limits<uint64_t>::max() - h.page_size) {
        std::fclose(f);
        return false;
    }
    const uint64_t expected_size = h.capacity_bytes + h.page_size;
#if defined(_WIN32)
    const bool seek_ok = _fseeki64(f, 0, SEEK_END) == 0;
    const __int64 actual_size = seek_ok ? _ftelli64(f) : -1;
#else
    const bool seek_ok = ::fseeko(f, 0, SEEK_END) == 0;
    const off_t actual_size = seek_ok ? ::ftello(f) : static_cast<off_t>(-1);
#endif
    std::fclose(f);
    return seek_ok && actual_size >= 0 &&
           static_cast<uint64_t>(actual_size) == expected_size;
}

}  // namespace flashtier
