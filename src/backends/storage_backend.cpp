#include "flashtier/backends/storage_backend.hpp"

#include <cstdio>
#include <cstring>

#include "store_common.hpp"

namespace flashtier {

bool store_header_valid(const std::string& path, uint64_t page_size,
                        uint64_t capacity_bytes) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return false;  // file does not exist -> not an existing store
    }
    detail::StoreHeader h;
    const std::size_t got = std::fread(&h, 1, sizeof(h), f);
    std::fclose(f);
    if (got != sizeof(h)) return false;
    return detail::header_mismatch_reason(h, page_size, capacity_bytes).empty();
}

}  // namespace flashtier
