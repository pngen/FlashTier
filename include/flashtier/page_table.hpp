#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "flashtier/page.hpp"

namespace flashtier {

// Thread-safe page table. The single authority for residency, physical
// handles, storage offsets, transfer state, access metadata, dirty state,
// pin state, checksums, and memory accounting.
class PageTable {
public:
    bool contains(PageId id) const;
    std::vector<PageId> ids() const;
    std::size_t size() const;

    // Mutate a page's metadata under the write lock. Throws
    // ErrorCode::NotFound when the page does not exist.
    void with(PageId id, const std::function<void(PageMetadata&)>& fn);

    // Read-only access under a shared lock.
    void with(PageId id, const std::function<void(const PageMetadata&)>& fn) const;

    void insert(PageMetadata meta);
    void erase(PageId id);

    // Copy of the metadata without a callback (convenience; snapshot).
    PageMetadata copy_of(PageId id) const;

    void for_each(const std::function<void(PageMetadata&)>& fn);
    void for_each(const std::function<void(const PageMetadata&)>& fn) const;

    // Snapshot of the metadata of every page (used by planner and policy).
    std::vector<PageMetadata> snapshot() const;

private:
    mutable std::shared_mutex mu_;
    std::unordered_map<uint64_t, PageMetadata> pages_;
};

}  // namespace flashtier
