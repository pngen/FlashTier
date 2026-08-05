#include "flashtier/page_table.hpp"

#include <algorithm>

#include "flashtier/error.hpp"

namespace flashtier {

bool PageTable::contains(PageId id) const {
    std::shared_lock lock(mu_);
    return pages_.count(id.value) != 0;
}

std::vector<PageId> PageTable::ids() const {
    std::shared_lock lock(mu_);
    std::vector<PageId> out;
    out.reserve(pages_.size());
    for (const auto& [k, v] : pages_) {
        (void)v;
        out.push_back(PageId{k});
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::size_t PageTable::size() const {
    std::shared_lock lock(mu_);
    return pages_.size();
}

void PageTable::with(PageId id, const std::function<void(PageMetadata&)>& fn) {
    std::unique_lock lock(mu_);
    auto it = pages_.find(id.value);
    if (it == pages_.end()) {
        throw Error(ErrorCode::NotFound, "page not found", id.to_string());
    }
    fn(it->second);
}

void PageTable::with(PageId id, const std::function<void(const PageMetadata&)>& fn) const {
    std::shared_lock lock(mu_);
    auto it = pages_.find(id.value);
    if (it == pages_.end()) {
        throw Error(ErrorCode::NotFound, "page not found", id.to_string());
    }
    fn(it->second);
}

void PageTable::insert(PageMetadata meta) {
    std::unique_lock lock(mu_);
    pages_.insert_or_assign(meta.id.value, std::move(meta));
}

void PageTable::erase(PageId id) {
    std::unique_lock lock(mu_);
    pages_.erase(id.value);
}

PageMetadata PageTable::copy_of(PageId id) const {
    std::shared_lock lock(mu_);
    auto it = pages_.find(id.value);
    if (it == pages_.end()) {
        throw Error(ErrorCode::NotFound, "page not found", id.to_string());
    }
    return it->second;
}

void PageTable::for_each(const std::function<void(PageMetadata&)>& fn) {
    std::unique_lock lock(mu_);
    for (auto& [k, v] : pages_) {
        (void)k;
        fn(v);
    }
}

void PageTable::for_each(const std::function<void(const PageMetadata&)>& fn) const {
    std::shared_lock lock(mu_);
    for (const auto& [k, v] : pages_) {
        (void)k;
        fn(v);
    }
}

std::vector<PageMetadata> PageTable::snapshot() const {
    std::shared_lock lock(mu_);
    std::vector<PageMetadata> out;
    out.reserve(pages_.size());
    for (const auto& [k, v] : pages_) {
        (void)k;
        out.push_back(v);
    }
    std::sort(out.begin(), out.end(),
              [](const PageMetadata& a, const PageMetadata& b) {
                  return a.id < b.id;
              });
    return out;
}

}  // namespace flashtier
