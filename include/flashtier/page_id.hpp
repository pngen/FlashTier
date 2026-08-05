#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace flashtier {

// Stable unique identifier for a managed region. Page IDs are never reused
// within a runtime instance: each allocation takes the next monotonically
// increasing ID.
struct PageId {
    uint64_t value = 0;

    bool operator==(const PageId&) const = default;
    bool operator!=(const PageId&) const = default;
    bool operator<(const PageId& other) const { return value < other.value; }

    std::string to_string() const;
};

struct PageIdHash {
    std::size_t operator()(const PageId& id) const noexcept {
        return std::hash<uint64_t>{}(id.value);
    }
};

}  // namespace flashtier
