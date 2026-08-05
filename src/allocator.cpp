#include "flashtier/allocator.hpp"

#include <algorithm>

namespace flashtier {

void TierBudget::set_limit(uint64_t limit_bytes, double reserve_margin) {
    std::lock_guard lock(mu_);
    limit_ = limit_bytes;
    margin_ = reserve_margin;
}

namespace {

uint64_t reserve_of(uint64_t limit, double margin) {
    return static_cast<uint64_t>(static_cast<double>(limit) * margin);
}

}  // namespace

bool TierBudget::try_reserve(uint64_t bytes) {
    std::lock_guard lock(mu_);
    const uint64_t usable = limit_ - reserve_of(limit_, margin_);
    if (bytes > usable || used_ > usable - bytes) {
        return false;
    }
    used_ += bytes;
    high_water_ = std::max(high_water_, used_);
    return true;
}

void TierBudget::release(uint64_t bytes) {
    std::lock_guard lock(mu_);
    used_ = (bytes >= used_) ? 0 : used_ - bytes;
}

void TierBudget::force_reserve(uint64_t bytes) {
    std::lock_guard lock(mu_);
    used_ += bytes;
    high_water_ = std::max(high_water_, used_);
}

uint64_t TierBudget::used() const {
    std::lock_guard lock(mu_);
    return used_;
}

uint64_t TierBudget::limit() const {
    std::lock_guard lock(mu_);
    return limit_;
}

uint64_t TierBudget::reserve_bytes() const {
    std::lock_guard lock(mu_);
    return reserve_of(limit_, margin_);
}

uint64_t TierBudget::headroom() const {
    std::lock_guard lock(mu_);
    const uint64_t usable = limit_ - reserve_of(limit_, margin_);
    return (used_ >= usable) ? 0 : usable - used_;
}

uint64_t TierBudget::high_water() const {
    std::lock_guard lock(mu_);
    return high_water_;
}

double TierBudget::margin() const {
    std::lock_guard lock(mu_);
    return margin_;
}

}  // namespace flashtier
