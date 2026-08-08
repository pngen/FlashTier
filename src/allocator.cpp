#include "flashtier/allocator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "flashtier/error.hpp"

namespace flashtier {

void TierBudget::set_limit(uint64_t limit_bytes, double reserve_margin) {
    if (!std::isfinite(reserve_margin) || reserve_margin < 0.0 ||
        reserve_margin >= 1.0) {
        throw Error(ErrorCode::InvalidArgument,
                    "tier reserve margin must be finite and in [0, 1)");
    }
    std::lock_guard lock(mu_);
    const uint64_t reserve = static_cast<uint64_t>(
        static_cast<double>(limit_bytes) * reserve_margin);
    const uint64_t usable = limit_bytes - reserve;
    if (used_ > usable) {
        throw Error(ErrorCode::Budget,
                    "new tier limit is below current reserved usage");
    }
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
    if (bytes > used_) {
        throw Error(ErrorCode::Invariant,
                    "tier budget release exceeds reserved usage",
                    "release " + std::to_string(bytes) +
                        " used " + std::to_string(used_));
    }
    used_ -= bytes;
}

void TierBudget::force_reserve(uint64_t bytes) {
    std::lock_guard lock(mu_);
    if (bytes > std::numeric_limits<uint64_t>::max() - used_) {
        throw Error(ErrorCode::Budget, "tier budget accounting overflow");
    }
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
