#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace flashtier {

// Memory tiers governed by the runtime.
enum class Tier : int {
    None = 0,
    Vram = 1,
    HostPinned = 2,
    Nvme = 3,
    HostPageable = 4,  // diagnostic/fallback only; never silently used for
                       // pinned-host performance claims
};

const char* tier_name(Tier tier) noexcept;
bool tier_from_name(std::string_view name, Tier& out) noexcept;

}  // namespace flashtier
