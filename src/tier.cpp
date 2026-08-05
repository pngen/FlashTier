#include "flashtier/tier.hpp"

namespace flashtier {

const char* tier_name(Tier tier) noexcept {
    switch (tier) {
        case Tier::None: return "none";
        case Tier::Vram: return "vram";
        case Tier::HostPinned: return "host_pinned";
        case Tier::Nvme: return "nvme";
        case Tier::HostPageable: return "host_pageable";
    }
    return "unknown";
}

bool tier_from_name(std::string_view name, Tier& out) noexcept {
    if (name == "vram") { out = Tier::Vram; return true; }
    if (name == "host_pinned" || name == "host") { out = Tier::HostPinned; return true; }
    if (name == "nvme" || name == "storage") { out = Tier::Nvme; return true; }
    if (name == "host_pageable" || name == "pageable") { out = Tier::HostPageable; return true; }
    return false;
}

}  // namespace flashtier
