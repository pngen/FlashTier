#include "flashtier/page.hpp"

#include "flashtier/error.hpp"

namespace flashtier {

const char* page_state_name(PageState state) noexcept {
    switch (state) {
        case PageState::Unallocated: return "unallocated";
        case PageState::ResidentNvme: return "resident_nvme";
        case PageState::LoadingToHost: return "loading_to_host";
        case PageState::ResidentHost: return "resident_host";
        case PageState::LoadingToVram: return "loading_to_vram";
        case PageState::ResidentVram: return "resident_vram";
        case PageState::EvictingToHost: return "evicting_to_host";
        case PageState::EvictingToNvme: return "evicting_to_nvme";
        case PageState::Error: return "error";
        case PageState::Released: return "released";
    }
    return "unknown";
}

bool transition_allowed(PageState from, PageState to) noexcept {
    switch (from) {
        case PageState::Unallocated:
            return to == PageState::LoadingToVram ||
                   to == PageState::ResidentHost ||
                   to == PageState::ResidentNvme ||
                   to == PageState::ResidentVram ||
                   to == PageState::Released;
        case PageState::LoadingToVram:
            return to == PageState::ResidentVram || to == PageState::Error;
        case PageState::LoadingToHost:
            return to == PageState::ResidentHost || to == PageState::Error;
        case PageState::ResidentVram:
            // EvictingToHost = demotion with transfer; ResidentNvme = clean
            // drop of a page whose NVMe copy is still valid (no transfer).
            return to == PageState::EvictingToHost || to == PageState::ResidentNvme ||
                   to == PageState::Error || to == PageState::Released;
        case PageState::ResidentHost:
            return to == PageState::LoadingToVram ||
                   to == PageState::EvictingToNvme || to == PageState::Error ||
                   to == PageState::Released;
        case PageState::ResidentNvme:
            return to == PageState::LoadingToHost || to == PageState::Error ||
                   to == PageState::Released;
        case PageState::EvictingToHost:
            return to == PageState::ResidentHost || to == PageState::Error;
        case PageState::EvictingToNvme:
            return to == PageState::ResidentNvme || to == PageState::Error;
        case PageState::Error:
            return to == PageState::Released;
        case PageState::Released:
            return false;
    }
    return false;
}

const char* semantic_class_name(SemanticClass cls) noexcept {
    switch (cls) {
        case SemanticClass::Generic: return "generic";
        case SemanticClass::Weight: return "weight";
        case SemanticClass::KvCache: return "kv_cache";
        case SemanticClass::MoeExpert: return "moe_expert";
        case SemanticClass::Activation: return "activation";
        case SemanticClass::Scratch: return "scratch";
    }
    return "unknown";
}

bool semantic_class_from_name(std::string_view name, SemanticClass& out) noexcept {
    if (name == "generic") { out = SemanticClass::Generic; return true; }
    if (name == "weight") { out = SemanticClass::Weight; return true; }
    if (name == "kv_cache") { out = SemanticClass::KvCache; return true; }
    if (name == "moe_expert") { out = SemanticClass::MoeExpert; return true; }
    if (name == "activation") { out = SemanticClass::Activation; return true; }
    if (name == "scratch") { out = SemanticClass::Scratch; return true; }
    return false;
}

void PageMetadata::transition(PageState to) {
    if (!transition_allowed(state, to)) {
        throw Error(ErrorCode::State,
                    "illegal page state transition " +
                        std::string(page_state_name(state)) + " -> " +
                        std::string(page_state_name(to)),
                    "page " + id.to_string());
    }
    state = to;
    switch (to) {
        case PageState::ResidentVram: current_tier = Tier::Vram; break;
        case PageState::ResidentHost: current_tier = Tier::HostPinned; break;
        case PageState::ResidentNvme: current_tier = Tier::Nvme; break;
        case PageState::Unallocated: current_tier = Tier::None; break;
        case PageState::Released: current_tier = Tier::None; break;
        default: break;  // transfer states keep the source tier
    }
}

}  // namespace flashtier
