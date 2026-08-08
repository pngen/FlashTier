#include "flashtier/config.hpp"

#include <cctype>
#include <cmath>
#include <cstdint>

#include "flashtier/error.hpp"

namespace flashtier {

const char* policy_kind_name(PolicyKind kind) noexcept {
    switch (kind) {
        case PolicyKind::Lru: return "lru";
        case PolicyKind::Predictive: return "predictive";
    }
    return "unknown";
}

bool policy_kind_from_name(std::string_view name, PolicyKind& out) noexcept {
    if (name == "lru") { out = PolicyKind::Lru; return true; }
    if (name == "predictive" || name == "temperature") { out = PolicyKind::Predictive; return true; }
    return false;
}

const char* prefetch_kind_name(PrefetchKind kind) noexcept {
    switch (kind) {
        case PrefetchKind::Off: return "off";
        case PrefetchKind::Sequential: return "sequential";
        case PrefetchKind::Predictive: return "predictive";
    }
    return "unknown";
}

bool prefetch_kind_from_name(std::string_view name, PrefetchKind& out) noexcept {
    if (name == "off" || name == "none") { out = PrefetchKind::Off; return true; }
    if (name == "sequential") { out = PrefetchKind::Sequential; return true; }
    if (name == "predictive") { out = PrefetchKind::Predictive; return true; }
    return false;
}

namespace {

constexpr uint64_t kKiB = 1024ull;
constexpr uint64_t kMiB = 1024ull * 1024;
constexpr uint64_t kGiB = 1024ull * 1024 * 1024;
constexpr uint64_t kTiB = 1024ull * 1024 * 1024 * 1024;

bool ascii_digit(char c) { return c >= '0' && c <= '9'; }

}  // namespace

uint64_t parse_bytesize(const std::string& text) {
    std::size_t i = 0;
    const std::size_t n = text.size();
    while (i < n && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
    if (i == n) {
        throw Error(ErrorCode::Config, "empty byte size", text);
    }
    uint64_t value = 0;
    bool any = false;
    while (i < n && ascii_digit(text[i])) {
        uint64_t d = static_cast<uint64_t>(text[i] - '0');
        if (value > (UINT64_MAX - d) / 10) {
            throw Error(ErrorCode::Config, "byte size overflows 64 bits", text);
        }
        value = value * 10 + d;
        any = true;
        ++i;
    }
    if (!any) {
        throw Error(ErrorCode::Config, "expected an unsigned integer byte size", text);
    }
    while (i < n && std::isspace(static_cast<unsigned char>(text[i]))) ++i;

    uint64_t multiplier = 1;
    if (i < n) {
        std::string suffix = text.substr(i);
        while (!suffix.empty() &&
               std::isspace(static_cast<unsigned char>(suffix.back()))) {
            suffix.pop_back();
        }
        // Normalize case.
        for (char& c : suffix) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        if (suffix == "B" || suffix == "BYTES") {
            multiplier = 1;
        } else if (suffix == "KIB") {
            multiplier = kKiB;
        } else if (suffix == "MIB") {
            multiplier = kMiB;
        } else if (suffix == "GIB") {
            multiplier = kGiB;
        } else if (suffix == "TIB") {
            multiplier = kTiB;
        } else {
            throw Error(ErrorCode::Config,
                        "unknown byte size suffix (expected KiB, MiB, GiB, TiB, or bare bytes)",
                        text);
        }
    }
    if (value > UINT64_MAX / multiplier) {
        throw Error(ErrorCode::Config, "byte size overflows 64 bits", text);
    }
    return value * multiplier;
}

std::string bytesize_to_string(uint64_t bytes) {
    const double gib = static_cast<double>(bytes) / static_cast<double>(kGiB);
    const double mib = static_cast<double>(bytes) / static_cast<double>(kMiB);
    const double kib = static_cast<double>(bytes) / static_cast<double>(kKiB);
    char buf[64];
    if (bytes >= kGiB) {
        std::snprintf(buf, sizeof(buf), "%.2f GiB", gib);
    } else if (bytes >= kMiB) {
        std::snprintf(buf, sizeof(buf), "%.2f MiB", mib);
    } else if (bytes >= kKiB) {
        std::snprintf(buf, sizeof(buf), "%.2f KiB", kib);
    } else {
        std::snprintf(buf, sizeof(buf), "%llu B", static_cast<unsigned long long>(bytes));
    }
    return buf;
}

ConfigValidation validate_config(const Config& cfg,
                                 const std::vector<std::string>& known_backends) {
    ConfigValidation out;
    out.ok = true;

    auto fail = [&](const std::string& msg) {
        out.ok = false;
        out.errors.push_back(msg);
    };

    if (!cfg.backend.empty() && cfg.backend != "auto") {
        bool found = false;
        for (const auto& b : known_backends) {
            if (b == cfg.backend) {
                found = true;
                break;
            }
        }
        if (!found) {
            fail("unknown device backend '" + cfg.backend + "'");
        }
    }
    if (!cfg.cuda_enabled && !cfg.backend.empty() && cfg.backend != "auto" &&
        cfg.backend != "cpu") {
        fail("disabled accelerator support conflicts with backend '" + cfg.backend + "'");
    }

    if (cfg.page_size == 0) {
        fail("page size must be nonzero");
    } else {
        if (cfg.page_size < 4096) fail("page size must be at least 4096 bytes");
        if (cfg.page_size % 4096 != 0) fail("page size must be a multiple of 4096 bytes");
        if ((cfg.page_size & (cfg.page_size - 1)) != 0) fail("page size must be a power of two");
    }
    if (cfg.queue_depth == 0 || cfg.queue_depth > 1024) {
        fail("queue depth must be in [1, 1024]");
    }
    if (cfg.worker_threads > 1024) fail("worker thread count must not exceed 1024");
    if (cfg.prefetch_depth == 0 || cfg.prefetch_depth > (1u << 20)) {
        fail("prefetch depth must be in [1, 1048576]");
    }
    if (cfg.iterations == 0 || cfg.iterations > 1000000) {
        fail("iterations must be in [1, 1000000]");
    }
    if (!std::isfinite(cfg.auto_vram_fraction) || cfg.auto_vram_fraction <= 0.0 ||
        cfg.auto_vram_fraction > 1.0) {
        fail("auto VRAM fraction must be finite and in (0, 1]");
    }
    if (!std::isfinite(cfg.auto_host_fraction) || cfg.auto_host_fraction <= 0.0 ||
        cfg.auto_host_fraction > 0.5) {
        fail("auto host fraction must be finite and in (0, 0.5]");
    }
    if (!std::isfinite(cfg.auto_nvme_fraction) || cfg.auto_nvme_fraction <= 0.0 ||
        cfg.auto_nvme_fraction > 0.5) {
        fail("auto NVMe fraction must be finite and in (0, 0.5]");
    }
    if (!std::isfinite(cfg.vram_reserve_margin) || cfg.vram_reserve_margin < 0.0 ||
        cfg.vram_reserve_margin >= 0.5) {
        fail("VRAM reserve margin must be finite and in [0, 0.5)");
    }
    if (cfg.page_size != 0 &&
        (cfg.auto_nvme_cap < cfg.page_size || cfg.auto_nvme_cap % cfg.page_size != 0)) {
        fail("auto NVMe cap must be a nonzero multiple of page size");
    }
    switch (cfg.policy) {
        case PolicyKind::Lru:
        case PolicyKind::Predictive: break;
        default: fail("invalid policy kind"); break;
    }
    switch (cfg.prefetch) {
        case PrefetchKind::Off:
        case PrefetchKind::Sequential:
        case PrefetchKind::Predictive: break;
        default: fail("invalid prefetch kind"); break;
    }
    if (cfg.device_id < 0) fail("device id must be nonnegative");

    return out;
}

std::string Config::describe() const {
    return "  backend: " + (backend.empty() ? std::string("auto") : backend) + "\n" +
           "  device: " + std::to_string(device_id) + "\n" +
           "  page size: " + bytesize_to_string(page_size) + "\n" +
           "  working set: " + (working_set_bytes ? bytesize_to_string(working_set_bytes) : std::string("auto")) + "\n" +
           "  vram budget: " + bytesize_to_string(vram_budget_bytes) + "\n" +
           "  host budget: " + bytesize_to_string(host_budget_bytes) + "\n" +
           "  nvme budget: " + bytesize_to_string(nvme_budget_bytes) + "\n" +
           "  nvme path: " + (nvme_path.empty() ? std::string("(temp)") : nvme_path) + "\n" +
           "  policy: " + policy_kind_name(policy) + "\n" +
           "  prefetch: " + prefetch_kind_name(prefetch) +
           (prefetch == PrefetchKind::Off ? std::string() : " (depth " + std::to_string(prefetch_depth) + ")") +
           "\n" +
           "  queue depth: " + std::to_string(queue_depth) + "\n" +
           "  iterations: " + std::to_string(iterations) + "\n" +
           "  seed: 0x" + [&]() {
               char buf[32];
               std::snprintf(buf, sizeof(buf), "%llX", static_cast<unsigned long long>(seed));
               return std::string(buf);
           }() + "\n" +
           "  vram reserve margin: " + std::to_string(vram_reserve_margin);
}

}  // namespace flashtier
