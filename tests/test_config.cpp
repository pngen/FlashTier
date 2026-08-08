#include "test_harness.hpp"

#include <limits>

#include "flashtier/cli.hpp"
#include "flashtier/config.hpp"

using namespace flashtier;

FT_TEST(valid_config_passes) {
    Config cfg;
    const auto v = validate_config(cfg);
    FT_ASSERT(v.ok);
}

FT_TEST(page_size_validation) {
    Config cfg;
    cfg.page_size = 0;
    FT_ASSERT(!validate_config(cfg).ok);
    cfg.page_size = 1024;  // < 4096
    FT_ASSERT(!validate_config(cfg).ok);
    cfg.page_size = 4096;
    FT_ASSERT(validate_config(cfg).ok);
    cfg.page_size = 6144;  // multiple of 4096 but not power of two
    FT_ASSERT(!validate_config(cfg).ok);
    cfg.page_size = 2 * 1024 * 1024;
    FT_ASSERT(validate_config(cfg).ok);
}

FT_TEST(queue_and_iteration_validation) {
    Config cfg;
    cfg.queue_depth = 0;
    FT_ASSERT(!validate_config(cfg).ok);
    cfg.queue_depth = 8;
    FT_ASSERT(validate_config(cfg).ok);
    cfg.iterations = 0;
    FT_ASSERT(!validate_config(cfg).ok);
}

FT_TEST(budget_fraction_validation) {
    Config cfg;
    cfg.auto_host_fraction = 0.9;
    FT_ASSERT(!validate_config(cfg).ok);
    cfg.auto_host_fraction = 0.25;
    FT_ASSERT(validate_config(cfg).ok);
    cfg.auto_vram_fraction = 0.0;
    FT_ASSERT(!validate_config(cfg).ok);
    cfg.auto_vram_fraction = 0.75;
    FT_ASSERT(validate_config(cfg).ok);
    cfg.vram_reserve_margin = 0.6;
    FT_ASSERT(!validate_config(cfg).ok);
}

FT_TEST(nonfinite_fractions_are_rejected) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    Config cfg;
    cfg.auto_vram_fraction = nan;
    FT_ASSERT(!validate_config(cfg).ok);
    cfg = Config{};
    cfg.auto_host_fraction = nan;
    FT_ASSERT(!validate_config(cfg).ok);
    cfg = Config{};
    cfg.auto_nvme_fraction = nan;
    FT_ASSERT(!validate_config(cfg).ok);
    cfg = Config{};
    cfg.vram_reserve_margin = nan;
    FT_ASSERT(!validate_config(cfg).ok);
}

FT_TEST(direct_config_bounds_and_enums_are_rejected) {
    Config cfg;
    cfg.queue_depth = 1025;
    FT_ASSERT(!validate_config(cfg).ok);
    cfg = Config{};
    cfg.worker_threads = 1025;
    FT_ASSERT(!validate_config(cfg).ok);
    cfg = Config{};
    cfg.prefetch_depth = (1u << 20) + 1;
    FT_ASSERT(!validate_config(cfg).ok);
    cfg = Config{};
    cfg.iterations = 1000001;
    FT_ASSERT(!validate_config(cfg).ok);
    cfg = Config{};
    cfg.auto_nvme_cap = cfg.page_size + 1;
    FT_ASSERT(!validate_config(cfg).ok);
    cfg = Config{};
    cfg.policy = static_cast<PolicyKind>(99);
    FT_ASSERT(!validate_config(cfg).ok);
    cfg = Config{};
    cfg.prefetch = static_cast<PrefetchKind>(99);
    FT_ASSERT(!validate_config(cfg).ok);
}

FT_TEST(cli_rejects_nonfinite_partial_and_truncating_numbers) {
    auto parse_value = [](const char* option, const char* value) {
        char program[] = "flashtier";
        char option_buf[64] = {};
        char value_buf[64] = {};
        std::snprintf(option_buf, sizeof(option_buf), "%s", option);
        std::snprintf(value_buf, sizeof(value_buf), "%s", value);
        char* argv[] = {program, option_buf, value_buf};
        return cli::parse_args(3, argv);
    };

    FT_ASSERT(!parse_value("--auto-vram-fraction", "nan").errors.empty());
    FT_ASSERT(!parse_value("--auto-vram-fraction", "0.5junk").errors.empty());
    FT_ASSERT(parse_value("--auto-vram-fraction", "0.5").errors.empty());
    FT_ASSERT(!parse_value("--um-prefetch-pages", "4294967296").errors.empty());
    FT_ASSERT(parse_value("--um-prefetch-pages", "4294967295").errors.empty());
}

FT_TEST(policy_name_roundtrip) {
    PolicyKind k;
    FT_ASSERT(policy_kind_from_name("lru", k) && k == PolicyKind::Lru);
    FT_ASSERT(policy_kind_from_name("predictive", k) && k == PolicyKind::Predictive);
    FT_ASSERT(!policy_kind_from_name("random", k));
    PrefetchKind p;
    FT_ASSERT(prefetch_kind_from_name("off", p) && p == PrefetchKind::Off);
    FT_ASSERT(prefetch_kind_from_name("sequential", p) && p == PrefetchKind::Sequential);
    FT_ASSERT(prefetch_kind_from_name("predictive", p) && p == PrefetchKind::Predictive);
    FT_ASSERT(!prefetch_kind_from_name("bogus", p));
}

int main() { return ft_test::run_all("test_config"); }
