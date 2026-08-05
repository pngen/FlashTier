#include "test_harness.hpp"

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
