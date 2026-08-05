#include <cstdio>

#include "flashtier/bench.hpp"
#include "flashtier/cli.hpp"

int main(int argc, char** argv) {
    flashtier::cli::Options o = flashtier::cli::parse_args(argc, argv);
    if (o.show_version) {
        flashtier::cli::print_version(stdout);
        return flashtier::cli::kExitOk;
    }
    if (o.show_help || !o.errors.empty()) {
        for (const auto& e : o.errors) std::fprintf(stderr, "flashtier: %s\n", e.c_str());
        flashtier::cli::print_usage(stderr);
        return flashtier::cli::kExitUsage;
    }
    return flashtier::bench::guard(o, flashtier::bench::run_tiers);
}
