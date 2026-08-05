#pragma once

#include "flashtier/cli.hpp"

namespace flashtier {
namespace bench {

// Benchmark command entry points. Each returns a stable CLI exit code
// (see cli::kExit*). Benchmarks never pretend to run: when a required
// capability is missing they print an explicit unsupported result and
// return cli::kExitUnsupported.
int run_tiers(const cli::Options& o);
int run_oversubscription(const cli::Options& o);
int run_prefetch(const cli::Options& o);
int run_sparse_experts(const cli::Options& o);
int run_unified_memory(const cli::Options& o);

int run_benchmark_command(const cli::Options& o);

// Runs fn with typed-error mapping for standalone benchmark executables.
int guard(const cli::Options& o, int (*fn)(const cli::Options&));

}  // namespace bench
}  // namespace flashtier
