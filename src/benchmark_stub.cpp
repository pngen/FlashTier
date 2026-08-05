#include <cstdio>

#include "flashtier/bench.hpp"
#include "flashtier/cli.hpp"

namespace flashtier {
namespace bench {

int run_benchmark_command(const cli::Options& o) {
    std::fprintf(stderr,
                 "flashtier: benchmark commands are not available in this build "
                 "(FLASHTIER_BUILD_BENCHMARKS=OFF)\n");
    (void)o;
    return cli::kExitUnsupported;
}

}  // namespace bench
}  // namespace flashtier
