# Benchmarks. Registered with CTest as performance tests so CI can run
# them explicitly without treating them as correctness gates.

# Shared benchmark library used by both the flashtier CLI and the
# standalone benchmark executables.
add_library(flashtier_benchlib STATIC bench_lib.cpp)
target_link_libraries(flashtier_benchlib PUBLIC flashtier_core)
if(FLASHTIER_HAVE_CUDA)
  target_link_libraries(flashtier_benchlib PUBLIC flashtier_cuda)
endif()
flashtier_apply_warnings(flashtier_benchlib)

function(ft_add_benchmark NAME)
  add_executable(${NAME} ${NAME}.cpp)
  target_link_libraries(${NAME} PRIVATE flashtier_benchlib)
  flashtier_apply_warnings(${NAME})
  add_test(NAME benchmark_${NAME} COMMAND ${NAME} --iterations 2)
  # Benchmarks are bounded local validation runs with small defaults.
  # Exit code 4 = "unsupported capability" (e.g. unified-memory on a
  # CPU-only build): CTest reports it as skipped, not failed.
  set_tests_properties(benchmark_${NAME} PROPERTIES TIMEOUT 60 LABELS benchmark
                       SKIP_RETURN_CODE 4)
endfunction()

ft_add_benchmark(benchmark_tiers)
ft_add_benchmark(benchmark_oversubscription)
ft_add_benchmark(benchmark_prefetch)
ft_add_benchmark(benchmark_sparse_experts)
ft_add_benchmark(benchmark_unified_memory)
