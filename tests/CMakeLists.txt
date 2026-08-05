# Unit tests. Each test is its own executable registered with CTest so a
# failure identifies the exact suite.
#
# GPU tests are registered with LABELS gpu and are NOT part of the CPU
# correctness suite (ctest -LE benchmark -LE gpu). They run only when
# CUDA is enabled and a device exists.

function(ft_add_test NAME)
  add_executable(${NAME} ${NAME}.cpp)
  target_link_libraries(${NAME} PRIVATE flashtier_core)
  if(FLASHTIER_HAVE_CUDA)
    # The core backend registry references register_cuda_backend, so every
    # target linking core in a CUDA build also links the CUDA backend.
    target_link_libraries(${NAME} PRIVATE flashtier_cuda)
  endif()
  flashtier_apply_warnings(${NAME})
  add_test(NAME ${NAME} COMMAND ${NAME})
  set_tests_properties(${NAME} PROPERTIES TIMEOUT 600)
endfunction()

ft_add_test(test_bytesize)
ft_add_test(test_config)
ft_add_test(test_page_table)
ft_add_test(test_policy)
ft_add_test(test_eviction)
ft_add_test(test_integrity)
ft_add_test(test_telemetry)
ft_add_test(test_nvme_backend)
ft_add_test(test_runtime)
ft_add_test(test_no_cuda)

# Generic device-backend conformance (mock + CPU emulation) runs everywhere.
add_executable(test_device_backend_contract
  test_device_backend_contract.cpp
  conformance.cpp)
target_link_libraries(test_device_backend_contract PRIVATE flashtier_core)
if(FLASHTIER_HAVE_CUDA)
  target_link_libraries(test_device_backend_contract PRIVATE flashtier_cuda)
endif()
flashtier_apply_warnings(test_device_backend_contract)
add_test(NAME test_device_backend_contract COMMAND test_device_backend_contract)
set_tests_properties(test_device_backend_contract PROPERTIES TIMEOUT 600)

if(FLASHTIER_HAVE_CUDA)
  # CUDA backend conformance + allocation/round-trip (needs a GPU).
  add_executable(test_cuda_device_backend test_cuda_device_backend.cpp conformance.cpp)
  target_link_libraries(test_cuda_device_backend PRIVATE flashtier_core flashtier_cuda)
  flashtier_apply_warnings(test_cuda_device_backend)
  add_test(NAME test_cuda_device_backend COMMAND test_cuda_device_backend)
  set_tests_properties(test_cuda_device_backend PROPERTIES TIMEOUT 600 LABELS gpu)

  # Governed runtime on the CUDA backend: tiering, integrity, shutdown.
  add_executable(test_cuda_runtime test_cuda_runtime.cpp)
  target_link_libraries(test_cuda_runtime PRIVATE flashtier_core flashtier_cuda)
  flashtier_apply_warnings(test_cuda_runtime)
  add_test(NAME test_cuda_runtime COMMAND test_cuda_runtime)
  set_tests_properties(test_cuda_runtime PROPERTIES TIMEOUT 900 LABELS gpu)

  # True three-tier device -> host -> NVMe oversubscription.
  add_executable(test_cuda_oversubscription test_cuda_oversubscription.cpp)
  target_link_libraries(test_cuda_oversubscription PRIVATE flashtier_core flashtier_cuda)
  flashtier_apply_warnings(test_cuda_oversubscription)
  add_test(NAME test_cuda_oversubscription COMMAND test_cuda_oversubscription)
  set_tests_properties(test_cuda_oversubscription PROPERTIES TIMEOUT 900 LABELS gpu)

  # Deterministic shutdown with in-flight work.
  add_executable(test_cuda_shutdown test_cuda_shutdown.cpp)
  target_link_libraries(test_cuda_shutdown PRIVATE flashtier_core flashtier_cuda)
  flashtier_apply_warnings(test_cuda_shutdown)
  add_test(NAME test_cuda_shutdown COMMAND test_cuda_shutdown)
  set_tests_properties(test_cuda_shutdown PROPERTIES TIMEOUT 600 LABELS gpu)
endif()
