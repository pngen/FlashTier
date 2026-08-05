# Unit tests. Each test is its own executable registered with CTest so a
# failure identifies the exact suite.

function(ft_add_test NAME)
  add_executable(${NAME} ${NAME}.cpp)
  target_link_libraries(${NAME} PRIVATE flashtier_core)
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

if(FLASHTIER_HAVE_CUDA)
  add_executable(test_cuda test_cuda.cu)
  target_link_libraries(test_cuda PRIVATE flashtier_core flashtier_cuda)
  flashtier_apply_warnings(test_cuda)
  add_test(NAME test_cuda COMMAND test_cuda)
  set_tests_properties(test_cuda PROPERTIES TIMEOUT 600)
endif()
