# Examples.

function(ft_add_example NAME)
  add_executable(${NAME} ${NAME}.cpp)
  target_link_libraries(${NAME} PRIVATE flashtier_core)
  if(FLASHTIER_HAVE_CUDA)
    target_link_libraries(${NAME} PRIVATE flashtier_cuda)
  endif()
  flashtier_apply_warnings(${NAME})
  add_test(NAME example_${NAME} COMMAND ${NAME})
  set_tests_properties(example_${NAME} PROPERTIES TIMEOUT 600)
endfunction()

ft_add_example(minimal_tiered_buffer)
ft_add_example(oversubscribed_working_set)
