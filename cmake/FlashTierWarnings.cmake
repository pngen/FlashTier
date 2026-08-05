# FlashTierWarnings.cmake
#
# Centralizes strict warning settings for project-owned code.
# Warnings-as-errors is applied ONLY to project-owned targets, never to
# external headers or third-party dependencies.
#
# Usage:
#   include(FlashTierWarnings)
#   flashtier_apply_warnings(<target>)
#
# Flags are scoped to CXX so nvcc never receives bare MSVC options; the
# CUDA target passes its own flags through -Xcompiler.
# Honors the FLASHTIER_WARNINGS_AS_ERRORS option.

function(flashtier_apply_warnings TARGET)
  if(MSVC)
    target_compile_options(${TARGET} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/W4>
      $<$<COMPILE_LANGUAGE:CXX>:/permissive->
      $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>)
    if(FLASHTIER_WARNINGS_AS_ERRORS)
      target_compile_options(${TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/WX>)
    endif()
  else()
    target_compile_options(${TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall>)
    target_compile_options(${TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
    if(FLASHTIER_WARNINGS_AS_ERRORS)
      target_compile_options(${TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
    endif()
  endif()
endfunction()
