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
# Honors the FLASHTIER_WARNINGS_AS_ERRORS option for native code. CUDA
# translation units receive /W4 or -Wall -Wextra through nvcc.

function(flashtier_apply_warnings TARGET)
  if(MSVC)
    target_compile_options(${TARGET} PRIVATE /W4 /permissive- /Zc:__cplusplus)
    if(FLASHTIER_WARNINGS_AS_ERRORS)
      target_compile_options(${TARGET} PRIVATE /WX)
    endif()
  else()
    target_compile_options(${TARGET} PRIVATE -Wall -Wextra)
    if(FLASHTIER_WARNINGS_AS_ERRORS)
      target_compile_options(${TARGET} PRIVATE -Werror)
    endif()
  endif()
endfunction()
