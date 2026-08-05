cmake_minimum_required(VERSION 3.24)

project(FlashTier
  VERSION 0.1.0
  DESCRIPTION "Heterogeneous-memory runtime governing AI workload residency across GPU VRAM, system RAM, and NVMe storage"
  HOMEPAGE_URL "https://github.com/pngen/FlashTier"
  LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(FLASHTIER_ENABLE_CUDA           "Build with NVIDIA CUDA support" ON)
option(FLASHTIER_ENABLE_DIRECTSTORAGE  "Probe Windows DirectStorage (experimental capability probe only; never required)" OFF)
option(FLASHTIER_ENABLE_GDS            "Enable GPUDirect Storage backend where available (Linux-first; never required)" OFF)
option(FLASHTIER_BUILD_TESTS           "Build unit tests" ON)
option(FLASHTIER_BUILD_BENCHMARKS      "Build benchmarks" ON)
option(FLASHTIER_BUILD_EXAMPLES        "Build examples" ON)
option(FLASHTIER_WARNINGS_AS_ERRORS    "Treat warnings in project-owned code as errors" OFF)

set(FLASHTIER_CUDA_ARCHITECTURES "native" CACHE STRING
    "CUDA architectures to target, e.g. 'native' (detect host GPU) or '90;120'")

set(FLASHTIER_MIN_CUDA_VERSION 12.0 CACHE STRING "Minimum CUDA toolkit version required when FLASHTIER_ENABLE_CUDA=ON")

# ---------------------------------------------------------------------------
# Git revision (best effort; source archives without .git degrade gracefully)
# ---------------------------------------------------------------------------
set(FLASHTIER_GIT_REVISION "unknown")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
  execute_process(
    COMMAND git -C "${CMAKE_CURRENT_SOURCE_DIR}" rev-parse --short=12 HEAD
    OUTPUT_VARIABLE FLASHTIER_GIT_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
endif()
set(FLASHTIER_GIT_REVISION "${FLASHTIER_GIT_REVISION}" CACHE STRING "Git revision baked into the build (set manually in archives)")

# ---------------------------------------------------------------------------
# Compiler setup
# ---------------------------------------------------------------------------
if(MSVC)
  add_compile_options(/W4 /permissive- /Zc:__cplusplus)
  add_compile_definitions(WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS)
else()
  add_compile_options(-Wall -Wextra)
endif()

# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------
set(FLASHTIER_HAVE_CUDA OFF)
if(FLASHTIER_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit "${FLASHTIER_MIN_CUDA_VERSION}" REQUIRED)
  set(FLASHTIER_HAVE_CUDA ON)
  if(FLASHTIER_CUDA_ARCHITECTURES STREQUAL "native")
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    set(CMAKE_CUDA_ARCHITECTURES "${FLASHTIER_CUDA_ARCHITECTURES}")
  endif()
  message(STATUS "FlashTier: CUDA toolkit ${CUDAToolkit_VERSION} at ${CUDAToolkit_LIBRARY_DIR}")
  message(STATUS "FlashTier: CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Core library (C++, compiles without CUDA)
# ---------------------------------------------------------------------------
add_library(flashtier_core STATIC
  src/config.cpp
  src/device_info.cpp
  src/page.cpp
  src/allocator.cpp
  src/planner.cpp
  src/policy.cpp
  src/telemetry.cpp
  src/integrity.cpp
  src/workload.cpp
  src/backends/storage_backend.cpp
  src/backends/host_backend.cpp
  src/backends/nvme_backend.cpp
  src/runtime.cpp)

target_include_directories(flashtier_core PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")

target_compile_definitions(flashtier_core PRIVATE
  FLASHTIER_VERSION="${PROJECT_VERSION}"
  FLASHTIER_GIT_REVISION="${FLASHTIER_GIT_REVISION}")

if(FLASHTIER_HAVE_CUDA)
  target_compile_definitions(flashtier_core PUBLIC FLASHTIER_HAVE_CUDA=1)
  target_include_directories(flashtier_core PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(flashtier_core PUBLIC CUDA::cudart)
else()
  target_compile_definitions(flashtier_core PUBLIC FLASHTIER_HAVE_CUDA=0)
endif()

target_link_libraries(flashtier_core PUBLIC Threads::Threads)

if(FLASHTIER_WARNINGS_AS_ERRORS)
  if(MSVC)
    target_compile_options(flashtier_core PRIVATE /WX)
  else()
    target_compile_options(flashtier_core PRIVATE -Werror)
  endif()
endif()

# ---------------------------------------------------------------------------
# CUDA-only library (device integration)
# ---------------------------------------------------------------------------
if(FLASHTIER_HAVE_CUDA)
  add_library(flashtier_cuda STATIC
    src/backends/vram_backend.cu
    src/backends/unified_memory.cu)
  target_include_directories(flashtier_cuda PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>")
  target_compile_definitions(flashtier_cuda PRIVATE FLASHTIER_HAVE_CUDA=1)
  target_compile_options(flashtier_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)
  if(FLASHTIER_WARNINGS_AS_ERRORS)
    if(MSVC)
      target_compile_options(flashtier_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/WX>)
    else()
      target_compile_options(flashtier_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>)
    endif()
  endif()
  target_link_libraries(flashtier_cuda PUBLIC CUDA::cudart)
endif()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
add_executable(flashtier src/cli/main.cpp)
target_link_libraries(flashtier PRIVATE flashtier_core)
if(FLASHTIER_HAVE_CUDA)
  target_link_libraries(flashtier PRIVATE flashtier_cuda)
endif()

if(FLASHTIER_WARNINGS_AS_ERRORS)
  if(MSVC)
    target_compile_options(flashtier PRIVATE /WX)
  else()
    target_compile_options(flashtier PRIVATE -Werror)
  endif()
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(FLASHTIER_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
if(FLASHTIER_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
if(FLASHTIER_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
