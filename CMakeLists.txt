cmake_minimum_required(VERSION 3.24)

project(FlashTier
  VERSION 0.1.0
  DESCRIPTION "Heterogeneous-memory runtime governing AI workload residency across GPU VRAM, system RAM, and NVMe storage"
  HOMEPAGE_URL "https://github.com/pngen/FlashTier"
  LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FlashTierWarnings)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(FLASHTIER_ENABLE_CUDA           "Build with NVIDIA CUDA support" ON)
option(FLASHTIER_ENABLE_HIP            "Build with AMD HIP/ROCm support (requires HIP toolchain)" OFF)
option(FLASHTIER_ENABLE_LEVEL_ZERO     "Build the fail-closed Intel Level Zero containment stub" ON)
option(FLASHTIER_ENABLE_VULKAN         "Build the fail-closed Vulkan containment stub" ON)
option(FLASHTIER_ENABLE_METAL          "Build the fail-closed Apple Metal containment stub (macOS only)" ON)
option(FLASHTIER_ENABLE_CPU_BACKEND    "Build the CPU emulation backend" ON)
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
    RESULT_VARIABLE FLASHTIER_GIT_RESULT
    ERROR_QUIET)
  if(NOT FLASHTIER_GIT_RESULT EQUAL 0 OR FLASHTIER_GIT_REVISION STREQUAL "")
    set(FLASHTIER_GIT_REVISION "unknown")
  endif()
endif()
set(FLASHTIER_GIT_REVISION "${FLASHTIER_GIT_REVISION}" CACHE STRING "Git revision baked into the build (set manually in archives)")

# ---------------------------------------------------------------------------
# Compiler setup
# ---------------------------------------------------------------------------
# FlashTier is C++20 throughout; the standard is a project property, not a
# preset-only setting.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(FLASHTIER_ENABLE_CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(MSVC)
  # Compiler warning flags are applied per-target by
  # flashtier_apply_warnings(), scoped to CXX so nvcc never receives bare
  # MSVC options. Defines are language-neutral.
  add_compile_definitions(WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS)
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

# ---------------------------------------------------------------------------
# HIP / ROCm (AMD)
# ---------------------------------------------------------------------------
set(FLASHTIER_HAVE_HIP OFF)
if(FLASHTIER_ENABLE_HIP)
  find_package(HIP REQUIRED)
  set(FLASHTIER_HAVE_HIP ON)
  message(STATUS "FlashTier: HIP toolchain ${HIP_VERSION}")
endif()

# These options retain registry/API compatibility while their implementations
# fail closed. They must not become available until conforming backends pass
# their hardware proof sequences.
set(FLASHTIER_HAVE_LEVEL_ZERO ${FLASHTIER_ENABLE_LEVEL_ZERO})
set(FLASHTIER_HAVE_VULKAN ${FLASHTIER_ENABLE_VULKAN})
if(APPLE)
  set(FLASHTIER_HAVE_METAL ${FLASHTIER_ENABLE_METAL})
else()
  set(FLASHTIER_HAVE_METAL OFF)
endif()
set(FLASHTIER_HAVE_CPU_BACKEND ${FLASHTIER_ENABLE_CPU_BACKEND})

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Core library (C++, compiles without any accelerator SDK)
# ---------------------------------------------------------------------------
set(FLASHTIER_CORE_SOURCES
  src/error.cpp
  src/config.cpp
  src/cli.cpp
  src/device_info.cpp
  src/page.cpp
  src/page_id.cpp
  src/page_table.cpp
  src/tier.cpp
  src/allocator.cpp
  src/planner.cpp
  src/policy.cpp
  src/telemetry.cpp
  src/integrity.cpp
  src/workload.cpp
  src/backends/backend_registry.cpp
  src/backends/storage_backend.cpp
  src/backends/host_backend.cpp
  src/backends/nvme_backend.cpp
  src/runtime.cpp)

if(FLASHTIER_HAVE_CPU_BACKEND)
  list(APPEND FLASHTIER_CORE_SOURCES src/backends/cpu_backend.cpp)
endif()
if(FLASHTIER_HAVE_LEVEL_ZERO)
  list(APPEND FLASHTIER_CORE_SOURCES src/backends/level_zero_backend.cpp)
endif()
if(FLASHTIER_HAVE_VULKAN)
  list(APPEND FLASHTIER_CORE_SOURCES src/backends/vulkan_backend.cpp)
endif()
if(FLASHTIER_HAVE_METAL)
  list(APPEND FLASHTIER_CORE_SOURCES src/backends/metal_backend.cpp)
endif()
if(FLASHTIER_HAVE_HIP)
  list(APPEND FLASHTIER_CORE_SOURCES src/backends/hip_backend.cpp)
endif()

add_library(flashtier_core STATIC ${FLASHTIER_CORE_SOURCES})
flashtier_apply_warnings(flashtier_core)

target_include_directories(flashtier_core PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")

target_compile_definitions(flashtier_core PRIVATE
  FLASHTIER_VERSION="${PROJECT_VERSION}"
  FLASHTIER_GIT_REVISION="${FLASHTIER_GIT_REVISION}"
  FLASHTIER_ENABLE_DIRECTSTORAGE=$<BOOL:${FLASHTIER_ENABLE_DIRECTSTORAGE}>
  FLASHTIER_ENABLE_GDS=$<BOOL:${FLASHTIER_ENABLE_GDS}>)

target_compile_definitions(flashtier_core PUBLIC
  FLASHTIER_HAVE_CUDA=$<BOOL:${FLASHTIER_HAVE_CUDA}>
  FLASHTIER_HAVE_HIP=$<BOOL:${FLASHTIER_HAVE_HIP}>
  FLASHTIER_HAVE_LEVEL_ZERO=$<BOOL:${FLASHTIER_HAVE_LEVEL_ZERO}>
  FLASHTIER_HAVE_VULKAN=$<BOOL:${FLASHTIER_HAVE_VULKAN}>
  FLASHTIER_HAVE_METAL=$<BOOL:${FLASHTIER_HAVE_METAL}>
  FLASHTIER_HAVE_CPU_BACKEND=$<BOOL:${FLASHTIER_HAVE_CPU_BACKEND}>)

if(FLASHTIER_HAVE_CUDA)
  target_include_directories(flashtier_core PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(flashtier_core PUBLIC CUDA::cudart)
endif()
if(FLASHTIER_HAVE_HIP)
  target_include_directories(flashtier_core PUBLIC ${HIP_INCLUDE_DIRS})
  target_link_libraries(flashtier_core PUBLIC hip::host)
endif()

target_link_libraries(flashtier_core PUBLIC Threads::Threads ${CMAKE_DL_LIBS})

# ---------------------------------------------------------------------------
# CUDA-only library (vendor backend implementation)
# ---------------------------------------------------------------------------
if(FLASHTIER_HAVE_CUDA)
  add_library(flashtier_cuda STATIC
    src/backends/cuda_backend.cu
    src/backends/unified_memory.cu)
  target_include_directories(flashtier_cuda PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>")
  target_compile_definitions(flashtier_cuda PRIVATE FLASHTIER_HAVE_CUDA=1)
  if(MSVC)
    target_compile_options(flashtier_cuda PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4,-permissive-,/Zc:__cplusplus>)
  else()
    target_compile_options(flashtier_cuda PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
  endif()
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
flashtier_apply_warnings(flashtier)
target_compile_definitions(flashtier PRIVATE
  FLASHTIER_VERSION="${PROJECT_VERSION}"
  FLASHTIER_GIT_REVISION="${FLASHTIER_GIT_REVISION}")
target_link_libraries(flashtier PRIVATE flashtier_core)
if(FLASHTIER_HAVE_CUDA)
  target_link_libraries(flashtier PRIVATE flashtier_cuda)
endif()
if(FLASHTIER_BUILD_BENCHMARKS)
  target_link_libraries(flashtier PRIVATE flashtier_benchlib)
else()
  target_sources(flashtier PRIVATE src/benchmark_stub.cpp)
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
