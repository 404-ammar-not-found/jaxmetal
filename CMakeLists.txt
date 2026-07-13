cmake_minimum_required(VERSION 3.20)
project(jax_metal_prototype LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Objective-C++ translation units use ARC; C++ ones are unaffected.
add_compile_options($<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)

# --- Embed hand-written .metal kernels as C++ string headers -----------------
# (runtime-compilation path: no `metal` CLI / .metallib required)
include(${CMAKE_SOURCE_DIR}/cmake/EmbedMetal.cmake)
set(GEN_DIR ${CMAKE_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GEN_DIR})
set(EMBED_OUTPUTS)
embed_metal(${CMAKE_SOURCE_DIR}/kernels/elementwise.metal
            ${GEN_DIR}/kernels_elementwise.h
            elementwise_msl)
embed_metal(${CMAKE_SOURCE_DIR}/kernels/matmul.metal
            ${GEN_DIR}/kernels_matmul.h
            matmul_msl)
embed_metal(${CMAKE_SOURCE_DIR}/kernels/nn.metal
            ${GEN_DIR}/kernels_nn.h
            nn_msl)
add_custom_target(embed_kernels DEPENDS ${EMBED_OUTPUTS})

# --- Apple frameworks --------------------------------------------------------
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)

# --- Metal runtime library ---------------------------------------------------
add_library(metal_rt STATIC
  src/metal/metal_context.mm
  src/metal/metal_buffer.mm
  src/metal/kernel_library.mm
  src/runtime/dispatcher.mm
  src/ops/elementwise.mm
  src/ops/matmul.mm
  src/ops/mps_matmul.mm
  src/ops/nn.mm
  src/ops/mlp.mm
  src/cpu/cpu_matmul.cpp
)
target_include_directories(metal_rt PUBLIC include ${GEN_DIR})
# Use Accelerate's current CBLAS headers (silences the cblas_sgemm deprecation).
target_compile_definitions(metal_rt PRIVATE ACCELERATE_NEW_LAPACK)
target_link_libraries(metal_rt PUBLIC
  ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK} ${ACCELERATE_FRAMEWORK} ${MPS_FRAMEWORK})
add_dependencies(metal_rt embed_kernels)  # ensure kernels are embedded first

# --- C ABI shared library (ctypes / Python frontend) -------------------------
# Optionally includes the XLA FFI custom-call handler (native jax.jit integration)
# when the jaxlib include dir is provided:
#   -DJAX_FFI_INCLUDE_DIR=$(python -c "import jax.ffi; print(jax.ffi.include_dir())")
set(JAX_FFI_INCLUDE_DIR "" CACHE PATH "jaxlib include dir providing xla/ffi/api headers")

set(CAPI_SOURCES src/capi/metal_capi.mm)
if(JAX_FFI_INCLUDE_DIR)
  list(APPEND CAPI_SOURCES src/ffi/metal_ffi.cpp)
endif()

add_library(metal_capi SHARED ${CAPI_SOURCES})
target_link_libraries(metal_capi PRIVATE metal_rt)
if(JAX_FFI_INCLUDE_DIR)
  target_include_directories(metal_capi PRIVATE ${JAX_FFI_INCLUDE_DIR})
  message(STATUS "XLA FFI handler enabled (jax.jit integration): ${JAX_FFI_INCLUDE_DIR}")
endif()

# --- Demo / smoke test -------------------------------------------------------
add_executable(arith_demo tools/arith_demo.mm)
target_link_libraries(arith_demo PRIVATE metal_rt)

# --- Unit tests --------------------------------------------------------------
enable_testing()

set(TEST_SRCS
  tests/cpp/runtime_test.cpp
  tests/cpp/elementwise_test.cpp
  tests/cpp/matmul_test.cpp
  tests/cpp/cpu_matmul_test.cpp
  tests/cpp/mps_matmul_test.cpp
  tests/cpp/nn_test.cpp
  tests/cpp/mlp_test.cpp
)

add_executable(kernel_tests
  tests/cpp/test_framework.cpp
  tests/cpp/gpu_test_util.cpp
  ${TEST_SRCS}
)
target_include_directories(kernel_tests PRIVATE tests/cpp)
target_link_libraries(kernel_tests PRIVATE metal_rt)

# Register each TEST(Name) as its own CTest case (source of truth = the macros).
foreach(_f ${TEST_SRCS})
  file(STRINGS ${CMAKE_SOURCE_DIR}/${_f} _hits REGEX "^TEST\\(")
  foreach(_line ${_hits})
    string(REGEX REPLACE "^TEST\\(([A-Za-z0-9_]+)\\).*" "\\1" _name "${_line}")
    add_test(NAME ${_name} COMMAND kernel_tests ${_name})
  endforeach()
endforeach()
