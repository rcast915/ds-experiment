cmake_minimum_required(VERSION 3.20)
project(ds_gemm LANGUAGES CXX CUDA)

# libds_gemm_ffi.so: the DS matmul as an XLA FFI handler (symbol `DsGemm`),
# loaded and registered by the proxy plugin. Built by ds_setup.sh into
# cpp/ds_gemm/build/ffi. The standalone benchmark (ds_gemm_bench.cu) is built
# separately by tests/run_ds_gemm.sh.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # Ampere, Ada (L40S), Hopper (H100); compute_90 PTX lets newer GPUs JIT it.
  set(CMAKE_CUDA_ARCHITECTURES "80-real;89-real;90")
endif()

find_package(CUDAToolkit REQUIRED)

add_library(ds_gemm_ffi SHARED ds_gemm_ffi.cu)
target_include_directories(ds_gemm_ffi PRIVATE /opt/xla-ffi ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ds_gemm_ffi PRIVATE CUDA::cudart)
