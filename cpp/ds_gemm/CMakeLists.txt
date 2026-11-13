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
  # Turing (T4), Ampere (A100; A6000/A2000), Ada (L40S), Hopper (H100),
  # Blackwell RTX PRO (sm_120); compute_90 PTX lets other newer GPUs JIT it.
  # PTX only runs on its own or a newer architecture, so older GPUs need
  # their own entry here.
  set(CMAKE_CUDA_ARCHITECTURES "75-real;80-real;86-real;89-real;90-real;120-real;90-virtual")
endif()

find_package(CUDAToolkit REQUIRED)

add_library(ds_gemm_ffi SHARED ds_gemm_ffi.cu)
target_include_directories(ds_gemm_ffi PRIVATE /opt/xla-ffi ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ds_gemm_ffi PRIVATE CUDA::cudart)
