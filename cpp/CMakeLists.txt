cmake_minimum_required(VERSION 3.20)
project(ds_ffi LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)

add_library(ds_ffi SHARED
  ds_from_f32.cpp
  ds_to_f32.cpp
  ds_from_f64.cpp
  ds_to_f64.cpp
  ds_add.cpp
  ds_mul.cpp
  ds_sub.cpp
  ds_fused_add_mul.cpp
  ds_fused_add_mul_to_f64.cpp
  ds_fused_add_mul_to_f64_cuda.cu
  ds_fused_add_sub_to_f64_cuda.cu
  ds_recurrence_add_mul_to_f64_cuda.cu
)

target_include_directories(ds_ffi PRIVATE /opt/xla-ffi)
target_link_libraries(ds_ffi PRIVATE CUDA::cudart)
set_target_properties(ds_ffi PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
