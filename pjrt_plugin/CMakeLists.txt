cmake_minimum_required(VERSION 3.17)
project(ds_pjrt_plugin)

find_package(CUDAToolkit REQUIRED)

# Assuming PJRT headers are in /opt/xla-pjrt based on your handoff document
add_library(ds_pjrt_plugin SHARED ds_pjrt_plugin.cpp)
target_include_directories(ds_pjrt_plugin PRIVATE /opt/xla-pjrt)
target_link_libraries(ds_pjrt_plugin PRIVATE CUDA::cudart)
