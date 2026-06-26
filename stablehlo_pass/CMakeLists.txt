cmake_minimum_required(VERSION 3.20)
project(DsTransformPass LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

find_package(MLIR REQUIRED CONFIG HINTS /opt/mlir/lib/cmake/mlir)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

file(GLOB MLIR_ALL_LIBS "/opt/mlir/lib/libMLIR*.a")
file(GLOB LLVM_ALL_LIBS "/opt/mlir/lib/libLLVM*.a")
set(VERSION_LIB /opt/mlir/lib/libVersion.a)

file(GLOB STABLEHLO_ALL_LIBS
    "/opt/mlir/lib/libStablehlo*.a"
    "/opt/mlir/lib/libChlo*.a"
    "/opt/mlir/lib/libVhlo*.a"
)

add_executable(mlir-ds-opt
    DsTransformPass.cpp
    mlir-ds-opt.cpp
)

target_include_directories(mlir-ds-opt PRIVATE
    ${MLIR_INCLUDE_DIRS}
    /opt/mlir/include
)

target_link_libraries(mlir-ds-opt PRIVATE
    -Wl,--start-group
    ${MLIR_ALL_LIBS}
    ${LLVM_ALL_LIBS}
    ${STABLEHLO_ALL_LIBS}
    ${VERSION_LIB}
    -Wl,--end-group
    -lrt -ldl -lm -lz -lzstd -lxml2
)

add_executable(mlir-ds-ffi-opt
    DsFFIPass.cpp
    mlir-ds-ffi-opt.cpp
)
target_include_directories(mlir-ds-ffi-opt PRIVATE
    ${MLIR_INCLUDE_DIRS}
    /opt/mlir/include
)
target_link_libraries(mlir-ds-ffi-opt PRIVATE
    -Wl,--start-group
    ${MLIR_ALL_LIBS}
    ${LLVM_ALL_LIBS}
    ${STABLEHLO_ALL_LIBS}
    ${VERSION_LIB}
    -Wl,--end-group
    -lrt -ldl -lm -lz -lzstd -lxml2
)
