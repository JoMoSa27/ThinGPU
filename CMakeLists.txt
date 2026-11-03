cmake_minimum_required(VERSION 3.18)
project(thingpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)   # NVIDIA L40S; override with -DCMAKE_CUDA_ARCHITECTURES=...
endif()

find_package(CUDAToolkit REQUIRED)

set(SPATIAL ${CMAKE_SOURCE_DIR}/third_party/spatial)
add_executable(thingpu-gpu
  src/thingpu_gpu.cu
  ${SPATIAL}/KDTree/src/KDBuilder.cu
  ${SPATIAL}/UniformGrid/src/GridBuilder.cu
  ${SPATIAL}/LinearBVH/src/BVHBuilder.cu)
target_include_directories(thingpu-gpu PRIVATE
  ${SPATIAL}/common/include ${SPATIAL}/KDTree/include
  ${SPATIAL}/UniformGrid/include ${SPATIAL}/LinearBVH/include)
set_target_properties(thingpu-gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(thingpu-gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O2 --expt-extended-lambda>)
