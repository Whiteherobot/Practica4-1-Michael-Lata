cmake_minimum_required(VERSION 3.20)
project(Practica4_1C_OpenCV_CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

set(CUDAToolkit_ROOT /opt/cuda)
set(CUDA_TOOLKIT_ROOT_DIR /opt/cuda)
set(ENV{PATH} "/opt/cuda/bin:$ENV{PATH}")

set(OpenCV_DIR /usr/lib/cmake/opencv5)
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

if(NOT OpenCV_CUDA_VERSION)
    message(WARNING "OpenCV compilado sin CUDA -- pipeline GPU no disponible")
else()
    message(STATUS "OpenCV CUDA version: ${OpenCV_CUDA_VERSION}")
endif()

find_package(CUDA)
if(CUDA_FOUND)
    message(STATUS "CUDA version: ${CUDA_VERSION}")
    enable_language(CUDA)
endif()

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(practica4_1c main.cpp)
target_link_libraries(practica4_1c ${OpenCV_LIBS})
