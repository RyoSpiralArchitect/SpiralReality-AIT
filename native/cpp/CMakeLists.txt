cmake_minimum_required(VERSION 3.14)
project(spiral_boundary_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit QUIET)

pybind11_add_module(spiral_boundary_cpp spiral_boundary_cpp.cpp)
install(TARGETS spiral_boundary_cpp LIBRARY DESTINATION spiralreality_AIT_onepass_aifcore_integrated/integrated)

option(SPIRAL_BUILD_GPU "Build the GPU accelerated boundary student" ON)
if(SPIRAL_BUILD_GPU)
    pybind11_add_module(spiral_boundary_gpu spiral_boundary_gpu.cpp)
    if(CUDAToolkit_FOUND)
        target_compile_definitions(spiral_boundary_gpu PRIVATE SPIRAL_HAS_CUDA=1)
        target_link_libraries(spiral_boundary_gpu PRIVATE CUDA::cudart)
    else()
        message(WARNING "CUDA toolkit not found; spiral_boundary_gpu will run in CPU emulation mode")
    endif()
    install(TARGETS spiral_boundary_gpu LIBRARY DESTINATION spiralreality_AIT_onepass_aifcore_integrated/integrated)
endif()
