cmake_minimum_required(VERSION 3.14)
project(spiral_boundary_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(spiral_boundary_cpp spiral_boundary_cpp.cpp)

install(TARGETS spiral_boundary_cpp LIBRARY DESTINATION spiralreality_AIT_onepass_aifcore_integrated/integrated)
