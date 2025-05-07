cmake_minimum_required(VERSION 3.10)
project(nbody_opencl)

find_package(OpenCL REQUIRED)

add_executable(nbody main.cpp)
target_include_directories(nbody PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(nbody PRIVATE ${OpenCL_LIBRARIES})