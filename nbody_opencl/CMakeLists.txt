cmake_minimum_required(VERSION 3.10)
project(nbody_opencl)

find_package(OpenCL REQUIRED)

add_executable(nbody
    src/main.cpp
    src/nbody_opencl.cpp
    src/nbody_cpu.cpp
)

target_include_directories(nbody PRIVATE
    ${OpenCL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(nbody PRIVATE ${OpenCL_LIBRARIES})

configure_file(kernel.cl kernel.cl COPYONLY)
configure_file(kernel_local.cl kernel_local.cl COPYONLY)
configure_file(kernel_2d.cl kernel_2d.cl COPYONLY)