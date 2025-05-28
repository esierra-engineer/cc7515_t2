cmake_minimum_required(VERSION 3.10)

# Project and language setup
project(nbody_cuda LANGUAGES CXX CUDA)

# Compiler setup
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES native)

# Standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Include directories
include_directories(include)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(nbody_cuda/src)

# Source files
set(SOURCES
        nbody_cuda/src/main.cpp
        nbody_cuda/src/nbody_cpu.cpp
        nbody_cuda/src/nbody_cuda.cu
        nbody_cuda/src/utils.cpp
)

# Kernel PTX generation
set(KERNEL_SRCS
        ${CMAKE_SOURCE_DIR}/nbody_cuda/src/kernel_1_global-memory_1D.cu
        ${CMAKE_SOURCE_DIR}/nbody_cuda/src/kernel_2_shared-memory_1D.cu
        ${CMAKE_SOURCE_DIR}/nbody_cuda/src/kernel_3_global-memory_2D.cu
        ${CMAKE_SOURCE_DIR}/nbody_cuda/src/kernel_4_shared-memory_2D.cu
)

set(PTX_FILES "")
foreach(kernel ${KERNEL_SRCS})
    get_filename_component(kernel_name ${kernel} NAME_WE)
    set(ptx_file ${CMAKE_BINARY_DIR}/${kernel_name}.ptx)

    add_custom_command(
            OUTPUT ${ptx_file}
            COMMAND ${CMAKE_CUDA_COMPILER} --ptx ${kernel} -o ${ptx_file}
            DEPENDS ${kernel}
            COMMENT "Compiling PTX from ${kernel}"
    )

    list(APPEND PTX_FILES ${ptx_file})
endforeach()

add_custom_target(compile_ptx ALL DEPENDS ${PTX_FILES})

# Executable target
add_executable(cc7515_t2 ${SOURCES})

# Link against low-level CUDA driver API
target_link_libraries(cc7515_t2 PRIVATE cuda)

# CUDA-specific target properties
set_target_properties(cc7515_t2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES 52 60 70 75 86
)

# Compiler options
target_compile_options(cc7515_t2 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-O2>
)

# Ensure PTX is built before main executable
add_dependencies(cc7515_t2 compile_ptx)
