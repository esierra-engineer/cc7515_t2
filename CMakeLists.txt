cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(nbody_cuda LANGUAGES CXX CUDA)

# Establecer estándares
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Incluir directorios de encabezados
include_directories(include)

# Archivos fuente
set(SOURCES
        nbody_cuda/src/main.cpp
        nbody_cuda/src/nbody_cpu.cpp
        nbody_cuda/src/nbody_cuda.cu
        nbody_cuda/src/utils.cpp
)

# Crear el ejecutable
add_executable(cc7515_t2 ${SOURCES})

# Propiedades específicas para CUDA
set_target_properties(cc7515_t2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES 52 60 70 75 86
)

# Opciones de compilación
target_compile_options(cc7515_t2 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-O2>
)
