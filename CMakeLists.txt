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

# Incluir directorios (por si usas headers de CUDA)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

include_directories((nbody_cuda/src))

# Archivos fuente
set(SOURCES
        nbody_cuda/src/main.cpp
        nbody_cuda/src/nbody_cpu.cpp
        nbody_cuda/src/nbody_cuda.cu
        nbody_cuda/src/utils.cpp
)

###################3

# Ruta completa al kernel
set(KERNEL_SRC ${CMAKE_SOURCE_DIR}/nbody_cuda/src/kernel_1.cu)
set(KERNEL_SRC ${CMAKE_SOURCE_DIR}/nbody_cuda/src/kernel_2.cu)
set(KERNEL_PTX ${CMAKE_BINARY_DIR}/kernel_1.ptx)
set(KERNEL_PTX ${CMAKE_BINARY_DIR}/kernel_2.ptx)

# Comando personalizado para generar PTX
add_custom_command(
        OUTPUT ${KERNEL_PTX}
        COMMAND ${CMAKE_CUDA_COMPILER} --ptx ${KERNEL_SRC} -o ${KERNEL_PTX}
        DEPENDS ${KERNEL_SRC}
        COMMENT "Compilando PTX desde kernel_1.cu"
)

# Agrega el archivo PTX como parte del proyecto
add_custom_target(compile_ptx ALL DEPENDS ${KERNEL_PTX})

###################
# Crear el ejecutable
add_executable(cc7515_t2 ${SOURCES})

# Linkear contra la biblioteca del driver (no "cudart", que es para Runtime API)
target_link_libraries(cc7515_t2 PRIVATE cuda)

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

# Asegúrate que se construya el PTX antes del ejecutable
add_dependencies(cc7515_t2 compile_ptx)