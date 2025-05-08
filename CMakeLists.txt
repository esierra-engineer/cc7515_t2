cmake_minimum_required(VERSION 3.10)
project(nbody_cuda)

set(CMAKE_CXX_STANDARD 11)

add_executable(cc7515_t2
        nbody_cuda/src/main.cpp)
