//
// Created by erick on 5/8/25.
//
#include "include/nbody.h"
#include "include/utils.h"
#include <cuda.h>
#include <iostream>

void check(CUresult err, const char* func, const char* file, int line) {
    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        std::cerr << "CUDA error at " << file << ":" << line << " code=" << err
                  << " \"" << func << "\" : " << errStr << "\n";
        exit(1);
    }
}

void generateRandomBodies(Body* bodies, int n) {
    for (int i = 0; i < n; ++i) {
        bodies[i].x = static_cast<float>(rand()) / RAND_MAX * 100.0f;
        bodies[i].y = static_cast<float>(rand()) / RAND_MAX * 100.0f;
        bodies[i].z = static_cast<float>(rand()) / RAND_MAX * 100.0f;
        bodies[i].vx = bodies[i].vy = bodies[i].vz = 0.0f;
        bodies[i].mass = 1e10f;
    }
}

CUfunction loadKernelSource(const char* filename) {
    checkCudaErrors(cuInit(0));
    CUdevice device;
    checkCudaErrors(cuDeviceGet(&device, 0));
    CUcontext context;
    checkCudaErrors(cuCtxCreate(&context, 0, device));

    // Load PTX file
    CUmodule module;
    checkCudaErrors(cuModuleLoad(&module, filename));

    // Get function
    CUfunction kernel;
    checkCudaErrors(cuModuleGetFunction(&kernel, module, "addKernel"));

    return kernel;
}