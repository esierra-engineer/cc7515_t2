#include "nbody_opencl.h"
#include <CL/cl.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <cstring>

#define CHECK_CL(err, msg) if (err != CL_SUCCESS) throw std::runtime_error(msg);

const char* loadKernelSource(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[OpenCL] Failed to open kernel file: " << filename << std::endl;
        return nullptr;
    }
    std::string src((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return _strdup(src.c_str());
}

void runNBodyOpenCL(std::vector<Body>& bodies, int steps, float dt, const char* kernelFilename, size_t localSize) {
    int n = bodies.size();
    cl_int err;

    // Plataforma y device
    cl_platform_id platform;
    cl_device_id device;
    err = clGetPlatformIDs(1, &platform, nullptr);
    CHECK_CL(err, "Failed to get platform");

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    CHECK_CL(err, "Failed to get device");

    // Context y cola
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CHECK_CL(err, "Failed to create context");

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_CL(err, "Failed to create command queue");

    // Buffers
    cl_mem posMassBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float4) * n, nullptr, &err);
    cl_mem velBuf     = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float4) * n, nullptr, &err);
    CHECK_CL(err, "Failed to create buffers");

    std::vector<cl_float4> posMass(n), vel(n);
    for (int i = 0; i < n; ++i) {
        posMass[i] = { bodies[i].x, bodies[i].y, bodies[i].z, bodies[i].mass };
        vel[i] = { bodies[i].vx, bodies[i].vy, bodies[i].vz, 0 };
    }

    err = clEnqueueWriteBuffer(queue, posMassBuf, CL_TRUE, 0, sizeof(cl_float4)*n, posMass.data(), 0, nullptr, nullptr);
    err |= clEnqueueWriteBuffer(queue, velBuf,     CL_TRUE, 0, sizeof(cl_float4)*n, vel.data(),     0, nullptr, nullptr);
    CHECK_CL(err, "Failed to write data");

    // Kernel
    const char* source = loadKernelSource(kernelFilename);
    if (!source) throw std::runtime_error("Could not load kernel.cl");
    std::cout << "[OpenCL] Kernel source loaded.\n";

    cl_program program = clCreateProgramWithSource(context, 1, &source, nullptr, &err);
    CHECK_CL(err, "Failed to create program");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        char log[2048];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        std::cerr << "Build error:\n" << log << std::endl;
        throw std::runtime_error("Failed to build program");
    }

    cl_kernel kernel = clCreateKernel(program, "nbody_step", &err);
    CHECK_CL(err, "Failed to create kernel");

    // Correr
    size_t globalSize = ((n + localSize - 1) / localSize) * localSize;

    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < steps; ++step) {
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &posMassBuf);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &velBuf);
        err |= clSetKernelArg(kernel, 2, sizeof(float), &dt);
        err |= clSetKernelArg(kernel, 3, sizeof(int), &n);

        if (std::string(kernelFilename).find("2d_local") != std::string::npos) {
            err |= clSetKernelArg(kernel, 4, localSize * sizeof(cl_float4), nullptr);
        }

        CHECK_CL(err, "Failed to set kernel args");

        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, nullptr);
        CHECK_CL(err, "Kernel launch failed");

        clFinish(queue);
    }

    auto end = std::chrono::high_resolution_clock::now();
    float timeSec = std::chrono::duration<float>(end - start).count();

    std::cout << "[OpenCL] Time: " << timeSec << "s (" << (n * steps) / timeSec << " bodies/sec)\n";

    // Limpiar
    clReleaseMemObject(posMassBuf);
    clReleaseMemObject(velBuf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}
