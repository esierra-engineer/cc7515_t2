#include <CL/cl.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <stdexcept>

#define CHECK_CL(err, msg) if (err != CL_SUCCESS) throw std::runtime_error(msg);

const char* loadKernelSource(const char* filename) {
    std::ifstream file(filename);
    std::string src((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return strdup(src.c_str());
}

struct Body {
    float x, y, z, mass;
    float vx, vy, vz, pad;
};

void runNBodyOpenCL(std::vector<Body>& bodies, int steps, float dt) {
    int n = bodies.size();
    cl_int err;

    // 1. Platform & device
    cl_platform_id platform;
    cl_device_id device;
    err = clGetPlatformIDs(1, &platform, nullptr);
    CHECK_CL(err, "Failed to get platform");

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    CHECK_CL(err, "Failed to get device");

    // 2. Context & queue
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CHECK_CL(err, "Failed to create context");

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_CL(err, "Failed to create command queue");

    // 3. Buffers
    cl_mem posMassBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float4) * n, nullptr, &err);
    cl_mem velBuf     = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float4) * n, nullptr, &err);
    CHECK_CL(err, "Failed to create buffers");

    std::vector<cl_float4> posMass(n), vel(n);
    for (int i = 0; i < n; ++i) {
        posMass[i] = { bodies[i].x, bodies[i].y, bodies[i].z, bodies[i].mass };
        vel[i] = { bodies[i].vx, bodies[i].vy, bodies[i].vz, 0 };
    }

    err = clEnqueueWriteBuffer(queue, posMassBuf, CL_TRUE, 0, sizeof(cl_float4)*n, posMass.data(), 0, nullptr, nullptr);
    err |= clEnqueueWriteBuffer(queue, velBuf,     CL_TRUE, 0, sizeof(cl_float4)*n, vel.data(),     0, nullptr, nullptr);
    CHECK_CL(err, "Failed to write data");

    // 4. Kernel
    const char* source = loadKernelSource("kernel.cl");
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

    // 5. Run
    size_t globalSize = n;

    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < steps; ++step) {
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &posMassBuf);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &velBuf);
        err |= clSetKernelArg(kernel, 2, sizeof(float), &dt);
        err |= clSetKernelArg(kernel, 3, sizeof(int), &n);
        CHECK_CL(err, "Failed to set kernel args");

        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
        CHECK_CL(err, "Kernel launch failed");

        clFinish(queue);
    }

    auto end = std::chrono::high_resolution_clock::now();
    float timeSec = std::chrono::duration<float>(end - start).count();

    std::cout << "OpenCL time: " << timeSec << "s (" << (n * steps) / timeSec << " bodies/sec)\n";

    // 6. Cleanup
    clReleaseMemObject(posMassBuf);
    clReleaseMemObject(velBuf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}
