//
// Created by erick on 5/8/25.
//
#include "include/nbody.h"
#include <cuda_runtime.h>
#include "include/utils.h"
#define G_CONSTANT 6.67430e-11f
#define NEAR_ZERO 1e-10f
#define BLOCK_SIZE (32 * 8)

// universal gravitational constant
const float G = G_CONSTANT;


/**
 * CUDA kernel. Uses global memory
 * bodies: pointer to bodies array
 * n number of bodies
 * **/
__global__ void updateBodies(Body* bodies, int n, float dt = 0.01f) {
    // i is the body index (global thread index),
    // each thread handles ONE BODY
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // index can go no longer than the number of bodies
    if (i >= n) return;

    // for this body
    Body bi = bodies[i];

    // border conditions, initial net force is null
    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

    // for each other body
    for (int j = 0; j < n; ++j) {
        // skip self
        if (i == j) continue;
        // this other body (global memory access here)
        Body bj = bodies[j];

        // the distance between bodies in x, y and z
        float dx = bj.x - bi.x;
        float dy = bj.y - bi.y;
        float dz = bj.z - bi.z;

        // euclidean distance (avoid division by zero by adding a small constant)
        float distSqr = dx * dx + dy * dy + dz * dz + NEAR_ZERO;
        // inverse of the distance
        float invDist = rsqrtf(distSqr);

        // Newton's gravity, vectorial form
        float F = G * bi.mass * bj.mass * powf(invDist, 3.0f);

        // update net force over body for x,y,z
        Fx += F * dx;
        Fy += F * dy;
        Fz += F * dz;
    }

    /** update velocity
     * if (F = m * a) and (a =  dv/dt)
     * then (F = m * dv/dt)
     * then (dv = F * dt / m)
     * then v = v + dv
     * **/
    bi.vx += Fx / bi.mass * dt;
    bi.vy += Fy / bi.mass * dt;
    bi.vz += Fz / bi.mass * dt;

    /** update position
     * v = dx/dt
     * dx = dv * dt
     * x = x + dx
     **/
    bi.x += bi.vx * dt;
    bi.y += bi.vy * dt;
    bi.z += bi.vz * dt;

    // store the body back into GLOBAL MEMORY
    bodies[i] = bi;
}

/**
* CUDA kernel. Uses shared memory
 * @param bodies pointer to bodies array
 * @param n number of bodies
 */
__global__ void updateBodiesUsingSharedMemory(Body* bodies, int n, float dt = 0.01f) {
    // i is the body index (global thread index),
    // each thread handles ONE BODY
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // index can go no longer than the number of bodies
    if (i >= n) return;

    // for this body
    Body bi = bodies[i];

    // border conditions, initial net force is null
    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

    // for each other body
    for (int j = 0; j < n; ++j) {
        // skip self
        if (i == j) continue;
        // this other body (global memory access here)
        Body bj = bodies[j];

        // the distance between bodies in x, y and z
        float dx = bj.x - bi.x;
        float dy = bj.y - bi.y;
        float dz = bj.z - bi.z;

        // euclidean distance (avoid division by zero by adding a small constant)
        float distSqr = dx * dx + dy * dy + dz * dz + NEAR_ZERO;
        // inverse of the distance
        float invDist = rsqrtf(distSqr);

        // Newton's gravity, vectorial form
        float F = G * bi.mass * bj.mass * powf(invDist, 3.0f);

        // update net force over body for x,y,z
        Fx += F * dx;
        Fy += F * dy;
        Fz += F * dz;
    }

    /** update velocity
     * if (F = m * a) and (a =  dv/dt)
     * then (F = m * dv/dt)
     * then (dv = F * dt / m)
     * then v = v + dv
     * **/
    bi.vx += Fx / bi.mass * dt;
    bi.vy += Fy / bi.mass * dt;
    bi.vz += Fz / bi.mass * dt;

    /** update position
     * v = dx/dt
     * dx = dv * dt
     * x = x + dx
     **/
    bi.x += bi.vx * dt;
    bi.y += bi.vy * dt;
    bi.z += bi.vz * dt;

    // store the body back into GLOBAL MEMORY
    bodies[i] = bi;
}

/**
 * Host Function (CPU-side)
 * Copy the input data from host memory to device memory, also known as host-to-device transfer.
 * Load the GPU program and execute, caching data on-chip for performance.
 * Copy the results from device memory to host memory, also called device-to-host transfer.
 * source: https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/

 * @param h_bodies
 * @param n number of bodies
 * @param steps simulation steps
 */
void simulateNBodyCUDA(Body* h_bodies,  int steps, float dt, const char* kernelFilename, size_t localSize, int n) {
    // destination memory address pointer
    Body* d_bodies;
    // in memory size of n bodies
    size_t size = n * sizeof(Body);

    // allocate GPU memory
    cudaMalloc(&d_bodies, size);

    // copy data between host and device
    cudaMemcpy(d_bodies, h_bodies, size, cudaMemcpyHostToDevice);

    // configure threads per block
    size_t threadsPerBlock = localSize;
    // The total number of blocks is the data size divided by the size of each block
    size_t numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Kernel
    CUfunction kernel = loadKernelSource(kernelFilename);

    // for each step
    for (int s = 0; s < steps; ++s) {
        // kernel launch
        //updateBodies<<<numBlocks, threadsPerBlock>>>(d_bodies, n);
        // Kernel args deben ser punteros a los datos
        void* kernelArgs[] = {
            (void*)&d_bodies,
            (void*)&n,
            (void*)&dt
        };

        checkCudaErrors(
            cuLaunchKernel(kernel,
            numBlocks, 1, 1,                    // grid
            threadsPerBlock, 1, 1,             // block
            0, nullptr,                        // shared memory and stream
            kernelArgs, nullptr)                // args
            );

        // necesary to exchange info between streams
        cudaDeviceSynchronize();
    }

    // retrieve the updated positions and velocities
    cudaMemcpy(h_bodies, d_bodies, size, cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_bodies);
}