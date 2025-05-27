//
// Created by erick on 5/8/25.
//
#include "include/nbody.h"
#include <cuda_runtime.h>
#include "include/utils.h"

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
        //updateBodies<<<numBlocks, threadsPerBlock>>>(d_bodies,

        // Kernel args deben ser punteros a los datos
        void* kernelArgs[] = {
            (void*) &d_bodies,
            (void*) &n,
            (void*) &dt
        };

        checkCudaErrors(
            cuLaunchKernel(kernel,
            numBlocks, 1, 1,                    // grid
            threadsPerBlock, 1, 1,             // block
            0, nullptr,                        // shared memory and stream
            kernelArgs, nullptr)                // args
            );

        cudaDeviceSynchronize();
    }

    // retrieve the updated positions and velocities
    cudaMemcpy(h_bodies, d_bodies, size, cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_bodies);
}