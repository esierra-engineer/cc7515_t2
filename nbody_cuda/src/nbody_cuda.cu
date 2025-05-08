//
// Created by erick on 5/8/25.
//
#include "nbody.h"
#include <cuda_runtime.h>
#include <cmath>

const float G = 6.67430e-11f;
const float dt = 0.01f;

__global__ void updateBodies(Body* bodies, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
    Body bi = bodies[i];

    for (int j = 0; j < n; ++j) {
        if (i == j) continue;
        Body bj = bodies[j];
        float dx = bj.x - bi.x;
        float dy = bj.y - bi.y;
        float dz = bj.z - bi.z;
        float distSqr = dx * dx + dy * dy + dz * dz + 1e-10f;
        float invDist = rsqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;
        float F = G * bi.mass * bj.mass * invDist3;
        Fx += F * dx;
        Fy += F * dy;
        Fz += F * dz;
    }

    bi.vx += Fx / bi.mass * dt;
    bi.vy += Fy / bi.mass * dt;
    bi.vz += Fz / bi.mass * dt;

    bi.x += bi.vx * dt;
    bi.y += bi.vy * dt;
    bi.z += bi.vz * dt;

    bodies[i] = bi;
}

void simulateNBodyCUDA(Body* h_bodies, int n, int steps) {
    Body* d_bodies;
    size_t size = n * sizeof(Body);
    cudaMalloc(&d_bodies, size);
    cudaMemcpy(d_bodies, h_bodies, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    for (int s = 0; s < steps; ++s) {
        updateBodies<<<numBlocks, blockSize>>>(d_bodies, n);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_bodies, d_bodies, size, cudaMemcpyDeviceToHost);
    cudaFree(d_bodies);
}
