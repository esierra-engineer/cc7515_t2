//
// kernel_2.cu — optimized with shared memory
//

#include "include/nbody.h"

#define G_CONSTANT 6.67430e-11f
#define NEAR_ZERO 1e-10f

const float G = G_CONSTANT;

extern "C" __global__ void updateBodies(Body* bodies, int n, float dt = 0.01f) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Load current body
    Body bi = bodies[i];

    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

    // Shared memory tile (for coalesced access)
    extern __shared__ Body tile[];

    // Loop through all bodies in tiles
    for (int tileStart = 0; tileStart < n; tileStart += blockDim.x) {
        // Each thread loads one body from global to shared memory
        int j = tileStart + threadIdx.x;
        if (j < n) {
            tile[threadIdx.x] = bodies[j];
        } else {
            tile[threadIdx.x].mass = 0.0f;  // Avoid using uninitialized memory
        }

        __syncthreads(); // Ensure all threads have loaded

        // Compute interaction with all bodies in the tile
        for (int k = 0; k < blockDim.x; ++k) {
            Body bj = tile[k];
            if ((tileStart + k) >= n || i == (tileStart + k)) continue;

            float dx = bj.x - bi.x;
            float dy = bj.y - bi.y;
            float dz = bj.z - bi.z;

            float distSqr = dx * dx + dy * dy + dz * dz + NEAR_ZERO;
            float invDist = rsqrtf(distSqr);
            float F = G * bi.mass * bj.mass * powf(invDist, 3.0f);

            Fx += F * dx;
            Fy += F * dy;
            Fz += F * dz;
        }

        __syncthreads(); // Ensure all threads are done before loading the next tile
    }

    // Update velocity
    bi.vx += Fx / bi.mass * dt;
    bi.vy += Fy / bi.mass * dt;
    bi.vz += Fz / bi.mass * dt;

    // Update position
    bi.x += bi.vx * dt;
    bi.y += bi.vy * dt;
    bi.z += bi.vz * dt;

    bodies[i] = bi;
}
