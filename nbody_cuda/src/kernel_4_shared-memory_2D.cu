//
// Created by erick on 5/28/25.
//
#include "include/nbody.h"

#define G_CONSTANT 6.67430e-11f
#define NEAR_ZERO 1e-10f

const float G = G_CONSTANT;

extern "C" __global__ void updateBodies(Body* bodies, int n, float dt = 0.01f) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bdx = blockDim.x;
    int bdy = blockDim.y;
    int gdx = gridDim.x;

    int threadsPerBlock = bdx * bdy;
    int i = (by * gdx + bx) * threadsPerBlock + (ty * bdx + tx);
    if (i >= n) return;

    Body bi = bodies[i];

    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

    extern __shared__ Body tile[];

    for (int tileStart = 0; tileStart < n; tileStart += threadsPerBlock) {
        int j = tileStart + ty * bdx + tx;
        if (j < n)
            tile[ty * bdx + tx] = bodies[j];
        else
            tile[ty * bdx + tx].mass = 0.0f;

        __syncthreads();

        for (int k = 0; k < threadsPerBlock; ++k) {
            int global_j = tileStart + k;
            if (global_j >= n || global_j == i) continue;

            Body bj = tile[k];

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

        __syncthreads();
    }

    bi.vx += Fx / bi.mass * dt;
    bi.vy += Fy / bi.mass * dt;
    bi.vz += Fz / bi.mass * dt;

    bi.x += bi.vx * dt;
    bi.y += bi.vy * dt;
    bi.z += bi.vz * dt;

    bodies[i] = bi;
}
