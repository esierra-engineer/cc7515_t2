//
// Created by erick on 5/28/25.
//
#include "include/nbody.h"

#define G_CONSTANT 6.67430e-11f
#define NEAR_ZERO 1e-10f

const float G = G_CONSTANT;

extern "C" __global__ void updateBodies(Body* bodies, int n, float dt = 0.01f) {
    // Índices 2D
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bdx = blockDim.x;
    int bdy = blockDim.y;
    int gdx = gridDim.x;

    // Índice global 1D desde 2D
    int i = (by * gdx + bx) * (bdx * bdy) + (ty * bdx + tx);

    if (i >= n) return;

    Body bi = bodies[i];

    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

    for (int j = 0; j < n; ++j) {
        if (i == j) continue;

        Body bj = bodies[j];

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

    bi.vx += Fx / bi.mass * dt;
    bi.vy += Fy / bi.mass * dt;
    bi.vz += Fz / bi.mass * dt;

    bi.x += bi.vx * dt;
    bi.y += bi.vy * dt;
    bi.z += bi.vz * dt;

    bodies[i] = bi;
}
