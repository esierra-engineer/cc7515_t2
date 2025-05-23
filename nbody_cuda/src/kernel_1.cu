//
// Created by erick on 5/23/25.
// kernel_1.cu
//

#include "include/nbody.h"
#define G_CONSTANT 6.67430e-11f
#define NEAR_ZERO 1e-10f

// universal gravitational constant
const float G = G_CONSTANT;

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