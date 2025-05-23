//
// Created by erick on 5/8/25.
//
#include "include/nbody.h"
#include <cmath>

const float G = 6.67430e-11f;

void simulateNBodyCPU(Body* bodies, int n, int steps, float dt) {
    for (int s = 0; s < steps; ++s) {
        for (int i = 0; i < n; ++i) {
            float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                float dx = bodies[j].x - bodies[i].x;
                float dy = bodies[j].y - bodies[i].y;
                float dz = bodies[j].z - bodies[i].z;
                float distSqr = dx * dx + dy * dy + dz * dz + 1e-10f;
                float invDist = 1.0f / sqrtf(distSqr);
                float invDist3 = invDist * invDist * invDist;
                float F = G * bodies[i].mass * bodies[j].mass * invDist3;
                Fx += F * dx;
                Fy += F * dy;
                Fz += F * dz;
            }
            bodies[i].vx += Fx / bodies[i].mass * dt;
            bodies[i].vy += Fy / bodies[i].mass * dt;
            bodies[i].vz += Fz / bodies[i].mass * dt;
        }
        for (int i = 0; i < n; ++i) {
            bodies[i].x += bodies[i].vx * dt;
            bodies[i].y += bodies[i].vy * dt;
            bodies[i].z += bodies[i].vz * dt;
        }
    }
}
