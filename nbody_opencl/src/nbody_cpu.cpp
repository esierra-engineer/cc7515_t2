#include "nbody_cpu.h"

void runNBodyCPU(std::vector<Body>& bodies, int steps, float dt) {
    const float G = 6.67430e-11f;
    int n = bodies.size();

    for (int step = 0; step < steps; ++step) {
        std::vector<Body> newBodies = bodies;

        for (int i = 0; i < n; ++i) {
            float ax = 0, ay = 0, az = 0;

            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                float dx = bodies[j].x - bodies[i].x;
                float dy = bodies[j].y - bodies[i].y;
                float dz = bodies[j].z - bodies[i].z;
                float distSqr = dx*dx + dy*dy + dz*dz + 1e-6f;
                float invDist = 1.0f / sqrt(distSqr);
                float invDist3 = invDist * invDist * invDist;

                float force = G * bodies[j].mass * invDist3;
                ax += force * dx;
                ay += force * dy;
                az += force * dz;
            }

            newBodies[i].vx += ax * dt;
            newBodies[i].vy += ay * dt;
            newBodies[i].vz += az * dt;

            newBodies[i].x += newBodies[i].vx * dt;
            newBodies[i].y += newBodies[i].vy * dt;
            newBodies[i].z += newBodies[i].vz * dt;
        }

        bodies = newBodies;
    }
}
