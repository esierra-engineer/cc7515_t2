//
// Created by erick on 5/8/25.
//
#include "include/nbody.h"
#include <cstdlib>

void generateRandomBodies(Body* bodies, int n) {
    for (int i = 0; i < n; ++i) {
        bodies[i].x = static_cast<float>(rand()) / RAND_MAX * 100.0f;
        bodies[i].y = static_cast<float>(rand()) / RAND_MAX * 100.0f;
        bodies[i].z = static_cast<float>(rand()) / RAND_MAX * 100.0f;
        bodies[i].vx = bodies[i].vy = bodies[i].vz = 0.0f;
        bodies[i].mass = 1e10f;
    }
}
