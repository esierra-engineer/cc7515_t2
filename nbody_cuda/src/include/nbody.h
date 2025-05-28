//
// Created by erick on 5/8/25.
//
#include <cstdlib>

#ifndef NBODY_H
#define NBODY_H

/**
 * A structure to define a body
 */
struct Body {
    // position
    float x, y, z;
    // velocity
    float vx, vy, vz;
    // mass
    float mass;
};

void simulateNBodyCPU(Body* bodies, int n, int steps, float dt = 0.01f);
void simulateNBodyCUDA(Body* h_bodies,  int steps, float dt, const char* kernelFilename, int localSize, int n);
void generateRandomBodies(Body* bodies, int n);

#endif
