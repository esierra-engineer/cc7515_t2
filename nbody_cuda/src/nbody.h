//
// Created by erick on 5/8/25.
//

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

void simulateNBodyCPU(Body* bodies, int n, int steps);
void simulateNBodyCUDA(Body* bodies, int n, int steps);
void generateRandomBodies(Body* bodies, int n);

#endif
