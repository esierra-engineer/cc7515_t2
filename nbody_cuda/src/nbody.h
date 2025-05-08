//
// Created by erick on 5/8/25.
//

#ifndef NBODY_H
#define NBODY_H

struct Body {
    float x, y, z;
    float vx, vy, vz;
    float mass;
};

void simulateNBodyCPU(Body* bodies, int n, int steps);
void simulateNBodyCUDA(Body* bodies, int n, int steps);
void generateRandomBodies(Body* bodies, int n);

#endif
