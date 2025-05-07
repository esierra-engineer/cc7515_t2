#include <iostream>
#include <vector>
#include <cstdlib>
#include "nbody_opencl.cpp"

int main() {
    int n = 1024;
    int steps = 100;
    float dt = 0.01f;

    std::vector<Body> bodies(n);
    for (int i = 0; i < n; ++i) {
        bodies[i] = {
            float(rand() % 1000), float(rand() % 1000), float(rand() % 1000), 1e12f,
            0, 0, 0, 0
        };
    }

    runNBodyOpenCL(bodies, steps, dt);
    return 0;
}
