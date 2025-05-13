//
// Created by erick on 5/8/25.
//
#include <iostream>
#include <chrono>
#include <algorithm>
#include "nbody.h"

int main() {
    const int n = 1024*8;
    const int steps = 20;

    Body* bodiesCPU = new Body[n];
    Body* bodiesGPU = new Body[n];

    generateRandomBodies(bodiesCPU, n);
    std::copy(bodiesCPU, bodiesCPU + n, bodiesGPU);

    auto startCPU = std::chrono::high_resolution_clock::now();
    simulateNBodyCPU(bodiesCPU, n, steps);
    auto endCPU = std::chrono::high_resolution_clock::now();
    double timeCPU = std::chrono::duration<double>(endCPU - startCPU).count();

    auto startGPU = std::chrono::high_resolution_clock::now();
    simulateNBodyCUDA(bodiesGPU, n, steps);
    auto endGPU = std::chrono::high_resolution_clock::now();
    double timeGPU = std::chrono::duration<double>(endGPU - startGPU).count();

    std::cout << "CPU time: " << timeCPU << " s\n";
    std::cout << "GPU time: " << timeGPU << " s\n";
    std::cout << "Speedup: " << timeCPU / timeGPU << "x\n";

    delete[] bodiesCPU;
    delete[] bodiesGPU;
    return 0;
}
