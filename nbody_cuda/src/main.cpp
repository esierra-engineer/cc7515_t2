//
// Created by erick on 5/8/25.
//
#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include "include/nbody.h"

void benchmarkCPU(Body* bodies, int n, int steps, float dt, std::ofstream& out) {
    std::cout << "\n[CPU] Simulando con n = " << n << ", steps = " << steps << "...\n";

    auto start = std::chrono::high_resolution_clock::now();
    simulateNBodyCPU(bodies, n, steps, dt);
    auto end = std::chrono::high_resolution_clock::now();
    float cpuTime = std::chrono::duration<float>(end - start).count();
    std::cout << "[CPU] Time: " << cpuTime << "s\n";

    out << n << "," << steps << ",CPU,-," << cpuTime << "\n";
}

void benchmarkGPU(int n, int steps, float dt, std::ofstream& out,
                const std::string& kernelName, int localSize,
                Body* gpuBodies) {
    std::cout << "\n[GPU] Simulando con n = " << n
            << ", steps = " << steps
            << ", kernel = " << kernelName
            << ", localSize = " << localSize << "...\n";

    //std::vector<Body> gpuBodies = originalBodies;

    try {
        auto start = std::chrono::high_resolution_clock::now();
        simulateNBodyCUDA(gpuBodies, steps, dt, kernelName.c_str(), localSize, n);
        auto end = std::chrono::high_resolution_clock::now();
        float gpuTime = std::chrono::duration<float>(end - start).count();

        out << n << "," << steps << "," << kernelName << "," << localSize << "," << gpuTime << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Falló experimento con kernel=" << kernelName
                << ", localSize=" << localSize << ", n=" << n
                << ", steps=" << steps << ": " << e.what() << "\n";
        out << n << "," << steps << "," << kernelName << "," << localSize << ",ERROR\n";
    }

}

int main() {
    const float dt = 0.01f;

//    std::vector<int> sizes = {128};
    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};

    std::vector<int> stepsList = {10, 100, 1000};
    std::vector<int> localSizes = {32, 64, 70, 96, 100, 128};

    std::vector<std::string> kernels = {
        "kernel_1_global-memory_1D.ptx",
        "kernel_2_shared-memory_1D.ptx",
        "kernel_3_global-memory_2D.ptx",
        "kernel_4_shared-memory_2D.ptx",
    };

    std::ofstream out("/media/storage/git/cc7515_t2/nbody_cuda/CUDA_resultados.csv");
    out << "n,steps,method,localSize,time\n";

    // === CPU ===

    for (int steps : stepsList) {
        for (int n : sizes) {
            Body* bodiesCPU = new Body[n];
            generateRandomBodies(bodiesCPU, n);
            benchmarkCPU(bodiesCPU, n, steps, dt, out);
            delete[] bodiesCPU;
        }
    }

    // === GPU por kernel ===
    for (const std::string& kernel : kernels) {
        for (int steps : stepsList) {
            for (int n : sizes) {
                for (int localSize : localSizes) {
                    Body* bodiesGPU = new Body[n];
                    generateRandomBodies(bodiesGPU, n);
                    benchmarkGPU(n, steps, dt, out, kernel, localSize, bodiesGPU);
                    delete[] bodiesGPU;
                }
            }
        }
    }

    out.close();
    std::cout << "\nResultados guardados en resultados.csv\n";
    return 0;
}
