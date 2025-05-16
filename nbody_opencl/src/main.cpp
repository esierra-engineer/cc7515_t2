#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include "body.h"
#include "nbody_cpu.h"
#include "nbody_opencl.h"

void benchmarkCPU(int n, int steps, float dt, std::ofstream& out) {
    std::cout << "\n[CPU] Simulando con n = " << n << ", steps = " << steps << "...\n";

    std::vector<Body> bodies(n);
    for (int i = 0; i < n; ++i) {
        bodies[i] = {
            float(rand() % 1000), float(rand() % 1000), float(rand() % 1000), 1e12f,
            0, 0, 0, 0
        };
    }

    auto start = std::chrono::high_resolution_clock::now();
    runNBodyCPU(bodies, steps, dt);
    auto end = std::chrono::high_resolution_clock::now();
    float cpuTime = std::chrono::duration<float>(end - start).count();
    std::cout << "[CPU] Time: " << cpuTime << "s\n";

    out << n << "," << steps << ",CPU,-," << cpuTime << "\n";
}

void benchmarkGPU(int n, int steps, float dt, std::ofstream& out,
                const std::string& kernelName, size_t localSize,
                const std::vector<Body>& originalBodies) {
    std::cout << "\n[GPU] Simulando con n = " << n
            << ", steps = " << steps
            << ", kernel = " << kernelName
            << ", localSize = " << localSize << "...\n";

    std::vector<Body> gpuBodies = originalBodies;

    try {
        auto start = std::chrono::high_resolution_clock::now();
        runNBodyOpenCL(gpuBodies, steps, dt, kernelName.c_str(), localSize);
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

    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};
    std::vector<int> stepsList = {10, 100, 1000};
    std::vector<std::string> kernels = {
        "kernel.cl", "kernel_2d.cl", "kernel_local.cl"
    };
    std::vector<size_t> localSizes = {32, 64, 70, 96, 100, 128};

    std::ofstream out("resultados.csv");
    out << "n,steps,method,localSize,time\n";

    // === CPU ===
    for (int steps : stepsList) {
        for (int n : sizes) {
            benchmarkCPU(n, steps, dt, out);
        }
    }

    // === GPU por kernel ===
    for (const std::string& kernel : kernels) {
        for (int steps : stepsList) {
            for (int n : sizes) {
                std::vector<Body> originalBodies(n);
                for (int i = 0; i < n; ++i) {
                    originalBodies[i] = {
                        float(rand() % 1000), float(rand() % 1000), float(rand() % 1000), 1e12f,
                        0, 0, 0, 0
                    };
                }

                for (size_t localSize : localSizes) {
                    benchmarkGPU(n, steps, dt, out, kernel, localSize, originalBodies);
                }
            }
        }
    }

    out.close();
    std::cout << "\nResultados guardados en resultados.csv\n";
    return 0;
}
