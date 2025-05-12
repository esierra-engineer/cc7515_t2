#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include "body.h"
#include "nbody_cpu.h"
#include "nbody_opencl.h"

void benchmarkNBody(int n, int steps, float dt, std::ofstream& out) {
    std::cout << "\nSimulando con n = " << n << " cuerpos...\n";

    // Inicializar cuerpos
    std::vector<Body> originalBodies(n);
    for (int i = 0; i < n; ++i) {
        originalBodies[i] = {
            float(rand() % 1000), float(rand() % 1000), float(rand() % 1000), 1e12f,
            0, 0, 0, 0
        };
    }

    // CPU
    std::vector<Body> cpuBodies = originalBodies;
    auto cpuStart = std::chrono::high_resolution_clock::now();
    runNBodyCPU(cpuBodies, steps, dt);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    float cpuTime = std::chrono::duration<float>(cpuEnd - cpuStart).count();
    std::cout << "[CPU] Time: " << cpuTime << "s\n";

    // GPU
    std::vector<Body> gpuBodies = originalBodies;
    auto gpuStart = std::chrono::high_resolution_clock::now();
    runNBodyOpenCL(gpuBodies, steps, dt);
    auto gpuEnd = std::chrono::high_resolution_clock::now();
    float gpuTime = std::chrono::duration<float>(gpuEnd - gpuStart).count();

    // Guardar en CSV
    out << n << "," << steps << "," << cpuTime << "," << gpuTime << "\n";
}

int main() {
    const float dt = 0.01f;
    const int steps = 100;

    std::ofstream out("resultados.csv");
    out << "n,steps,cpu_time,gpu_time\n";

    std::vector<int> sizes = {256, 512, 1024, 2048};

    for (int n : sizes) {
        benchmarkNBody(n, steps, dt, out);
    }

    out.close();
    std::cout << "\nResultados guardados en resultados.csv\n";
    return 0;
}
