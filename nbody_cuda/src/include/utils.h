//
// Created by erick on 5/23/25.
//

#ifndef UTILS_H
#define UTILS_H
#include <cuda_runtime.h>
#include <cuda.h>
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__ )

CUfunction loadKernelSource(const char* filename);
void generateRandomBodies(Body* bodies, int n);
void check(CUresult err, const char* func, const char* file, int line);

#endif //UTILS_H
