#pragma once
#include <vector>
#include "body.h"

void runNBodyOpenCL(std::vector<Body>& bodies, int steps, float dt);