################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= galaxy

# Cuda source files (compiled with cudacc)
CUFILES		:= galaxy_kernel.cu

# C/C++ source files (compiled with gcc / c++)
CCFILES		:= \
	galaxy.cpp ParticleRenderer.cpp\

USEGLLIB	:= 1
USEPARAMGL	:= 1
USEGLUT		:= 1

################################################################################
# Rules and targets

include common.mk
