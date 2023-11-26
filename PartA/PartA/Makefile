### ECE-GY 9143 - High Performance Computing for Machine Learning
### Instructor: Parijat Dubey
### Makefile for Part A Assignment 5

# CUDA SDK path
SDK_INSTALL_PATH := /usr/local/cuda
NVCC := $(SDK_INSTALL_PATH)/bin/nvcc
LIB := -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
OPTIONS := -O3

# Executable names
EXECS := vecadd00

# Default target
all: $(EXECS)

# Clean objects and executables
clean:
	rm -f $(EXECS) *.o

# Compile timer object file
timer.o : timer.cu timer.h
	$(NVCC) $< -c -o $@ $(OPTIONS)

# Compile vecaddKernel00 object file
vecaddKernel00.o : vecaddKernel00.cu
	$(NVCC) $< -c -o $@ $(OPTIONS)

# Link vecadd00 executable
vecadd00 : vecadd.cu vecaddKernel.h vecaddKernel00.o timer.o
	$(NVCC) $< vecaddKernel00.o -o $@ $(LIB) timer.o $(OPTIONS)

# Add a short comment describing the program
# vecadd00: Program to add two vectors using non-coalesced memory access in CUDA.
# It measures the performance impact of different numbers of values per thread.
