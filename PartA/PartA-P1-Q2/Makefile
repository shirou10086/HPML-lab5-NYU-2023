### ECE-GY 9143 - High Performance Computing for Machine Learning
### Instructor: Parijat Dubey
### Makefile for Part A Assignment 5

SDK_INSTALL_PATH := /usr/local/cuda
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB := -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
OPTIONS := -O3

EXECS := vecadd00 vecadd01 matmult00 matmult01
all: $(EXECS)

#######################################################################
clean:
    rm -f $(EXECS) *.o

#######################################################################
timer.o : timer.cu timer.h
    $(NVCC) $< -c -o $@ $(OPTIONS)

#######################################################################
# Vector Addition Kernels
vecaddKernel00.o : vecaddKernel00.cu
    $(NVCC) $< -c -o $@ $(OPTIONS)

vecadd00 : vecadd.cu vecaddKernel00.h vecaddKernel00.o timer.o
    $(NVCC) $< vecaddKernel00.o -o $@ $(LIB) timer.o $(OPTIONS)

vecaddKernel01.o : vecaddKernel01.cu
    $(NVCC) $< -c -o $@ $(OPTIONS)

vecadd01 : vecadd.cu vecaddKernel01.h vecaddKernel01.o timer.o
    $(NVCC) $< vecaddKernel01.o -o $@ $(LIB) timer.o $(OPTIONS)

#######################################################################
# Matrix Multiplication Kernels
matmultKernel00.o : matmultKernel00.cu matmultKernel.h
    $(NVCC) $< -c -o $@ $(OPTIONS)

matmult00 : matmult.cu matmultKernel.h matmultKernel00.o timer.o
    $(NVCC) $< matmultKernel00.o -o $@ $(LIB) timer.o $(OPTIONS)

matmultKernel01.o : matmultKernel01.cu matmultKernel.h
    $(NVCC) $< -c -o $@ $(OPTIONS)

matmult01 : matmult.cu matmultKernel.h matmultKernel01.o timer.o
    $(NVCC) $< matmultKernel01.o -o $@ $(LIB) timer.o $(OPTIONS)
