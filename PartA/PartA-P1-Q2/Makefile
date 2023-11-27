### ECE-GY 9143 - High Performance Computing for Machine Learning
### Instructor: Parijat Dubey
### Makefile for Part A Assignment 5

SDK_INSTALL_PATH := /usr/local/cuda
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB := -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
OPTIONS := -O3

EXECS := vecadd01
all: $(EXECS)

#######################################################################
clean:
	rm -f $(EXECS) *.o

#######################################################################
tar:
	tar -cvf YourNameCUDA1.tar Makefile *.h *.cu *.pdf *.txt vecaddKernel01.cu

#######################################################################
timer.o : timer.cu timer.h
	$(NVCC) $< -c -o $@ $(OPTIONS)

#######################################################################
# Vector Addition Kernel for vecadd01
vecaddKernel01.o : vecaddKernel01.cu
	$(NVCC) $< -c -o $@ $(OPTIONS)

vecadd01 : vecadd.cu vecaddKernel01.h vecaddKernel01.o timer.o
	$(NVCC) $< vecaddKernel01.o -o $@ $(LIB) timer.o $(OPTIONS)
