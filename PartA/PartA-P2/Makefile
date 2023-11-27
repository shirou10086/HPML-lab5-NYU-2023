SDK_INSTALL_PATH := /usr/local/cuda
NVCC = $(SDK_INSTALL_PATH)/bin/nvcc
LIB = -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
OPTIONS = -O3
TAR_FILE_NAME = YourNameCUDA1.tar
EXECS = matmult00  # 修改为matmult00

all: $(EXECS)

clean:
	rm -f $(EXECS) *.o

tar:
	tar -cvf $(TAR_FILE_NAME) Makefile *.h *.cu *.pdf *.txt

timer.o: timer.cu timer.h
	${NVCC} $< -c -o $@ $(OPTIONS)

matmultKernel00.o: matmultKernel00.cu matmultKernel.h  # 修改为matmultKernel00.o
	${NVCC} $< -c -o $@ $(OPTIONS)

matmult00: matmult.cu matmultKernel.h matmultKernel00.o timer.o  # 修改为matmult00
	${NVCC} $< matmultKernel00.o -o $@ $(LIB) timer.o $(OPTIONS)
