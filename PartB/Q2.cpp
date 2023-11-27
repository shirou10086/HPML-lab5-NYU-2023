#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

// CUDA内核函数
__global__ void addKernel(int* a, int* b, int* c, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int K_values[] = {1, 5, 10, 50, 100};

    for (int i = 0; i < 5; i++) {
        int K = K_values[i];
        int size = K * 1000000;
        int* host_a = new int[size];
        int* host_b = new int[size];
        int* host_c = new int[size];
        int* device_a;
        int* device_b;
        int* device_c;

        // 初始化主机数组

        // 分配GPU上的内存
        cudaMalloc((void**)&device_a, size * sizeof(int));
        cudaMalloc((void**)&device_b, size * sizeof(int));
        cudaMalloc((void**)&device_c, size * sizeof(int));

        // 将数据从主机复制到GPU
        cudaMemcpy(device_a, host_a, size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(device_b, host_b, size * sizeof(int), cudaMemcpyHostToDevice);

        // 计时
        auto start_time = std::chrono::high_resolution_clock::now();

        // 启动CUDA内核，使用不同的场景
        int num_blocks = (size + 255) / 256;
        addKernel<<<num_blocks, 256>>>(device_a, device_b, device_c, size);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "K = " << K << " million, Execution Time: " << duration.count() << " ms" << std::endl;

        // 将结果从GPU复制回主机

        // 释放GPU上的内存
        cudaFree(device_a);
        cudaFree(device_b);
        cudaFree(device_c);

        // 释放主机上的内存
        delete[] host_a;
        delete[] host_b;
        delete[] host_c;
    }

    return 0;
}
