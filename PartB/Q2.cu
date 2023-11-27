#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

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

        for (int j = 0; j < size; j++) {
            host_a[j] = 1; // 或者任何其他的初始化值
            host_b[j] = 2; // 或者任何其他的初始化值
        }

        cudaMalloc((void**)&device_a, size * sizeof(int));
        cudaMalloc((void**)&device_b, size * sizeof(int));
        cudaMalloc((void**)&device_c, size * sizeof(int));

        cudaMemcpy(device_a, host_a, size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(device_b, host_b, size * sizeof(int), cudaMemcpyHostToDevice);

        // Scenario 1: One block with 1 thread
        int num_blocks = 1;
        int num_threads_per_block = 1;
        auto start = std::chrono::high_resolution_clock::now();
        addKernel<<<num_blocks, num_threads_per_block>>>(device_a, device_b, device_c, size);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "K = " << K << ", 1 Block, 1 Thread, Time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

        // Scenario 2: One block with 256 threads
        num_blocks = 1;
        num_threads_per_block = 256;
        start = std::chrono::high_resolution_clock::now();
        addKernel<<<num_blocks, num_threads_per_block>>>(device_a, device_b, device_c, size);
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        std::cout << "K = " << K << ", 1 Block, 256 Threads, Time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

        // Scenario 3: Multiple blocks with 256 threads per block
        num_threads_per_block = 256;
        num_blocks = (size + num_threads_per_block - 1) / num_threads_per_block;
        start = std::chrono::high_resolution_clock::now();
        addKernel<<<num_blocks, num_threads_per_block>>>(device_a, device_b, device_c, size);
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        std::cout << "K = " << K << ", " << num_blocks << " Blocks, 256 Threads/Block, Time: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " ms" << std::endl;

        cudaMemcpy(host_c, device_c, size * sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(device_a);
        cudaFree(device_b);
        cudaFree(device_c);

        delete[] host_a;
        delete[] host_b;
        delete[] host_c;
    }

    return 0;
}
