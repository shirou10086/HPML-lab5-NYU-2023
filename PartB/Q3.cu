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
        int* unified_memory;

        // Allocate data in unified memory
        cudaMallocManaged(&unified_memory, size * sizeof(int));

        // Initialize data
        for (int j = 0; j < size; j++) {
            unified_memory[j] = rand();
        }

        // Profile execution time
        auto start_time = std::chrono::high_resolution_clock::now();

        // Launch the kernel with different scenarios
        // Scenario 1: Using one block with 1 thread
        // addKernel<<<1, 1>>>(unified_memory, unified_memory, unified_memory, size);

        // Scenario 2: Using one block with 256 threads
        // addKernel<<<1, 256>>>(unified_memory, unified_memory, unified_memory, size);

        // Scenario 3: Using multiple blocks with 256 threads per block
        // int num_blocks = (size + 255) / 256;
        // addKernel<<<num_blocks, 256>>>(unified_memory, unified_memory, unified_memory, size);

        cudaDeviceSynchronize();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "Time to execute for K = " << K << " million: " << duration.count() << " ms" << std::endl;

        // Free unified memory
        cudaFree(unified_memory);
    }

    return 0;
}
