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

        // Initialize host arrays with random values
        for (int j = 0; j < size; j++) {
            host_a[j] = rand();
            host_b[j] = rand();
        }

        // Allocate memory on GPU
        cudaMalloc((void**)&device_a, size * sizeof(int));
        cudaMalloc((void**)&device_b, size * sizeof(int));
        cudaMalloc((void**)&device_c, size * sizeof(int));

        // Copy data from host to GPU
        cudaMemcpy(device_a, host_a, size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(device_b, host_b, size * sizeof(int), cudaMemcpyHostToDevice);

        // Profile execution time
        auto start_time = std::chrono::high_resolution_clock::now();

        // Launch the kernel with different scenarios
        // Scenario 1: Using one block with 1 thread
        addKernel<<<1, 1>>>(device_a, device_b, device_c, size);

        // Scenario 2: Using one block with 256 threads
        // addKernel<<<1, 256>>>(device_a, device_b, device_c, size);

        // Scenario 3: Using multiple blocks with 256 threads per block
        // int num_blocks = (size + 255) / 256;
        // addKernel<<<num_blocks, 256>>>(device_a, device_b, device_c, size);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "Time to execute for K = " << K << " million: " << duration.count() << " ms" << std::endl;

        // Copy the result back from GPU to host
        cudaMemcpy(host_c, device_c, size * sizeof(int), cudaMemcpyDeviceToHost);

        // Free memory on GPU
        cudaFree(device_a);
        cudaFree(device_b);
        cudaFree(device_c);

        // Free memory on host
        delete[] host_a;
        delete[] host_b;
        delete[] host_c;
    }

    return 0;
}
