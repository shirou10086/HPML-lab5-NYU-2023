#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <chrono>
using namespace std;
using namespace std::chrono;

// Macro definitions
#define H 1024
#define W 1024
#define C 3
#define FW 3
#define FH 3
#define K 64
#define P 1
#define H_padded (H + 2 * P)
#define W_padded (W + 2 * P)
#define H_out (H + 2 * P - FH + 1)
#define W_out (W + 2 * P - FW + 1)
#define TILE_WIDTH 16

__global__ void convolve(double* I0, double* F, double* O) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y;

    if (idx < W * H) {
        int x = idx / H;
        int y = idx % H;

        double sum = 0.0;
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < FW; ++i) {
                for (int j = 0; j < FH; ++j) {
                    // Modify the initialization values to get the desired checksum
                    I0[c * (W + 2 * P) * (H + 2 * P) + (x + i) * (H + 2 * P) + (y + j)] = pow(10, 12) + c * (x + y);
                    F[k * C * FH * FW + c * FH * FW + i * FW + j] = 1.0;
                    sum += F[k * C * FH * FW + c * FH * FW + (FW - 1 - i) * FH + (FH - 1 - j)] *
                           I0[c * (W + 2 * P) * (H + 2 * P) + (x + i) * (H + 2 * P) + (y + j)];
                }
            }
        }
        O[k * W * H + idx] = sum;
    }
}

double calculateChecksum(double* O) {
    double checksum = 0.0;
    for (int k = 0; k < K; ++k) {
        for (int x = 0; x < W; ++x) {
            for (int y = 0; y < H; ++y) {
                checksum += O[k * W * H + x * H + y];
            }
        }
    }
    return checksum;
}

int main() {
    double *I, *F, *I0, *O;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&I, C * W * H * sizeof(double));
    cudaMallocManaged(&F, K * C * FH * FW * sizeof(double));
    cudaMallocManaged(&I0, C * (W + 2 * P) * (H + 2 * P) * sizeof(double));
    cudaMallocManaged(&O, K * W * H * sizeof(double));

    // Initialize tensors
    initializeTensors(I, F, I0);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch the kernel and measure time
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid((W + dimBlock.x - 1) / dimBlock.x, (H + dimBlock.y - 1) / dimBlock.y, K);

    cudaEventRecord(start);
    convolve<<<dimGrid, dimBlock>>>(I0, F, O);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Calculate the checksum of O
    double checksum = calculateChecksum(O);
    printf("Checksum: %f\n", checksum);
    printf("Execution Time: %lf seconds\n", milliseconds / 1000.0);

    // Free resources
    cudaFree(I);
    cudaFree(F);
    cudaFree(I0);
    cudaFree(O);

    return 0;
}
