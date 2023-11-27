#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

#define H 1024
#define W 1024
#define C 3
#define FW 3
#define FH 3
#define K 64
#define P 1

__global__ void convolve(double* I0, double* F, double* O) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < W && y < H && k < K) {
        double sum = 0.0;
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < FW; ++i) {
                for (int j = 0; j < FH; ++j) {
                    sum += F[k * C * FH * FW + c * FH * FW + (FW - 1 - i) * FH + (FH - 1 - j)] *
                           I0[c * (W + 2 * P) * (H + 2 * P) + (x + i) * (H + 2 * P) + (y + j)];
                }
            }
        }
        O[k * W * H + x * H + y] = sum;
    }
}

void initializeTensors(double* I, double* F, double* I0) {
    // Initialize I
    for (int c = 0; c < C; ++c) {
        for (int x = 0; x < W; ++x) {
            for (int y = 0; y < H; ++y) {
                I[c * W * H + x * H + y] = c * (x + y);
            }
        }
    }

    // Initialize F
    for (int k = 0; k < K; ++k) {
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < FH; ++i) {
                for (int j = 0; j < FW; ++j) {
                    F[k * C * FH * FW + c * FH * FW + i * FW + j] = (c + k) * (i + j);
                }
            }
        }
    }

    // Initialize I0 with padding
    for (int c = 0; c < C; ++c) {
        for (int x = 0; x < W + 2 * P; ++x) {
            for (int y = 0; y < H + 2 * P; ++y) {
                if (x == 0 || y == 0 || x == W + 2 * P - 1 || y == H + 2 * P - 1) {
                    I0[c * (W + 2 * P) * (H + 2 * P) + x * (H + 2 * P) + y] = 0;
                } else {
                    I0[c * (W + 2 * P) * (H + 2 * P) + x * (H + 2 * P) + y] = I[c * W * H + (x - 1) * H + (y - 1)];
                }
            }
        }
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

    // Launch the kernel
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid((W + dimBlock.x - 1) / dimBlock.x, (H + dimBlock.y - 1) / dimBlock.y, K);

    auto start = high_resolution_clock::now();
    convolve<<<dimGrid, dimBlock>>>(I0, F, O);
    cudaDeviceSynchronize();
    auto stop = high_resolution_clock::now();

    // Calculate the checksum of O
    double checksum = calculateChecksum(O);
    cout << "Checksum: " << checksum << endl;

    // Report execution time
    duration<double> kernelDuration = duration_cast<duration<double>>(stop - start);
    cout << "Execution Time: " << kernelDuration.count() << " seconds" << endl;

    // Free resources
    cudaFree(I);
    cudaFree(F);
    cudaFree(I0);
    cudaFree(O);

    return 0;
}
