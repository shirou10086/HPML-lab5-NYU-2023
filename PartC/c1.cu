#include <cuda_runtime.h>
#include <stdio.h>

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

// CUDA kernel for convolution without tiling
__global__ void convolutionKernel(const double *I_padded, const double *F_flipped, double *O) {
    int k = blockIdx.z * blockDim.z + threadIdx.z; // Depth
    int x = blockIdx.y * blockDim.y + threadIdx.y; // Row
    int y = blockIdx.x * blockDim.x + threadIdx.x; // Column

    if (k >= K || x >= H_out || y >= W_out) return;

    double result = 0.0;
    for (int c = 0; c < C; c++) {
        for (int i = 0; i < FH; i++) {
            for (int j = 0; j < FW; j++) {
                int ix = x + i;
                int iy = y + j;
                int inputIndex = (c * H_padded + ix) * W_padded + iy;
                int filterIndex = ((k * C + c) * FH + i) * FW + j;
                result += I_padded[inputIndex] * F_flipped[filterIndex];
            }
        }
    }
    O[(k * H_out + x) * W_out + y] = result;
}

// Function to initialize input and filters
void initialize(double *I, double *F) {
    // Initialize the input tensor
    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                I[(c * H + h) * W + w] = c * (h + w);
            }
        }
    }

    // Initialize the filters
    for (int k = 0; k < K; ++k) {
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < FH; ++i) {
                for (int j = 0; j < FW; ++j) {
                    F[((k * C + c) * FH + i) * FW + j] = (c + k) * (i + j);
                }
            }
        }
    }
}

// Main program
int main() {
    double *I = (double *)malloc(C * H * W * sizeof(double));
    double *F = (double *)malloc(K * C * FH * FW * sizeof(double));
    double *O = (double *)malloc(K * H_out * W_out * sizeof(double));

    // Initialize data
    initialize(I, F);

    double *d_I_padded, *d_F_flipped, *d_O;
    cudaMalloc(&d_I_padded, C * H_padded * W_padded * sizeof(double));
    cudaMalloc(&d_F_flipped, K * C * FH * FW * sizeof(double));
    cudaMalloc(&d_O, K * H_out * W_out * sizeof(double));

    // Copy data from host to device
    cudaMemcpy(d_I_padded, I, C * H_padded * W_padded * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_F_flipped, F, K * C * FH * FW * sizeof(double), cudaMemcpyHostToDevice);

    // Define kernel launch parameters
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim((W_out + TILE_WIDTH - 1) / TILE_WIDTH, (H_out + TILE_WIDTH - 1) / TILE_WIDTH, K);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch the kernel
    convolutionKernel<<<gridDim, blockDim>>>(d_I_padded, d_F_flipped, d_O);
    cudaDeviceSynchronize();

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f milliseconds\n", milliseconds);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy result back to host
    cudaMemcpy(O, d_O, K * H_out * W_out * sizeof(double), cudaMemcpyDeviceToHost);

    // Calculate checksum
    double checksum = 0;
    for (int k = 0; k < K; ++k) {
        for (int h = 0; h < H_out; ++h) {
            for (int w = 0; w < W_out; ++w) {
                checksum += O[(k * H_out + h) * W_out + w];
            }
        }
    }
    printf("Checksum: %lf\n", checksum);

    // Free device memory
    cudaFree(d_I_padded);
    cudaFree(d_F_flipped);
    cudaFree(d_O);

    // Free host memory
    free(I);
    free(F);
    free(O);

    return 0;
}
