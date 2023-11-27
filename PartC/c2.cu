#include <cuda_runtime.h>
#include <stdio.h>

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

__global__ void convolutionKernel(const float *I_padded, const float *F_flipped, float *O) {
    int k = blockIdx.z * blockDim.z + threadIdx.z; //depth
    int x = blockIdx.y * blockDim.y + threadIdx.y; //row
    int y = blockIdx.x * blockDim.x + threadIdx.x; //col

    if (k >= K || x >= H_out || y >= W_out) return;

    float result = 0.0f;
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

__global__ void convolutionKernelTiled(const float* I_padded, const float* F, float* O) {
    int k = blockIdx.z * blockDim.z + threadIdx.z; //same as above
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float tile[TILE_WIDTH + FH - 1][TILE_WIDTH + FW - 1];

    for (int c = 0; c < C; ++c) {
        int input_row = x - FH / 2 + c;
        int input_col = y - FW / 2 + c;
        if (input_row >= 0 && input_row < H_padded && input_col >= 0 && input_col < W_padded) {
            tile[threadIdx.y][threadIdx.x] = I_padded[(c * H_padded + input_row) * W_padded + input_col];
        }
        else {
            tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
    }
    __syncthreads();

    float output = 0.0f;
    if (threadIdx.y < TILE_WIDTH && threadIdx.x < TILE_WIDTH) {
        for (int i = 0; i < FH; ++i) {
            for (int j = 0; j < FW; ++j) {
                output += tile[threadIdx.y + i][threadIdx.x + j] * F[k * C * FH * FW + i * FW + j];
            }
        }
        if (x < H_out && y < W_out) {
            O[(k * H_out + x) * W_out + y] = output;
        }
    }
    __syncthreads();
}

void runTiledConvolution(const float *I, const float *F, float *O) {
    float *d_I_padded, *d_F_flipped, *d_O;
    size_t size_I_padded = C * H_padded * W_padded * sizeof(float);
    size_t size_F_flipped = K * C * FH * FW * sizeof(float);
    size_t size_O = K * H_out * W_out * sizeof(float);

    cudaMalloc(&d_I_padded, size_I_padded);
    cudaMalloc(&d_F_flipped, size_F_flipped);
    cudaMalloc(&d_O, size_O);

    float *I_padded = (float *)malloc(size_I_padded);
    float *F_flipped = (float *)malloc(size_F_flipped);

    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H_padded; ++h) {
            for (int w = 0; w < W_padded; ++w) {
                if (h < P || h >= H + P || w < P || w >= W + P) {
                    I_padded[(c * H_padded + h) * W_padded + w] = 0;
                } else {
                    I_padded[(c * H_padded + h) * W_padded + w] = I[(c * H + (h - P)) * W + (w - P)];
                }
            }
        }
    }

    for (int k = 0; k < K; ++k) {
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < FH; ++i) {
                for (int j = 0; j < FW; ++j) {
                    F_flipped[((k * C + c) * FH + i) * FW + j] = F[((k * C + c) * FH + (FH - 1 - i)) * FW + (FW - 1 - j)];
                }
            }
        }
    }

    cudaMemcpy(d_I_padded, I_padded, size_I_padded, cudaMemcpyHostToDevice);
    cudaMemcpy(d_F_flipped, F_flipped, size_F_flipped, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 blocksPerGrid((W_out + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (H_out + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (K + threadsPerBlock.z - 1) / threadsPerBlock.z);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    // Launch the tiled convolution kernel
    convolutionKernelTiled<<<blocksPerGrid, threadsPerBlock>>>(d_I_padded, d_F_flipped, d_O);

    // Record the stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tiled convolution kernel execution time: %f milliseconds\n", milliseconds);

    // Destroy the events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    cudaMemcpy(O, d_O, size_O, cudaMemcpyDeviceToHost);

    cudaFree(d_I_padded);
    cudaFree(d_F_flipped);
    cudaFree(d_O);
}

int main() {
    float *I = (float *)malloc(C * H * W * sizeof(float));
    float *F = (float *)malloc(K * C * FH * FW * sizeof(float));
    float *O = (float *)malloc(K * H_out * W_out * sizeof(float));

    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                I[(c * H + h) * W + w] = c * (h + w);
            }
        }
    }

    for (int k = 0; k < K; ++k) {
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < FH; ++i) {
                for (int j = 0; j < FW; ++j) {
                    F[((k * C + c) * FH + i) * FW + j] = (c + k) * (i + j);
                }
            }
        }
    }

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    int gridX = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int gridY = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;
    dim3 gridDim(gridX, gridY, K);

    runTiledConvolution(I, F, O);

    // Calculate and print the checksum
    float checksum = 0;
    for (int k = 0; k < K; ++k) {
        for (int h = 0; h < H_out; ++h) {
            for (int w = 0; w < W_out; ++w) {
                checksum += O[(k * H_out + h) * W_out + w];
            }
        }
    }
    printf("Checksum of the tiled output tensor O: %f\n", checksum);


    int hold;
    scanf("%d", &hold);

    free(I);
    free(F);
    free(O);

    return 0;
}
