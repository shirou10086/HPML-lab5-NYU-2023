#include <stdio.h>
#include <time.h>
#include <cudnn.h>

#define H   1024
#define W   1024
#define C   3
#define FH  3
#define FW  3
#define K   64
#define ITER 5

#define checkCUDNN(expression)                                  \
{                                                               \
    cudnnStatus_t status = (expression);                        \
    if (status != CUDNN_STATUS_SUCCESS) {                       \
        printf("cuDNN error on line %d: %s\n", __LINE__,        \
               cudnnGetErrorString(status));                    \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
}

// Function to load image into memory
void loadImageInMem(int h, int w, int c, double *it) {
    for (int ki = 0; ki < c; ++ki) {
        for (int j = 0; j < h; ++j) {
            for (int i = 0; i < w; ++i) {
                it[ki*w*h + j*w + i] = ki * (i+j);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // Allocate host memory
    double *it = (double *)malloc(C * H * W * sizeof(double));
    double *ot = (double *)malloc(K * H * W * sizeof(double));
    double *f = (double *)malloc(K * C * FH * FW * sizeof(double));

    // Allocate device memory
    double *itg, *otg, *gpuf;
    cudaMalloc(&itg, C * H * W * sizeof(double));
    cudaMalloc(&otg, K * H * W * sizeof(double));
    cudaMalloc(&gpuf, K * C * FH * FW * sizeof(double));

    // Initialize host memory
    loadImageInMem(H, W, C, it);

    // Copy input and filter to GPU
    cudaMemcpy(itg, it, C * H * W * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuf, f, K * C * FH * FW * sizeof(double), cudaMemcpyHostToDevice);

    // Set input and output tensor descriptors
    cudnnTensorDescriptor_t input_descriptor, output_descriptor;
    cudnnFilterDescriptor_t kernel_descriptor;
    cudnnConvolutionDescriptor_t convolution_descriptor;

    // Create and set tensor descriptors
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));

    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, C, H, W));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, K, H, W));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, K, C, FH, FW));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor, 1, 1, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_DOUBLE));

    // Find the best convolution algorithm
    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn, input_descriptor, kernel_descriptor, convolution_descriptor, output_descriptor, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &convolution_algorithm));

    // Allocate workspace for cuDNN
    size_t workspace_bytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_descriptor, kernel_descriptor, convolution_descriptor, output_descriptor, convolution_algorithm, &workspace_bytes));
    void *d_workspace = nullptr;
    cudaMalloc(&d_workspace, workspace_bytes);

    // Perform the convolution
    const double alpha = 1.0, beta = 0.0;
    for (int i = 0; i < ITER; ++i) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        checkCUDNN(cudnnConvolutionForward(cudnn, &alpha, input_descriptor, itg, kernel_descriptor, gpuf, convolution_descriptor, convolution_algorithm, d_workspace, workspace_bytes, &beta, output_descriptor, otg));
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Iteration %d: Convolution execution time: %f milliseconds\n", i, milliseconds);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Copy the output back to host
    cudaMemcpy(ot, otg, K * H * W * sizeof(double), cudaMemcpyDeviceToHost);

    // Calculate checksum
    double checksum = 0;
    for (int i = 0; i < K * H * W; ++i) {
        checksum += ot[i];
    }
    printf("Checksum: %lf\n", checksum);

    // Cleanup
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudaFree(itg);
    cudaFree(otg);
    cudaFree(gpuf);
    cudaFree(d_workspace);
    cudnnDestroy(cudnn);

    free(it);
    free(ot);
    free(f);

    return 0;
}
