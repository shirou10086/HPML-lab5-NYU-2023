#include <stdio.h>
#include <cudnn.h>

#define H   1024
#define W   1024
#define C   3
#define FH  3
#define FW  3
#define K   64

#define checkCUDNN(expression)                                  \
{                                                               \
    cudnnStatus_t status = (expression);                        \
    if (status != CUDNN_STATUS_SUCCESS) {                       \
        printf("cuDNN error on line %d: %s\n", __LINE__,        \
               cudnnGetErrorString(status));                    \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
}

void loadImageInMem(int h, int w, int c, double *it) {
    for (int ki = 0; ki < c; ++ki) {
        for (int j = 0; j < h; ++j) {
            for (int i = 0; i < w; ++i) {
                it[ki * w * h + j * w + i] = ki * (i + j);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    // 创建 cuDNN 句柄
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // 分配主机内存和设备内存
    double *it, *ot, *f;
    double *itg, *otg, *gpuf;

    it = (double *)malloc(C * H * W * sizeof(double));
    ot = (double *)malloc(K * H * W * sizeof(double));
    f = (double *)malloc(K * C * FH * FW * sizeof(double));

    cudaMalloc(&itg, C * H * W * sizeof(double));
    cudaMalloc(&otg, K * H * W * sizeof(double));
    cudaMalloc(&gpuf, K * C * FH * FW * sizeof(double));

    // 初始化主机内存
    loadImageInMem(H, W, C, it);

    // 将输入和滤波器复制到 GPU
    cudaMemcpy(itg, it, C * H * W * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuf, f, K * C * FH * FW * sizeof(double), cudaMemcpyHostToDevice);

    // 设置输入和输出张量描述符、滤波器描述符、卷积描述符
    cudnnTensorDescriptor_t input_descriptor, output_descriptor;
    cudnnFilterDescriptor_t kernel_descriptor;
    cudnnConvolutionDescriptor_t convolution_descriptor;

    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));

    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, C, H, W));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, K, H, W));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, K, C, FH, FW));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor, 1, 1, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_DOUBLE));

    // 查找最佳的卷积算法
    cudnnConvolutionFwdAlgoPerf_t convolution_algorithm_perf;
    int returnedAlgoCount;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnn, input_descriptor, kernel_descriptor, convolution_descriptor, output_descriptor, 1, &returnedAlgoCount, &convolution_algorithm_perf));

    cudnnConvolutionFwdAlgo_t convolution_algorithm = convolution_algorithm_perf.algo;

    // 为 cuDNN 分配工作空间
    size_t workspace_bytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_descriptor, kernel_descriptor, convolution_descriptor, output_descriptor, convolution_algorithm, &workspace_bytes));
    void *d_workspace = nullptr;
    cudaMalloc(&d_workspace, workspace_bytes);

    // 执行卷积操作
    const double alpha = 1.0, beta = 0.0;
    checkCUDNN(cudnnConvolutionForward(cudnn, &alpha, input_descriptor, itg, kernel_descriptor, gpuf, convolution_descriptor, convolution_algorithm, d_workspace, workspace_bytes, &beta, output_descriptor, otg));

    // 将输出复制回主机
    cudaMemcpy(ot, otg, K * H * W * sizeof(double), cudaMemcpyDeviceToHost);

    // 清理资源
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

    // 计算并输出校验和
    double checksum = 0.0;
    for (int i = 0; i < K * H * W; ++i) {
        checksum += ot[i];
    }
    printf("Checksum: %lf\n", checksum);

    return 0;
}
