#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>

__global__ void matMul(float *A, float *B, float *C, int M, int N, int K, int D) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int depth = blockIdx.z * blockDim.z + threadIdx.z;
    if (row < M && col < N && depth < D) {
        float sum = 0;
        for (int i = 0; i < K; i++) {
            sum += A[(depth * M + row) * K + i] * B[depth * (i * N + col)];
        }
        C[(depth * M + row) * N + col] = sum;
    }
}

__global__ void addBias(float *C, float *bias, int M, int N, int D) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int depth = blockIdx.z * blockDim.z + threadIdx.z;
    if (row < M && col < N && depth < D) {
        C[(depth * M + row) * N + col] += bias[depth * N + col];
    }
}

float * mlp(
    float* input,
    float* weight,
    float* bias,
    float* output,
    const int batch_size,
    const int depth,
    const int input_size,
    const int output_size,
    bool log = false
) {
    const int M = batch_size;
    const int N = output_size;
    const int K = input_size;
    const int D = depth;

    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_bias;
    cudaMalloc(&d_A, M * D * K * sizeof(float));
    cudaMalloc(&d_B, D * K * N * sizeof(float));
    cudaMalloc(&d_C, M * D * N * sizeof(float));
    cudaMalloc(&d_bias, D * N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_A, input, M * D * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, weight, D * K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, D * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block size
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y, (D + blockSize.z - 1) / blockSize.z);

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start); // Start timing

    // Launch kernels
    matMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K, D);
    addBias<<<gridSize, blockSize>>>(d_C, d_bias, M, N, D);

    // Stop, sync and copy data back
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_C, M * D * N * sizeof(float), cudaMemcpyDeviceToHost);

    if (log) {
        // Print result
        for (int d = 0; d < D; d++) {
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    printf("%f ", output[(d * M + i) * N + j]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_bias);

    // Free host memory
    free(input);
    free(weight);
    free(bias);

    // Calculate elapsed time
    float elapsed_time = 0.0f;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    if (log) {
        std::cout << "Elapsed time: " << elapsed_time << " ms" << std::endl; // Print elapsed time
    }

    return output;
}

int main(int argc, char* argv[]) {
    const int M = 256; // batch size
    const int D = 3; // depth
    const int K = 1024; // input size
    const int N = 512; // output size

    bool log = false;
    if (argc > 1) {
        log = true;
    }

    // Allocate memory
    float *A = (float *)malloc(M * D * K * sizeof(float));
    float *B = (float *)malloc(D * K * N * sizeof(float));
    float *output = (float *)malloc(M * D * N * sizeof(float));
    float *bias = (float *)malloc(D * N * sizeof(float));

    // Initialize A, B and bias
    for (int i = 0; i < M * D * K; i++) {
        A[i] = 1.0;
    }
    for (int i = 0; i < D * K * N; i++) {
        B[i] = 2.0;
    }
    for (int i = 0; i < D * N; i++) {
        bias[i] = -1.0;
    }

    // Call mlp
    output = mlp(A, B, bias, output, M, D, K, N, log);

    // Free host memory
    free(output);

    return 0;
}
