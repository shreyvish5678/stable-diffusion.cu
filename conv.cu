#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

__global__ void conv2dkernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int batch_size,
    const int input_channels,
    const int output_channels,
    const int input_size,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_size
) {
    CUDA_KERNEL_LOOP(index, batch_size * output_channels * output_size * output_size) {
        const int w_out = index % output_size;
        const int h_out = (index / output_size) % output_size;
        const int c_out = (index / (output_size * output_size)) % output_channels;
        const int b = index / (output_channels * output_size * output_size);

        float sum = 0.0f;
        for (int c_in = 0; c_in < input_channels; ++c_in) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int h_in = h_out * stride - padding + kh;
                    int w_in = w_out * stride - padding + kw;

                    if (h_in >= 0 && h_in < input_size && w_in >= 0 && w_in < input_size) {
                        int input_index = ((b * input_channels + c_in) * input_size + h_in) * input_size + w_in;
                        int weight_index = ((c_out * input_channels + c_in) * kernel_size + kh) * kernel_size + kw;
                        sum += input[input_index] * weight[weight_index];
                    }
                }
            }
        }
        int output_index = ((b * output_channels + c_out) * output_size + h_out) * output_size + w_out;
        output[output_index] = sum + bias[c_out];
    }
}

float * conv2d(
    float* input,
    float* weight,
    float* bias,
    float* output,
    const int batch_size,
    const int input_channels,
    const int output_channels,
    const int input_size,
    const int kernel_size,
    const int stride,
    const int padding,
    bool log = false
) {
    const int output_size = (input_size + 2 * padding - kernel_size) / stride + 1;
    const int num_kernels = batch_size * output_channels * output_size * output_size;

    // Allocate device memory
    float *d_input, *d_weight, *d_bias, *d_output;
    cudaMalloc(&d_input, batch_size * input_channels * input_size * input_size * sizeof(float));
    cudaMalloc(&d_weight, output_channels * input_channels * kernel_size * kernel_size * sizeof(float));
    cudaMalloc(&d_bias, output_channels * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_channels * output_size * output_size * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, input, batch_size * input_channels * input_size * input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, output_channels * input_channels * kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, output_channels * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block size
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel
    conv2dkernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
        d_input, d_weight, d_bias, d_output,
        batch_size, input_channels, output_channels, input_size,
        kernel_size, stride, padding,
        output_size
    );

    // Copy data back to host
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, batch_size * output_channels * output_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);

    if (log) {
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                printf("%f ", output[i * output_size + j]);
            }
            printf("\n");
        }
    }

    // Calculate elapsed time
    float elapsed_time = 0.0f;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    if (log) {
        std::cout << "Elapsed time: " << elapsed_time << " ms" << std::endl;
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);

    // Free host memory
    free(input);
    free(weight);
    free(bias);

    return output;
}

int main(int argc, char* argv[]) {
    const int batch_size = 1;
    const int input_channels = 3;
    const int output_channels = 64;
    const int input_size = 224;
    const int kernel_size = 3;
    const int stride = 1;
    const int padding = 1;

    const int output_size = (input_size + 2 * padding - kernel_size) / stride + 1;

    bool log = false;
    if (argc > 1) {
        log = true;
    }

    const int input_elements = batch_size * input_channels * input_size * input_size;
    const int weight_elements = output_channels * input_channels * kernel_size * kernel_size;
    const int bias_elements = output_channels;
    const int output_elements = batch_size * output_channels * output_size * output_size;

    // Allocate memory
    float* h_input = (float *)malloc(input_elements * sizeof(float));
    float* h_weight = (float *)malloc(weight_elements * sizeof(float));
    float* h_bias = (float *)malloc(bias_elements * sizeof(float));
    float* h_output = (float *)malloc(output_elements * sizeof(float));

    for (int i = 0; i < input_elements; i++) {
        h_input[i] = 1.0f;
    }
    for (int i = 0; i < weight_elements; i++) {
        h_weight[i] = 1.0f;
    }
    for (int i = 0; i < bias_elements; i++) {
        h_bias[i] = 1.0f;
    }

    h_output = conv2d(h_input, h_weight, h_bias, h_output, batch_size, input_channels, output_channels, input_size, kernel_size, stride, padding, log);

    free(h_output);

    return 0;
}
