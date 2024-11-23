#ifndef DIS_CU_H
#define DIS_CU_H

#include <cuda_runtime.h>    

// Check for CUDA errors
void check_cuda_error(const char* file, int line);
#define CHECK_ERROR() check_cuda_error(__FILE__, __LINE__);

// CUDA kernel prototypes
__global__ void init_zero_kernel(float* data, int size);
__global__ void init_one_kernel(float* data, int size);
__global__ void init_rand_kernel(float* data, int size, unsigned long long seed);
__global__ void add_kernel(float* result, float* a, float* b, int size);
__global__ void add_bias_kernel(float* result, float* a, float* b, int* a_dims, int* b_dims, int a_rows, int a_cols, int b_cols);
__global__ void add_bias_3d_kernel(float* result, float* a, float* b, int* a_dims, int* b_dims, int a_batch, int a_rows, int a_cols, int b_cols);
__global__ void multiply_kernel(float* result, float* a, float* b, int size);
__global__ void multiply_scalar_kernel(float* result, float* a, float scalar, int size);
__global__ void matmul_kernel(float* result, float* a, float* b, int a_rows, int b_rows, int a_cols, int b_cols);
__global__ void matmul_3d_kernel(float* result, float* a, float* b, int a_batch, int a_rows, int b_rows, int a_cols, int b_cols);
__global__ void matmul_4d_kernel(float* result, float* a, float* b, int B, int C, int M, int K, int N);
__global__ void transpose_kernel(float* result, float* data, int dim0, int dim1, int dim2, int dim3, int swap0, int swap1, int size);
__global__ void softmax_kernel(float* input, float* output, int B, int C, int H, int W);
__global__ void mask_kernel(float* mask, int batch_size, int seq_len, int heads, int d_head, int mask_value);

#endif 