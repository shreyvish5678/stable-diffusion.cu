#include <cuda_runtime.h>      
#include <curand_kernel.h>       
#include <cmath>           
#include <cstdio>         
#include <float.h>      
#include "kernels.h"

// Check for CUDA errors
void check_cuda_error(const char* file, int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        cudaDeviceReset();
        exit(1);
    }
}

// CUDA kernel to initialize an array with zeros
__global__ void init_kernel(float* data, int size, float value, bool if_random, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (if_random) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        if (idx < size) {
            data[idx] = curand_uniform(&state);
        }
    }
    else {
        if (idx < size) {
            data[idx] = value;
        }
    }
}

// CUDA kernel to add two tensors
__global__ void add_kernel(float* result, float* a, float* b, int* result_dims, int* a_dims, int* b_dims, int result_ndims, int a_ndims, int b_ndims) {
    int result_idx = 0;
    int a_idx = 0;
    int b_idx = 0;
    if (a_ndims == b_ndims) {
        a_idx = blockIdx.x * blockDim.x + threadIdx.x;
        b_idx = blockIdx.x * blockDim.x + threadIdx.x;
        result_idx = blockIdx.x * blockDim.x + threadIdx.x;
    }
    if (b_ndims == 1) {
        if (a_ndims == 3) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            int batch = blockIdx.z;
            a_idx = batch * a_dims[1] * a_dims[2] + row * a_dims[2] + col;
            b_idx = col;
            result_idx = batch * result_dims[1] * result_dims[2] + row * result_dims[2] + col;
        }
        else {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            a_idx = row * a_dims[1] + col;
            b_idx = col;
            result_idx = row * result_dims[1] + col;
        }
    }
    result[result_idx] = a[a_idx] + b[b_idx];
}

// CUDA kernel to add a constant to an array
__global__ void add_scalar_kernel(float* result, float* a, float scalar, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + scalar;
    }
}

// CUDA kernel to subtract two arrays
__global__ void subtract_kernel(float* result, float* a, float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] - b[idx];
    }
}

// CUDA kernel to multiply two arrays element-wise
__global__ void multiply_kernel(float* result, float* a, float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] * b[idx];
    }
}


// CUDA kernel to multiply an array by a scalar
__global__ void multiply_scalar_kernel(float* result, float* a, float scalar, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] * scalar;
    }
}

// CUDA kernel to perform matrix multiplication on two tensors
__global__ void matmul_kernel(float* result, float* a, float* b, int* result_dims, int* a_dims, int* b_dims, int result_ndims, int a_ndims, int b_ndims) {
    int result_idx = 0; 
    int a_idx = 0;
    int b_idx = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int a_rows = a_dims[a_ndims - 2];
    int b_cols = b_dims[b_ndims - 1];
    if (a_ndims == 2 && b_ndims == 2) {
        a_idx = row * a_dims[1];
        b_idx = col;
        result_idx = row * b_cols + col;
    }
    if (a_ndims == 3 && b_ndims == 2) {
        int batch = blockIdx.z;
        a_idx = batch * a_dims[1] * a_dims[2] + row * a_dims[2];
        b_idx = col;
        result_idx = batch * a_dims[1] * b_cols + row * b_cols + col;
    }
    else if (a_ndims == 4 && b_ndims == 4) {
        int batch = blockIdx.z / a_dims[1];
        int channel = blockIdx.z % a_dims[1];
        a_idx = batch * a_dims[1] * a_dims[2] * a_dims[3] + channel * a_dims[2] * a_dims[3] + row * a_dims[3];
        b_idx = batch * b_dims[1] * b_dims[2] * b_dims[3] + channel * b_dims[2] * b_dims[3] + col;  
        result_idx = batch * a_dims[1] * a_dims[2] * b_cols + channel * a_dims[2] * b_cols + row * b_cols + col;
    }
    if (row < a_rows && col < b_cols) {
        float sum = 0.0f;
        for (int i = 0; i < a_dims[a_ndims - 1]; i++) {
            sum += a[a_idx + i] * b[b_idx + i * b_dims[b_ndims - 1]];
        }
        result[result_idx] = sum;
    }
}

// CUDA kernel to perform 4D tensor transposition
__global__ void transpose_kernel(float* result, float* data, int dim0, int dim1, int dim2, int dim3, int swap0, int swap1, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int idx3 = idx % dim3;
        int idx2 = (idx / dim3) % dim2;
        int idx1 = (idx / (dim3 * dim2)) % dim1;
        int idx0 = idx / (dim3 * dim2 * dim1);
        int indices[4] = {idx0, idx1, idx2, idx3};
        int dims[4] = {dim0, dim1, dim2, dim3};
        
        int temp = indices[swap0];
        indices[swap0] = indices[swap1];
        indices[swap1] = temp;

        temp = dims[swap0];
        dims[swap0] = dims[swap1];
        dims[swap1] = temp;

        int output_idx = indices[0] * dims[1] * dims[2] * dims[3] + indices[1] * dims[2] * dims[3] + indices[2] * dims[3] + indices[3];
        result[output_idx] = data[idx];
    }
}

// CUDA kernel to perform softmax along the last dimension of a 4D tensor
__global__ void softmax_kernel(float* output, float* input, int B, int C, int H, int W) {
    int b = blockIdx.x;   
    int c = blockIdx.y; 
    int h = blockIdx.z;
    int w = threadIdx.x; 

    int slice_idx = b * (C * H * W) + c * (H * W) + h * W; 
    int idx = slice_idx + w;                   

    float max_val = -FLT_MAX;
    for (int i = 0; i < W; i++) {
        max_val = fmaxf(max_val, input[slice_idx + i]);
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < W; i++) {
        sum_exp += expf(input[slice_idx + i] - max_val);
    }

    if (w < W) {
        output[idx] = expf(input[idx] - max_val) / sum_exp;
    }
}

// CUDA kernel to compute the inverse square root of an array
__global__ void inv_sqrt_kernel(float* result, float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = 1.0f / sqrtf(input[idx]);
    }
}

// CUDA kernel to apply a mask to the attention weights
__global__ void mask_kernel(float* mask, int batch_size, int seq_len, int heads) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_head = blockIdx.z;
    
    if (row >= seq_len || col >= seq_len) {
        return;
    }
    
    int batch_idx = batch_head / heads;
    int head_idx = batch_head % heads;
    
    int index = batch_idx * (heads * seq_len * seq_len) + head_idx * (seq_len * seq_len) + row * seq_len + col;
    
    if (col > row) {
        mask[index] = -INFINITY;
    }
}

// CUDA kernel for embedding lookup 
__global__ void embedding_lookup_kernel(float* result, float* weight, int* tokens, int batch_size, int seq_len, int d_embed, int vocab_size, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int batch = idx / (seq_len * d_embed);
        int token = (idx / d_embed) % seq_len;
        int dim = idx % d_embed;
        int token_idx = tokens[batch * seq_len + token];
        if (token_idx >= 0 && token_idx < vocab_size) {
            result[idx] = weight[token_idx * d_embed + dim];
        }   
    }
}

// CUDA kernel for mean of 3D tensor on last dimension
__global__ void mean(float *input, float *mean, int rows, int cols, int depth) {
    int row = blockIdx.y;
    int col = blockIdx.x;

    float sum = 0.0f;
    for (int i = 0; i < depth; ++i) {
        int index = row * cols * depth + col * depth + i;
        sum += input[index];
    }

    int mean_index = row * cols + col;
    mean[mean_index] = sum / depth;
}

// CUDA kernel for variance of 3D tensor on last dimension
__global__ void variance(float *input, float *mean, float *variance, int rows, int cols, int depth) {
    int row = blockIdx.y;
    int col = blockIdx.x;

    float var_sum = 0.0f;
    int mean_index = row * cols + col;
    float mean_value = mean[mean_index];

    for (int i = 0; i < depth; ++i) {
        int index = row * cols * depth + col * depth + i;
        float diff = input[index] - mean_value;
        var_sum += diff * diff;
    }

    variance[mean_index] = var_sum / depth;
}
