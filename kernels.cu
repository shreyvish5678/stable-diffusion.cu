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
__global__ void init_zero_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 0;
    }
}

// CUDA kernel to initialize an array with ones
__global__ void init_one_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 1;
    }
}

// CUDA kernel to initialize an array with random values
__global__ void init_rand_kernel(float* data, int size, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state; // State for the random number generator
        curand_init(seed, idx, 0, &state); // Initialize the random number generator
        data[idx] = curand_normal(&state);
    }
}

// CUDA kernel to perform element-wise operations on two arrays
__global__ void ops_kernel(float* result, float* a, float* b, int size, int op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        switch (op) {
            case 0:
                result[idx] = a[idx] + b[idx];
                break;
            case 1:
                result[idx] = a[idx] - b[idx];
                break;
            case 2:
                result[idx] = a[idx] * b[idx];
                break;
            case 3:
                result[idx] = a[idx] / b[idx];
                break;
        }
    }
}

// CUDA kernel to perform element-wise operations on an array and a scalar
__global__ void ops_scalar_kernel(float* result, float* a, float scalar, int size, int op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        switch (op) {
            case 0:
                result[idx] = a[idx] + scalar;
                break;
            case 1:
                result[idx] = a[idx] - scalar;
                break;
            case 2:
                result[idx] = a[idx] * scalar;
                break;
            case 3:
                result[idx] = a[idx] / scalar;
                break;
        }
    }
}

// CUDA kernel to perform element-wise operations on 2d and a 1d tensor
__global__ void ops_bias_kernel(float* result, float* a, float* b, int* a_dims, int* b_dims, int a_rows, int a_cols, int b_cols, int op) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < a_rows && col < b_cols) {
        switch (op) {
            case 0:
                result[row * b_cols + col] = a[row * a_cols + col] + b[col];
                break;
            case 1:
                result[row * b_cols + col] = a[row * a_cols + col] - b[col];
                break;
            case 2:
                result[row * b_cols + col] = a[row * a_cols + col] * b[col];
                break;
            case 3:
                result[row * b_cols + col] = a[row * a_cols + col] / b[col];
                break;
        }
    }
}

// CUDA kernel to perform element-wise operations on 3d and a 1d tensor
__global__ void ops_bias_3d_kernel(float* result, float* a, float* b, int* a_dims, int* b_dims, int a_batch, int a_rows, int a_cols, int b_cols, int op) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < a_rows && col < b_cols) {
        switch (op) {
            case 0:
                result[batch * a_rows * b_cols + row * b_cols + col] = a[batch * a_rows * a_cols + row * a_cols + col] + b[col];
                break;
            case 1:
                result[batch * a_rows * b_cols + row * b_cols + col] = a[batch * a_rows * a_cols + row * a_cols + col] - b[col];
                break;
            case 2:
                result[batch * a_rows * b_cols + row * b_cols + col] = a[batch * a_rows * a_cols + row * a_cols + col] * b[col];
                break;
            case 3:
                result[batch * a_rows * b_cols + row * b_cols + col] = a[batch * a_rows * a_cols + row * a_cols + col] / b[col];
                break;
        }
    }
}

// CUDA kernel to perform element-wise operations on 3d and a 2d tensor
__global__ void ops_channel_kernel(float* result, float* a, float* b, int batch_size, int seq_len, int d_model, int op) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y; 
    int channel = blockIdx.z * blockDim.z + threadIdx.z; 

    if (row < batch_size && col < seq_len && channel < d_model) {
        switch (op) {
            case 0:
                result[channel * batch_size * seq_len + row * seq_len + col] = 
                a[channel * batch_size * seq_len + row * seq_len + col] + b[row * seq_len + col];
                break;
            case 1:
                result[channel * batch_size * seq_len + row * seq_len + col] = 
                a[channel * batch_size * seq_len + row * seq_len + col] - b[row * seq_len + col];
                break;
            case 2:
                result[channel * batch_size * seq_len + row * seq_len + col] = 
                a[channel * batch_size * seq_len + row * seq_len + col] * b[row * seq_len + col];
                break;
            case 3:
                result[channel * batch_size * seq_len + row * seq_len + col] = 
                a[channel * batch_size * seq_len + row * seq_len + col] / b[row * seq_len + col];
                break;
        }
    }
}

// CUDA kernel to perform element-wise operations on 3d and a 2d tensor where the 3d tensor is a batch of 2d tensor
__global__ void ops_batch_kernel(float* result, float* a, float* b, int batch_size, int seq_len, int d_model, int op) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;  
    int col = blockIdx.y * blockDim.y + threadIdx.y; 
    int channel = blockIdx.z * blockDim.z + threadIdx.z; 

    if (row < batch_size && col < seq_len && channel < d_model) {
        switch (op) {
            case 0: 
                result[row * seq_len * d_model + col * d_model + channel] = 
                a[row * seq_len * d_model + col * d_model + channel] + b[col * d_model + channel];
                break;
            case 1: 
                result[row * seq_len * d_model + col * d_model + channel] = 
                a[row * seq_len * d_model + col * d_model + channel] - b[col * d_model + channel];
                break;
            case 2: 
                result[row * seq_len * d_model + col * d_model + channel] = 
                a[row * seq_len * d_model + col * d_model + channel] * b[col * d_model + channel];
                break;
            case 3: 
                result[row * seq_len * d_model + col * d_model + channel] = 
                a[row * seq_len * d_model + col * d_model + channel] / b[col * d_model + channel];
                break;
        }
    }
}


// CUDA kernel to perform matrix multiplication
__global__ void matmul_kernel(float* result, float* a, float* b, int a_rows, int b_rows, int a_cols, int b_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < a_rows && col < b_cols) {
        float sum = 0.0f;
        for (int i = 0; i < a_cols; i++) {
            sum += a[row * a_cols + i] * b[i * b_cols + col];
        }
        result[row * b_cols + col] = sum;
    }
}

// CUDA kernel to perform 3D tensor multiplication
__global__ void matmul_3d_kernel(float* result, float* a, float* b, int a_batch, int a_rows, int b_rows, int a_cols, int b_cols) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < a_rows && col < b_cols) {
        float sum = 0.0f;
        for (int i = 0; i < a_cols; i++) {
            sum += a[batch * a_rows * a_cols + row * a_cols + i] * b[i * b_cols + col];
        }
        result[batch * a_rows * b_cols + row * b_cols + col] = sum;
    }
}

// CUDA kernel to perform 4D tensor multiplication
__global__ void matmul_4d_kernel(float* result, float* a, float* b, int B, int C, int M, int K, int N) {
    int batch = blockIdx.z / C; 
    int channel = blockIdx.z % C; 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += a[batch * C * M * K + channel * M * K + row * K + i] * 
                   b[batch * C * K * N + channel * K * N + i * N + col];
        }
        result[batch * C * M * N + channel * M * N + row * N + col] = sum;
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

// CUDA kernel to compute the mean of a 3D tensor along the last dimension
__global__ void mean_kernel(float* result, float* input, int batch_size, int seq_len, int d_model) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < batch_size && col < seq_len) {
        float sum = 0.0f;
        for (int i = 0; i < d_model; i++) {
            sum += input[row * seq_len * d_model + col * d_model + i];
        }
        result[row * seq_len + col] = sum / d_model;
    }
}

// CUDA kernel to compute the variance of a 3D tensor along the last dimension
__global__ void variance_kernel(float* result, float* input, float* mean, int batch_size, int seq_len, int d_model) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < batch_size && col < seq_len) {
        float sum = 0.0f;
        for (int i = 0; i < d_model; i++) {
            float diff = input[row * seq_len * d_model + col * d_model + i] - mean[row * seq_len + col];
            sum += diff * diff;
        }
        result[row * seq_len + col] = sum / d_model;
    }
}

// CUDA kernel to get square root of a tensor
__global__ void sqrt_kernel(float* result, float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = sqrtf(input[idx]);
    }
}