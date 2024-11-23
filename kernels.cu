#include <cuda_runtime.h>      
#include <curand_kernel.h>       
#include <cmath>           
#include <cstdio>         
#include <float.h>      

// Check for CUDA errors
void check_cuda_error(const char* file, int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        cudaDeviceReset();
        exit(1);
    }
}

#define CHECK_ERROR() check_cuda_error(__FILE__, __LINE__)

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
        data[idx] = (curand(&state) % 100) / 100.0f;
    }
}


// CUDA kernel to add two arrays
__global__ void add_kernel(float* result, float* a, float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

// CUDA kernel to add a bias vector to a matrix
__global__ void add_bias_kernel(float* result, float* a, float* b, int* a_dims, int* b_dims, int a_rows, int a_cols, int b_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < a_rows && col < b_cols) {
        result[row * b_cols + col] = a[row * a_cols + col] + b[col];
    }
}

// CUDA kernel to add a bias vector to a 3D tensor
__global__ void add_bias_3d_kernel(float* result, float* a, float* b, int* a_dims, int* b_dims, int a_batch, int a_rows, int a_cols, int b_cols) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < a_rows && col < b_cols) {
        result[batch * a_rows * b_cols + row * b_cols + col] = a[batch * a_rows * a_cols + row * a_cols + col] + b[col];
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
__global__ void multiply_scalar_kernel(float* result, float* a, int scalar, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] * scalar;
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


// CUDA kernel to perform 3D tensor multiplication
// WARNING: This kernel is not working properly
__global__ void transpose_kernel(float* result, float* data, int* dims, int* perm, int* strides, int total_size, int ndims) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        int indices[4];  
        int rem = idx;  
        for (int i = 0; i < ndims; i++) {
            indices[i] = rem / strides[i];
            rem = rem % strides[i];
        }
        int new_idx = 0;
        for (int i = 0; i < ndims; i++) {
            new_idx = new_idx * dims[perm[i]] + indices[perm[i]];
        }
        printf("here\n");
        printf("idx: %d, indices: [%d, %d, %d, %d], new_idx: %d\n", 
        idx, indices[0], indices[1], indices[2], indices[3], new_idx);
        result[new_idx] = data[idx];
    }
}

// CUDA kernel to perform softmax along the last dimension of a 4D tensor
__global__ void softmax_kernel(float* input, float* output, int B, int C, int H, int W) {
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
__global__ void mask_kernel(float* mask, int batch_size, int seq_len, int heads, int d_head, int mask_value) {
    int b = blockIdx.x; 
    int d = blockIdx.y;
    int row = threadIdx.y + blockIdx.z * blockDim.y; 
    int col = threadIdx.x;  

    if (b < batch_size && d < seq_len && row < heads && col < d_head) {
        if (col < row + mask_value) { 
            mask[b * seq_len * heads * d_head + d * heads * d_head + row * d_head + col] = -FLT_MAX;
        }
    }
}