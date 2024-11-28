#include <iostream>
#include <cassert>
#include "kernels.h"
#include <fstream>
#include <string>
#include "tensor.h"

// Default Constructor
Tensor::Tensor() {
    data_cpu = NULL;
    data_gpu = NULL;
    dims = NULL;
    ndims = 0;
    size = 0;
}
// Constructor: Create a tensor with the given properties
Tensor::Tensor(int* dims, int ndims) {
    this->ndims = ndims;
    this->dims = (int*)malloc(ndims * sizeof(int));
    size = 1;
    for (int i = 0; i < ndims; i++) {
        this->dims[i] = dims[i];
        size *= dims[i];
    }
    allocate_cpu();
    allocate_gpu();
}
// Allocate memory on CPU
void Tensor::allocate_cpu() {
    cudaError_t err = cudaMallocHost((void**)&data_cpu, size * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Error allocating memory on CPU" << std::endl;
        exit(1);
    }
}
// Allocate memory on GPU
void Tensor::allocate_gpu() {
    cudaError_t err = cudaMalloc((void**)&data_gpu, size * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Error allocating memory on GPU" << std::endl;
        exit(1);
    }
}
// Destructor
void Tensor::free_memory() {
    cudaFreeHost(data_cpu);
    cudaFree(data_gpu);
}

void Tensor::init(float value, bool if_random, unsigned long long seed) {
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    init_kernel<<<num_blocks, block_size>>>(data_gpu, size, value, if_random, seed);
    cudaDeviceSynchronize();
    CHECK_ERROR();
    cudaMemcpy(data_cpu, data_gpu, size * sizeof(float), cudaMemcpyDeviceToHost);
}

float Tensor::at(int i) {
    if (ndims != 1) {
        throw std::runtime_error("Wrong number of indices");
    }
    return data_cpu[i];
}

// Following methods are used to access elements of the tensor
float Tensor::at(int i, int j) {
    if (ndims != 2) {
        throw std::runtime_error("Wrong number of indices");
    }
    return data_cpu[i * dims[1] + j];
}

float Tensor::at(int i, int j, int k) {
    if (ndims != 3) {
        throw std::runtime_error("Wrong number of indices");
    }
    return data_cpu[i * dims[1] * dims[2] + j * dims[2] + k];
}

float Tensor::at(int* indices) {
    int index = 0;
    for (int i = 0; i < ndims; i++) {
        index = index * dims[i] + indices[i];
    }
    return data_cpu[index];
}

// Following methods are used to print the whole tensor
void Tensor::print() {
    print_recursive(data_cpu, 0, 0);
    std::cout << std::endl;
}

// Overloaded addition operator
Tensor Tensor::operator+(Tensor& other) {
    Tensor result = Tensor(dims, ndims);
    dim3 block_size;
    dim3 num_blocks;
    if (ndims == other.ndims) {
        dim3 block_size(256);
        dim3 num_blocks((size + block_size.x - 1) / block_size.x);
    }
    else if (ndims == 2 && other.ndims == 1) {
        dim3 block_size(16, 16);
        dim3 num_blocks((dims[1] + block_size.x - 1) / block_size.x, (dims[0] + block_size.y - 1) / block_size.y);
    }
    else if (ndims == 3 && other.ndims == 1) {
        dim3 block_size(16, 16);
        dim3 num_blocks((dims[2] + block_size.x - 1) / block_size.x, (dims[1] + block_size.y - 1) / block_size.y, dims[0]);
    }
    else {
        std::cerr << "Invalid dimensions for addition" << std::endl;
        exit(1);
    }
    add_kernel<<<num_blocks, block_size>>>(result.data_gpu, data_gpu, other.data_gpu, result.dims, dims, other.dims, result.ndims, ndims, other.ndims);
    cudaDeviceSynchronize();
    CHECK_ERROR();
    cudaMemcpy(result.data_cpu, result.data_gpu, size * sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}

Tensor Tensor::operator+(float scalar) {
    Tensor result = Tensor(dims, ndims);
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    add_scalar_kernel<<<num_blocks, block_size>>>(result.data_gpu, data_gpu, scalar, size);
    cudaDeviceSynchronize();
    CHECK_ERROR();
    cudaMemcpy(result.data_cpu, result.data_gpu, size * sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}

// Overloaded subtraction operator
Tensor Tensor::operator-(Tensor& other) {
    assert(size == other.size);
    Tensor result = Tensor(dims, ndims);
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    subtract_kernel<<<num_blocks, block_size>>>(result.data_gpu, data_gpu, other.data_gpu, size);
    cudaDeviceSynchronize();
    CHECK_ERROR();
    cudaMemcpy(result.data_cpu, result.data_gpu, size * sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}

// Overloaded multiplication operator
Tensor Tensor::operator*(Tensor& other) {
    assert(size == other.size);

    Tensor result = Tensor(dims, ndims);
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    multiply_kernel<<<num_blocks, block_size>>>(result.data_gpu, data_gpu, other.data_gpu, size);
    cudaDeviceSynchronize();
    CHECK_ERROR();

    cudaMemcpy(result.data_cpu, result.data_gpu, size * sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}

Tensor Tensor::operator*(float scalar) {
    Tensor result = Tensor(dims, ndims);
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    multiply_scalar_kernel<<<num_blocks, block_size>>>(result.data_gpu, data_gpu, scalar, size);
    cudaDeviceSynchronize();
    CHECK_ERROR();

    cudaMemcpy(result.data_cpu, result.data_gpu, size * sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}

// Static method to perform matrix multiplication
Tensor Tensor::matmul(Tensor& a, Tensor& b) {
    Tensor result;
    dim3 block_size;
    dim3 num_blocks;
    if (a.ndims == 2 && b.ndims == 2) {
        assert(a.dims[1] == b.dims[0]);
        int result_dims[] = {a.dims[0], b.dims[1]};
        result = Tensor(result_dims, 2);
        block_size = dim3(16, 16);
        num_blocks = dim3((result_dims[1] + block_size.x - 1) / block_size.x, (result_dims[0] + block_size.y - 1) / block_size.y);
    }
    else if (a.ndims == 3 && b.ndims == 2) {
        assert (a.dims[2] == b.dims[0]);
        int result_dims[] = {a.dims[0], a.dims[1], b.dims[1]};
        result = Tensor(result_dims, 3);
        block_size = dim3(16, 16);
        num_blocks = dim3((result_dims[2] + block_size.x - 1) / block_size.x, (result_dims[1] + block_size.y - 1) / block_size.y, result_dims[0]);
    }
    else if (a.ndims == 4 && b.ndims == 4) {
        assert(a.dims[0] == b.dims[0]);
        assert (a.dims[1] == b.dims[1]);
        assert (a.dims[3] == b.dims[2]);
        int result_dims[] = {a.dims[0], a.dims[1], a.dims[2], b.dims[3]};
        result = Tensor(result_dims, 4);
        block_size = dim3(16, 16);
        num_blocks = dim3((result_dims[3] + block_size.x - 1) / block_size.x, 
                        (result_dims[2] + block_size.y - 1) / block_size.y, 
                        result_dims[0] * result_dims[1]);
    }
    else {
        std::cout << a.ndims << " " << b.ndims << std::endl;
        std::cerr << "Invalid dimensions for matrix multiplication" << std::endl;
        exit(1);
    }
    matmul_kernel<<<num_blocks, block_size>>>(result.data_gpu, a.data_gpu, b.data_gpu, result.dims, a.dims, b.dims, result.ndims, a.ndims, b.ndims);
    cudaDeviceSynchronize();
    CHECK_ERROR();
    cudaMemcpy(result.data_cpu, result.data_gpu, result.size * sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}

// Reshape the tensor
Tensor Tensor::reshape(int* new_dims, int new_ndims) {
    int new_size = 1;
    for (int i = 0; i < new_ndims; i++) {
        new_size *= new_dims[i];
    }
    assert(new_size == size);

    Tensor result = Tensor(new_dims, new_ndims);
    cudaMemcpy(result.data_gpu, data_gpu, size * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(result.data_cpu, result.data_gpu, size * sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}

// Transpose the tensor
Tensor Tensor::transpose(int dim1, int dim2) {
    assert (ndims == 4);
    int new_dims[] = {dims[0], dims[1], dims[2], dims[3]};
    std::swap(new_dims[dim1], new_dims[dim2]);
    Tensor result = Tensor(new_dims, ndims);
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    transpose_kernel<<<grid_size, block_size>>>(result.data_gpu, data_gpu, dims[0], dims[1], dims[2], dims[3], dim1, dim2, size);
    cudaDeviceSynchronize();
    CHECK_ERROR();
    cudaMemcpy(result.data_cpu, result.data_gpu, size * sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}

// Softmax function 
Tensor Tensor::softmax(Tensor& input) {
    assert (input.ndims == 4);
    Tensor result = Tensor(input.dims, input.ndims);
    dim3 block_size(input.dims[0], input.dims[1], input.dims[2]);
    dim3 threads(input.dims[3], 1, 1);
    softmax_kernel<<<block_size, threads>>>(result.data_gpu, input.data_gpu, input.dims[0], input.dims[1], input.dims[2], input.dims[3]);
    cudaDeviceSynchronize();
    CHECK_ERROR();
    cudaMemcpy(result.data_cpu, result.data_gpu, input.size * sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}

// Square root function
Tensor Tensor::invsqrt(Tensor& input) {
    Tensor result = Tensor(input.dims, input.ndims);
    int block_size = 256;
    int num_blocks = (input.size + block_size - 1) / block_size;
    inv_sqrt_kernel<<<num_blocks, block_size>>>(result.data_gpu, input.data_gpu, input.size);
    cudaDeviceSynchronize();
    CHECK_ERROR();
    cudaMemcpy(result.data_cpu, result.data_gpu, input.size * sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}

// Save the tensor to a file
void Tensor::save(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file" << std::endl;
        exit(1);
    }
    int total_size = 1;
    for (int i = 0; i < ndims; ++i) {
        total_size *= dims[i];
    }
    file.write(reinterpret_cast<char*>(data_cpu), total_size * sizeof(float));

    file.close();
}

// Load the tensor from a file
void Tensor::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file" << std::endl;
        exit(1);
    }
    int total_size = 1;
    for (int i = 0; i < ndims; ++i) {
        total_size *= dims[i];
    }
    file.read(reinterpret_cast<char*>(data_cpu), total_size * sizeof(float));

    file.close();

    cudaMemcpy(data_gpu, data_cpu, size * sizeof(float), cudaMemcpyHostToDevice);
}

void Tensor::print_recursive(float* data, int dim_idx, int offset) {
    if (dim_idx == ndims) {
        std::cout << data[offset] << " ";
        return;
    }

    int stride = 1;
    for (int i = dim_idx + 1; i < ndims; i++) {
        stride *= dims[i];
    }

    std::cout << "[";
    for (int i = 0; i < dims[dim_idx]; i++) {
        print_recursive(data, dim_idx + 1, offset + i * stride);
        if (i != dims[dim_idx] - 1) std::cout << " ";
    }
    std::cout << "]";
}