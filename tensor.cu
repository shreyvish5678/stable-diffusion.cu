#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "kernels.h"
#include <fstream>

class Tensor {
public:
    float* data_cpu;
    float* data_gpu;
    int* dims;
    int ndims;
    int size;
    // Default Constructor
    Tensor() {
        data_cpu = NULL;
        data_gpu = NULL;
        dims = NULL;
        ndims = 0;
        size = 0;
    }
    // Constructor: Create a tensor with the given properties
    Tensor(int* dims, int ndims) {
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
    void allocate_cpu() {
        cudaError_t err = cudaMallocHost((void**)&data_cpu, size * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "Error allocating memory on CPU" << std::endl;
            exit(1);
        }
    }
    // Allocate memory on GPU
    void allocate_gpu() {
        cudaError_t err = cudaMalloc((void**)&data_gpu, size * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "Error allocating memory on GPU" << std::endl;
            exit(1);
        }
    }
    // Destructor
    void free_memory() {
        cudaFreeHost(data_cpu);
        cudaFree(data_gpu);
    }

    void init_zero() {
        int block_size = 256;
        int num_blocks = (size + block_size - 1) / block_size;
        init_zero_kernel<<<num_blocks, block_size>>>(data_gpu, size);
        cudaDeviceSynchronize();
        CHECK_ERROR();
        cudaMemcpy(data_cpu, data_gpu, size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    void init_one() {
        int block_size = 256;
        int num_blocks = (size + block_size - 1) / block_size;
        init_one_kernel<<<num_blocks, block_size>>>(data_gpu, size);
        cudaDeviceSynchronize();
        CHECK_ERROR();
        cudaMemcpy(data_cpu, data_gpu, size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    void init_rand() {
        int block_size = 256;
        int num_blocks = (size + block_size - 1) / block_size;
        unsigned long long seed = rand();
        init_rand_kernel<<<num_blocks, block_size>>>(data_gpu, size, seed);
        cudaDeviceSynchronize();
        CHECK_ERROR();
        cudaMemcpy(data_cpu, data_gpu, size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // Following methods are used to access elements of the tensor
    float at(int i, int j) {
        if (ndims != 2) {
            throw std::runtime_error("Wrong number of indices");
        }
        return data_cpu[i * dims[1] + j];
    }

    float at(int i, int j, int k) {
        if (ndims != 3) {
            throw std::runtime_error("Wrong number of indices");
        }
        return data_cpu[i * dims[1] * dims[2] + j * dims[2] + k];
    }

    float at(int* indices) {
        int index = 0;
        for (int i = 0; i < ndims; i++) {
            index = index * dims[i] + indices[i];
        }
        return data_cpu[index];
    }

    // Following methods are used to print the whole tensor
    void print() {
        print_recursive(data_cpu, 0, 0);
        std::cout << std::endl;
    }

    // Overloaded addition operator
    Tensor operator+(const Tensor& other) {
        if (size == other.size) {
            Tensor result = Tensor(dims, ndims);
            int block_size = 256;
            int num_blocks = (size + block_size - 1) / block_size;
            add_kernel<<<num_blocks, block_size>>>(result.data_gpu, data_gpu, other.data_gpu, size);
            cudaDeviceSynchronize();
            CHECK_ERROR();
            cudaMemcpy(result.data_cpu, result.data_gpu, size * sizeof(float), cudaMemcpyDeviceToHost);
            return result;
        }
        else if (dims[ndims - 1] == other.dims[other.ndims - 1]) {
            Tensor result = Tensor(dims, ndims);
            if (ndims == 2 && other.ndims == 1) {
                dim3 block_size(16, 16);
                dim3 num_blocks((dims[1] + block_size.x - 1) / block_size.x, (dims[0] + block_size.y - 1) / block_size.y);
                add_bias_kernel<<<num_blocks, block_size>>>(result.data_gpu, data_gpu, other.data_gpu, dims, other.dims, dims[0], dims[1], other.dims[0]);
                cudaDeviceSynchronize();
                CHECK_ERROR();
                cudaMemcpy(result.data_cpu, result.data_gpu, size * sizeof(float), cudaMemcpyDeviceToHost);
                return result;
            }
            else if (ndims == 3 && other.ndims == 1) {
                dim3 block_size(16, 16);
                dim3 num_blocks((dims[2] + block_size.x - 1) / block_size.x, (dims[1] + block_size.y - 1) / block_size.y, dims[0]);
                add_bias_3d_kernel<<<num_blocks, block_size>>>(result.data_gpu, data_gpu, other.data_gpu, dims, other.dims, dims[0], dims[1], dims[2], other.dims[0]);
                cudaDeviceSynchronize();
                CHECK_ERROR();
                cudaMemcpy(result.data_cpu, result.data_gpu, size * sizeof(float), cudaMemcpyDeviceToHost);
                return result;
            }
            else {
                std::cerr << "Invalid dimensions for addition" << std::endl;
                exit(1);
            }
        }
        else {
            std::cerr << "Invalid dimensions for addition" << std::endl;
            exit(1);
        }
    }

    // Overloaded multiplication operator
    Tensor operator*(const Tensor& other) {
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

    Tensor operator*(int scalar) {
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
    static Tensor matmul(const Tensor& a, const Tensor& b) {
        if (a.ndims == 2 && b.ndims == 2) {
            assert(a.dims[1] == b.dims[0]);

            int result_dims[] = {a.dims[0], b.dims[1]};
            Tensor result = Tensor(result_dims, 2);

            dim3 block_size(16, 16);
            dim3 num_blocks((result_dims[1] + block_size.x - 1) / block_size.x, (result_dims[0] + block_size.y - 1) / block_size.y);
            matmul_kernel<<<num_blocks, block_size>>>(result.data_gpu, a.data_gpu, b.data_gpu, a.dims[0], b.dims[0], a.dims[1], b.dims[1]);
            cudaDeviceSynchronize();
            CHECK_ERROR();

            cudaMemcpy(result.data_cpu, result.data_gpu, result.size * sizeof(float), cudaMemcpyDeviceToHost);
            return result;
        }
        else if (a.ndims == 3 && b.ndims == 2) {
            assert (a.dims[2] == b.dims[0]);

            int result_dims[] = {a.dims[0], a.dims[1], b.dims[1]};
            Tensor result = Tensor(result_dims, 3);

            dim3 block_size(16, 16);
            dim3 num_blocks((result_dims[2] + block_size.x - 1) / block_size.x, (result_dims[1] + block_size.y - 1) / block_size.y, result_dims[0]);

            matmul_3d_kernel<<<num_blocks, block_size>>>(result.data_gpu, a.data_gpu, b.data_gpu, a.dims[0], a.dims[1], b.dims[0], a.dims[2], b.dims[1]);
            cudaDeviceSynchronize();
            CHECK_ERROR();

            cudaMemcpy(result.data_cpu, result.data_gpu, result.size * sizeof(float), cudaMemcpyDeviceToHost);
            return result;
        }
        else if (a.ndims == 4 && b.ndims == 4) {
            assert(a.dims[0] == b.dims[0]);
            assert (a.dims[1] == b.dims[1]);
            assert (a.dims[3] == b.dims[2]);
            int result_dims[] = {a.dims[0], a.dims[1], a.dims[2], b.dims[2]};
            Tensor result = Tensor(result_dims, 4);
            dim3 block_size(16, 16);
            dim3 num_blocks((result_dims[3] + block_size.x - 1) / block_size.x, 
                            (result_dims[2] + block_size.y - 1) / block_size.y, 
                            result_dims[0] * result_dims[1]);

            matmul_4d_kernel<<<num_blocks, block_size>>>(
                result.data_gpu, a.data_gpu, b.data_gpu, 
                a.dims[0], a.dims[1], a.dims[2], a.dims[3], b.dims[3]
            );

            cudaDeviceSynchronize();
            CHECK_ERROR();

            cudaMemcpy(result.data_cpu, result.data_gpu, result.size * sizeof(float), cudaMemcpyDeviceToHost);
            return result;
        }
        else {
            std::cout << a.ndims << " " << b.ndims << std::endl;
            std::cerr << "Invalid dimensions for matrix multiplication" << std::endl;
            exit(1);
        }
    }

    // Reshape the tensor
    Tensor reshape(int* new_dims, int new_ndims) {
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
    // WARNING: This method is not working properly
    Tensor transpose(int dim1, int dim2) {
        int* perm = new int[ndims];
        for (int i = 0; i < ndims; i++) {
            perm[i] = i;
        }
        std::swap(perm[dim1], perm[dim2]);
        
        int* new_dims = new int[ndims];
        for (int i = 0; i < ndims; i++) {
            new_dims[i] = dims[i];
        }
        std::swap(new_dims[dim1], new_dims[dim2]);
        
        int* strides = new int[ndims];
        strides[ndims - 1] = 1;  
        for (int i = ndims - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * dims[i + 1];
        }

        int total_size = size; 
        Tensor result = Tensor(new_dims, ndims);
        int block_size = 256;
        int num_blocks = (total_size + block_size - 1) / block_size;
        transpose_kernel<<<num_blocks, block_size>>>(result.data_gpu, data_gpu, dims, perm, strides, total_size, ndims);
        cudaDeviceSynchronize();
        CHECK_ERROR();
        cudaMemcpy(result.data_cpu, result.data_gpu, total_size * sizeof(float), cudaMemcpyDeviceToHost);
        
        delete[] perm; 
        delete[] new_dims;
        delete[] strides;
        return result;
    }

    // Softmax function 
    Tensor softmax() {
        assert (ndims == 4);
        Tensor result = Tensor(dims, ndims);
        dim3 block_size(dims[0], dims[1], dims[2]);
        dim3 threads(dims[3], 1, 1);
        softmax_kernel<<<block_size, threads>>>(result.data_gpu, data_gpu, dims[0], dims[1], dims[2], dims[3]);
        cudaDeviceSynchronize();
        CHECK_ERROR();
        cudaMemcpy(result.data_cpu, result.data_gpu, size * sizeof(float), cudaMemcpyDeviceToHost);
        return result;
    }

    // Save the tensor to a file
    void save(const std::string& filename) {
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
    void load(const std::string& filename) {
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

private:
    void print_recursive(float* data, int dim_idx, int offset) {
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
};