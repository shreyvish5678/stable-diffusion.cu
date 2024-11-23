#ifndef TENSOR_H
#define TENSOR_H

#include <cuda_runtime.h>

class Tensor {
public:
    float* data_cpu;    // Data pointer on the CPU
    float* data_gpu;    // Data pointer on the GPU
    int* dims;          // Tensor dimensions
    int ndims;          // Number of dimensions
    int size;           // Total size of the tensor (number of elements)

    // Default Constructor
    Tensor();

    // Constructor: Create a tensor with the given properties
    Tensor(int* dims, int ndims);

    // Destructor
    ~Tensor();

    // Allocate memory on CPU
    void allocate_cpu();

    // Allocate memory on GPU
    void allocate_gpu();

    // Free allocated memory
    void free_memory();

    // Initialize tensor with zeros
    void init_zero();

    // Initialize tensor with ones
    void init_one();

    // Initialize tensor with random values
    void init_rand();

    // Access elements (2D, 3D, or nD access)
    float at(int i, int j);
    float at(int i, int j, int k);
    float at(int* indices);

    // Print the whole tensor
    void print();

    // Overloaded addition operator
    Tensor operator+(const Tensor& other);

    // Overloaded multiplication operator (element-wise)
    Tensor operator*(const Tensor& other);

    // Overloaded multiplication operator (scalar multiplication)
    Tensor operator*(int scalar);

    // Static method to perform matrix multiplication
    static Tensor matmul(const Tensor& a, const Tensor& b);

    // Reshape the tensor
    Tensor reshape(int* new_dims, int new_ndims);

    // Transpose the tensor (not working properly, see warning)
    Tensor transpose(int dim1, int dim2);

    // Softmax operation
    Tensor softmax();

private:
    // Helper method for printing recursively (internal use)
    void print_recursive(float* data, int dim, int level);
};

#endif // TENSOR_H
