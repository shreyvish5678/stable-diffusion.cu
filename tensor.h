#ifndef TENSOR_H
#define TENSOR_H

#include <string>

class Tensor {
public:
    float* data_cpu;   
    float* data_gpu;   
    int* dims;      
    int ndims;         
    int size;          

    Tensor();
    Tensor(int* dims, int ndims);

    void allocate_cpu();
    void allocate_gpu();
    void free_memory();
    void init_zero();
    void init_one();
    void init_rand();

    float at(int i, int j);
    float at(int i, int j, int k);
    float at(int* indices);

    void print();

    Tensor operator+(const Tensor& other);
    Tensor operator*(const Tensor& other);
    Tensor operator*(int scalar);

    static Tensor matmul(const Tensor& a, const Tensor& b);

    Tensor reshape(int* new_dims, int new_ndims);
    Tensor transpose(int dim1, int dim2);
    Tensor softmax();

    void save(const std::string& filename);
    void load(const std::string& filename);

private:
    void print_recursive(float* data, int dim, int level);
};

#endif