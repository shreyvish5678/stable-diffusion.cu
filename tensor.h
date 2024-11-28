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
    void init(float value = 0.0f, bool if_random = false, unsigned long long seed = 42);    
    float at(int i);
    float at(int i, int j);
    float at(int i, int j, int k);
    float at(int* indices);

    void print();

    Tensor operator+(Tensor& other);
    Tensor operator+(float scalar);
    Tensor operator-(Tensor& other);
    Tensor operator*(Tensor& other);
    Tensor operator*(float scalar);

    static Tensor matmul(Tensor& a, Tensor& b);

    Tensor reshape(int* new_dims, int new_ndims);
    Tensor transpose(int dim1, int dim2);
    static Tensor softmax(Tensor& input);
    static Tensor invsqrt(Tensor& input);

    void save(const std::string& filename);
    void load(const std::string& filename);

private:
    void print_recursive(float* data, int dim, int level);
};

#endif