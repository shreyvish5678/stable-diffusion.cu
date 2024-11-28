#include "tensor.h"
#include "linear.h" 

Linear::Linear() {
    if_bias = true;
}

Linear::Linear(int in_features, int out_features, bool if_bias) {
    int dims[] = {in_features, out_features};
    this->if_bias = if_bias;
    weights = Tensor(dims, 2);
    weights.init(0, true, 42);
    int bias_dims[] = {out_features};
    if (if_bias) {
        bias = Tensor(bias_dims, 1);
        bias.init(0, true, 42);
    }
}

Tensor Linear::forward(Tensor& input) {
    if (if_bias) {
        return Tensor::matmul(input, weights) + bias;
    }
    else {
        return Tensor::matmul(input, weights);
    }
}

void Linear::free_memory() {
    weights.free_memory();
    if (if_bias) {
        bias.free_memory();
    }
}