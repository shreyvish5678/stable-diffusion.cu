#include "tensor.h"

class Linear {
public:
    Tensor weights;
    Tensor bias;
    bool if_bias;
    Linear() {
        if_bias = true;
    }
    Linear(int in_features, int out_features, bool if_bias = true) {
        int dims[] = {in_features, out_features};
        this->if_bias = if_bias;
        weights = Tensor(dims, 2);
        weights.init_rand();
        int bias_dims[] = {out_features};
        if (if_bias) {
            bias = Tensor(bias_dims, 1);
            bias.init_rand();
        }
    }

    Tensor forward(const Tensor& input) {
        if (if_bias) {
            return Tensor::matmul(input, weights) + bias;
        }
        else {
            return Tensor::matmul(input, weights);
        }
    }
};