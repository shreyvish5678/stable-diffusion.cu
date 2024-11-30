#include "tensor.h"
#include "kernels.h"
#include "layernorm.h"

LayerNorm::LayerNorm() {
    d_model = 0;
    eps = 1e-5;
}   

LayerNorm::LayerNorm(int d_model, float eps) {
    this->d_model = d_model;
    this->eps = eps;
    gamma = Tensor(new int[1]{d_model}, 1);
    beta = Tensor(new int[1]{d_model}, 1);
    gamma.init_rand();    
    beta.init_rand();
}

Tensor LayerNorm::forward(Tensor& input) {
    Tensor mean = Tensor::mean(input);
    Tensor vareps = Tensor::variance(input, mean) + eps;
    Tensor rstd = Tensor::sqrt(vareps);
    Tensor out = ((input - mean) / rstd) * gamma + beta;
    return out;
}

void LayerNorm::free_memory() {
    gamma.free_memory();
    beta.free_memory();
}