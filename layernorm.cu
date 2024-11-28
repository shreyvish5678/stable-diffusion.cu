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
    gamma.init(0, true, 42);
    beta.init(0, true, 42);
}

Tensor LayerNorm::forward(Tensor& input) {
    /*
    Tensor mean = Tensor::mean(input);
    Tensor vareps = Tensor::var(input) + eps;
    Tensor shifted = (input - mean) * Tensor::invsqrt(vareps);
    Tensor out = shifted * gamma + beta;
    return out;
    */
   return Tensor();
}