#ifndef LAYERNORM_H
#define LAYERNORM_H

#include "tensor.h"

class LayerNorm {
public:
    float eps;
    int d_model;
    Tensor gamma;
    Tensor beta;

    LayerNorm();

    LayerNorm(int d_model, float eps = 1e-5);

    Tensor forward(Tensor& input);

    void free_memory();

};

#endif 