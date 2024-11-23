#ifndef LINEAR_H
#define LINEAR_H

#include "tensor.h"

class Linear {
public:
    Tensor weights; 
    Tensor bias;     
    bool if_bias;   

    Linear();
    Linear(int in_features, int out_features, bool if_bias = true);

    Tensor forward(const Tensor& input);
};

#endif // LINEAR_H
