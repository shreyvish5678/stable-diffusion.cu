#ifndef LINEAR_H
#define LINEAR_H

#include "tensor.h"

class Linear {
public:
    Tensor weights;  // Weight matrix
    Tensor bias;     // Bias vector (optional)
    bool if_bias;    // Flag to determine if bias is used

    // Default constructor
    Linear();

    // Parameterized constructor
    Linear(int in_features, int out_features, bool if_bias = true);

    // Forward pass method
    Tensor forward(const Tensor& input);
};

#endif // LINEAR_H
