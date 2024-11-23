#ifndef ATTENTION_H
#define ATTENTION_H

#include "linear.h"
#include "tensor.h"

class Attention {
public:
    int heads;
    int d_embed;
    int d_head;
    Linear q_proj;
    Linear k_proj;
    Linear v_proj;
    Linear out_proj;
    // Constructor
    Attention(int heads, int d_embed, bool proj_bias = true, bool out_bias = true);

    // Forward pass function
    Tensor forward(const Tensor& input, bool mask = false);
};

#endif // ATTENTION_H
