#ifndef ATTENTION_H
#define ATTENTION_H

#include "linear.h"
#include "tensor.h"

class SelfAttention {
public:
    int heads;
    int d_embed;
    int d_head;
    Linear q_proj;
    Linear k_proj;
    Linear v_proj;
    Linear out_proj;
    SelfAttention();
    SelfAttention(int heads, int d_embed, bool proj_bias = true, bool out_bias = true);

    Tensor forward(Tensor& input, bool mask = false);
    void free_memory();
};

class CrossAttention {
public:
    int heads;
    int d_embed;
    int d_head;
    int d_cross;
    Linear q_proj;
    Linear k_proj;
    Linear v_proj;
    Linear out_proj;
    CrossAttention();
    CrossAttention(int heads, int d_embed, int d_cross, bool proj_bias = true, bool out_bias = true);

    Tensor forward(Tensor& input, Tensor& context);
};

#endif 