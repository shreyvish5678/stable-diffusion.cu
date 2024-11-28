#include "kernels.h"
#include "tensor.h"
#include "linear.h"
#include "attention.h"

SelfAttention::SelfAttention() {
    heads = 0;
    d_embed = 0;
    d_head = 0;
}

SelfAttention::SelfAttention(int heads, int d_embed, bool proj_bias, bool out_bias) {
    q_proj = Linear(d_embed, d_embed, proj_bias);
    k_proj = Linear(d_embed, d_embed, proj_bias);
    v_proj = Linear(d_embed, d_embed, proj_bias);
    out_proj = Linear(d_embed, d_embed, out_bias);
    this->heads = heads;
    this->d_embed = d_embed;
    this->d_head = d_embed / heads;
}

Tensor SelfAttention::forward(Tensor& input, bool mask) {
    int batch_size = input.dims[0];
    int seq_len = input.dims[1];
    int interim_dims[] = {batch_size, seq_len, heads, d_head};
    Tensor q = q_proj.forward(input).reshape(interim_dims, 4).transpose(1, 2);
    Tensor k = k_proj.forward(input).reshape(interim_dims, 4).transpose(1, 2).transpose(2, 3);
    Tensor v = v_proj.forward(input).reshape(interim_dims, 4).transpose(1, 2);
    Tensor weight = Tensor::matmul(q, k);
    if (mask) { 
        dim3 block_size(16, 16);
        dim3 grid_size((seq_len + block_size.x - 1) / block_size.x, (seq_len + block_size.y - 1) / block_size.y, batch_size * heads);
        mask_kernel<<<grid_size, block_size>>>(weight.data_gpu, batch_size, seq_len, heads);
        cudaDeviceSynchronize();
        CHECK_ERROR();
        cudaMemcpy(weight.data_cpu, weight.data_gpu, weight.size * sizeof(float), cudaMemcpyDeviceToHost);
    }
    weight = weight * (1.0 / sqrt(d_head));
    weight = Tensor::softmax(weight);
    Tensor result = Tensor::matmul(weight, v).transpose(1, 2).reshape(input.dims, 3);
    return out_proj.forward(result);
}

void SelfAttention::free_memory() {
    q_proj.free_memory();
    k_proj.free_memory();
    v_proj.free_memory();
    out_proj.free_memory();
}

CrossAttention::CrossAttention() {
    q_proj = Linear();
    k_proj = Linear();
    v_proj = Linear();
    out_proj = Linear();
    heads = 0;
    d_embed = 0;
    d_head = 0;
}

CrossAttention::CrossAttention(int heads, int d_embed, int d_cross, bool proj_bias, bool out_bias) {
    q_proj = Linear(d_embed, d_embed, proj_bias);
    k_proj = Linear(d_cross, d_embed, proj_bias);
    v_proj = Linear(d_cross, d_embed, proj_bias);
    out_proj = Linear(d_embed, d_embed, out_bias);
    this->heads = heads;
    this->d_embed = d_embed;
    this->d_head = d_embed / heads;
    this->d_cross = d_cross;
}

Tensor CrossAttention::forward(Tensor& input, Tensor& context) {
    int batch_size = input.dims[0];
    int seq_len = input.dims[1];
    int interim_dims[] = {batch_size, seq_len, heads, d_head};
    Tensor q = q_proj.forward(input).reshape(interim_dims, 4).transpose(1, 2);
    Tensor k = k_proj.forward(context).reshape(interim_dims, 4).transpose(1, 2).transpose(2, 3);
    Tensor v = v_proj.forward(context).reshape(interim_dims, 4).transpose(1, 2);
    Tensor weight = Tensor::matmul(q, k);
    weight = weight * (1.0 / sqrt(d_head));
    weight = Tensor::softmax(weight);
    Tensor result = Tensor::matmul(weight, v).transpose(1, 2).reshape(input.dims, 3);
    return out_proj.forward(result);
}