#include "kernels.h"
#include "tensor.h"
#include "linear.h"
#include "attention.h"

Attention::Attention(int heads, int d_embed, bool proj_bias, bool out_bias) {
    q_proj = Linear(d_embed, d_embed, proj_bias);
    k_proj = Linear(d_embed, d_embed, proj_bias);
    v_proj = Linear(d_embed, d_embed, proj_bias);
    out_proj = Linear(d_embed, d_embed, out_bias);
    this->heads = heads;
    this->d_embed = d_embed;
    this->d_head = d_embed / heads;
}

Tensor Attention::forward(const Tensor& input, bool mask) {
    int batch_size = input.dims[0];
    int seq_len = input.dims[1];
    int interim_dims[] = {batch_size, seq_len, heads, d_embed};
    Tensor q = q_proj.forward(input).reshape(interim_dims, 4).transpose(1, 2);
    Tensor k = k_proj.forward(input).reshape(interim_dims, 4).transpose(1, 2);
    Tensor v = v_proj.forward(input).reshape(interim_dims, 4).transpose(1, 2);
    Tensor weight = Tensor::matmul(q, k.transpose(2, 3));
    if (mask) { 
        dim3 block_size(16, 16);
        dim3 grid_size(batch_size, seq_len, (heads + block_size.y - 1) / block_size.y);
        mask_kernel<<<grid_size, block_size>>>(weight.data_gpu, batch_size, seq_len, heads, d_head, 1);
    }
    weight = weight * (1.0 / sqrt(d_head));
    weight = weight.softmax();
    Tensor result = Tensor::matmul(weight, v).transpose(1, 2).reshape(input.dims, 3);
    return out_proj.forward(result);
}

int main() {
    Tensor input = Tensor(new int[3]{32, 128, 512}, 3);
    input.init_rand();
    return 0;
}