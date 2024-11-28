#include "kernels.h"
#include "tensor.h"
#include "embedding.h"

Embedding::Embedding() {
    d_embed = 0;
    vocab_size = 0;
}

Embedding::Embedding(int vocab_size, int d_embed) {
    this->vocab_size = vocab_size;
    this->d_embed = d_embed;
    weight = Tensor(new int[2]{vocab_size, d_embed}, 2);
    weight.init_rand();
}

Tensor Embedding::forward(const int* tokens, int batch_size, int seq_len) {
    Tensor result = Tensor(new int[3]{batch_size, seq_len, d_embed}, 3);
    int block_size = 256;
    int grid_size = (result.size + block_size - 1) / block_size;
    int* tokens_gpu;
    cudaMalloc(&tokens_gpu, batch_size * seq_len * sizeof(int));
    cudaMemcpy(tokens_gpu, tokens, batch_size * seq_len * sizeof(int), cudaMemcpyHostToDevice);
    embedding_lookup_kernel<<<grid_size, block_size>>>(result.data_gpu, weight.data_gpu, tokens_gpu, batch_size, seq_len, d_embed, vocab_size, result.size);
    cudaDeviceSynchronize();
    CHECK_ERROR();
    cudaMemcpy(result.data_cpu, result.data_gpu, result.size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(tokens_gpu);
    
    return result;
}

void Embedding::free_memory() {
    weight.free_memory();
}