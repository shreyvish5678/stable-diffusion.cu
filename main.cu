#include "attention.h"
#include "kernels.h"
#include "tensor.h"
#include "linear.h"
#include "embedding.h"

int main() {
    int batch_size = 2;
    int seq_len = 3;
    int d_embed = 768;
    int vocab_size = 1024;
    Embedding embedding = Embedding(vocab_size, d_embed);
    int input_tokens[batch_size][seq_len] = {
        {1, 2, 3},
        {4, 5, 6}
    };
    Tensor result = embedding.forward((int*)input_tokens, batch_size, seq_len);
    printf("%f\n", result.data_cpu[0]);
}