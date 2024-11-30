#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "tensor.h"

class Embedding {
public:
    Tensor weight;
    int vocab_size;
    int d_embed;

    Embedding();
    Embedding(int vocab_size, int d_embed);

    Tensor forward(int* tokens, int batch_size, int seq_len);
    void free_memory();
};

#endif 