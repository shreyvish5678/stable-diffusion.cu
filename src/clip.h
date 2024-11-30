#ifndef CLIP_H
#define CLIP_H

#include "layers/tensor.h"
#include "layers/linear.h"
#include "layers/attention.h"
#include "layers/embedding.h"
#include "layers/layernorm.h"

class ClipEmbedding {
public:
    int n_vocab;
    int n_embed;
    int n_token;
    Embedding token_embed;
    Tensor pos_embed;
    ClipEmbedding();
    ClipEmbedding(int n_vocab, int n_embed, int n_token);
    Tensor forward(int* tokens, int batch_size);
    void free_memory();
};

class ClipLayer {
public:
    int n_head;
    int n_embed;
    LayerNorm layernorm1;
    LayerNorm layernorm2;
    SelfAttention self_attention;
    Linear linear1;
    Linear linear2;
    ClipLayer();
    ClipLayer(int n_head, int n_embed);
    Tensor forward(Tensor& input);
    void free_memory();
};

class CLIP {
public:
    ClipEmbedding embedding;
    ClipLayer layers[12];
    LayerNorm layernorm;
    CLIP();
    Tensor forward(int* tokens, int batch_size);
    void free_memory();
};

#endif