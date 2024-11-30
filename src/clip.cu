#include "layers/tensor.h"
#include "layers/linear.h"
#include "layers/attention.h"
#include "layers/embedding.h"
#include "layers/layernorm.h"
#include "clip.h"

ClipEmbedding::ClipEmbedding() {
    this->n_vocab = 0;
    this->n_embed = 0;
    this->n_token = 0;
}

ClipEmbedding::ClipEmbedding(int n_vocab, int n_embed, int n_token) {
    this->n_vocab = n_vocab;
    this->n_embed = n_embed;
    this->n_token = n_token;
    token_embed = Embedding(n_vocab, n_embed);
    pos_embed = Tensor(new int[2]{n_token, n_embed}, 2);
    pos_embed.init_rand();
}

Tensor ClipEmbedding::forward(int* tokens, int batch_size) {
    Tensor embeddings = token_embed.forward(tokens, batch_size, n_token) + pos_embed;
    return embeddings;
}

void ClipEmbedding::free_memory() {
    token_embed.free_memory();
    pos_embed.free_memory();
}

ClipLayer::ClipLayer() {
    this->n_head = 0;
    this->n_embed = 0;
}

ClipLayer::ClipLayer(int n_head, int n_embed) {
    this->n_head = n_head;
    this->n_embed = n_embed;
    self_attention = SelfAttention(n_head, n_embed);
    layernorm1 = LayerNorm(n_embed);
    linear1 = Linear(n_embed, n_embed * 4);
    layernorm2 = LayerNorm(n_embed);
    linear2 = Linear(n_embed * 4, n_embed);
}

Tensor ClipLayer::forward(Tensor& input) {
    Tensor norm1 = layernorm1.forward(input);
    Tensor attention = self_attention.forward(norm1, true) + input;
    Tensor norm2 = layernorm2.forward(attention);
    Tensor linear1_out = linear1.forward(norm2);
    Tensor gelu = Tensor::gelu(linear1_out);
    Tensor linear2_out = linear2.forward(gelu);
    Tensor output = linear2_out + attention;
    return output;
};

void ClipLayer::free_memory() {
    self_attention.free_memory();
    layernorm1.free_memory();
    linear1.free_memory();
    layernorm2.free_memory();
    linear2.free_memory();
}

CLIP::CLIP() {
    embedding = ClipEmbedding(49408, 768, 77);
    for (int i = 0; i < 12; i++) {
        layers[i] = ClipLayer(12, 768);
    }
    layernorm = LayerNorm(768);
}

Tensor CLIP::forward(int* tokens, int batch_size) {
    Tensor embeddings = embedding.forward(tokens, batch_size);
    Tensor output = embeddings;
    for (int i = 0; i < 12; i++) {
        output = layers[i].forward(output);
    }
    output = layernorm.forward(output);
    return output;
}

void CLIP::free_memory() {
    for (int i = 0; i < 12; i++) {
        layers[i].free_memory();
    }
    embedding.free_memory();
}