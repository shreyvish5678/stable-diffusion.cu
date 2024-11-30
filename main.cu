#include "attention.h"
#include "kernels.h"
#include "tensor.h"
#include "linear.h"
#include "embedding.h"
#include "layernorm.h"

int main() {
    int tokens[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Embedding embedding = Embedding(10, 3);
    Tensor output = embedding.forward(&tokens[0][0], 2, 3);
    printf("%f\n", embedding.weight.data_cpu[0]);
    printf("%f\n", output.data_cpu[0]);
    embedding.weight.save("weight.bin");
}