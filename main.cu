#include "attention.h"
#include "kernels.h"
#include "tensor.h"
#include "linear.h"

int main() {
    Tensor input = Tensor(new int[3]{16, 77, 768}, 3);
    input.init_rand();
    printf("%f\n", input.at(8, 8, 8));
    SelfAttention attention = SelfAttention(12, 768, true, true);
    Tensor output = attention.forward(input, true);
    printf("%f\n", attention.q_proj.weights.at(8, 8));
    printf("%f\n", attention.q_proj.bias.at(8));
    printf("%f\n", attention.k_proj.weights.at(8, 8));
    printf("%f\n", attention.k_proj.bias.at(8));
    printf("%f\n", attention.v_proj.weights.at(8, 8));
    printf("%f\n", attention.v_proj.bias.at(8));
    printf("%f\n", attention.out_proj.weights.at(8, 8));
    printf("%f\n", attention.out_proj.bias.at(8));
    printf("%f\n", output.at(8, 8, 8));
    input.save("input.bin");
    attention.q_proj.weights.save("q_proj_weights.bin");
    attention.k_proj.weights.save("k_proj_weights.bin");
    attention.v_proj.weights.save("v_proj_weights.bin");
    attention.out_proj.weights.save("out_proj_weights.bin");
    attention.q_proj.bias.save("q_proj_bias.bin");
    attention.k_proj.bias.save("k_proj_bias.bin");
    attention.v_proj.bias.save("v_proj_bias.bin");
    attention.out_proj.bias.save("out_proj_bias.bin");
    output.save("output.bin");
}