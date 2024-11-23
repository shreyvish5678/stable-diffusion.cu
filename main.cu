#include "attention.h"
#include "kernels.h"
#include "tensor.h"
#include "linear.h"

int main() {
    Tensor a = Tensor(new int[3]{4, 64, 768}, 3);
    a.init_rand();
    Attention attn1 = Attention(12, 768, false, false);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    Tensor b = attn1.forward(a, false);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time: %f\n", elapsedTime);
    printf("%f\n", b.at(0, 0, 0));
    a.save("input.bin");
    b.save("output.bin");
    attn1.q_proj.weights.save("q_proj_weights.bin");
    attn1.k_proj.weights.save("k_proj_weights.bin");
    attn1.v_proj.weights.save("v_proj_weights.bin");
    attn1.out_proj.weights.save("out_proj_weights.bin");
    return 0;
}