#include <iostream>
#include "src/clip.h"

int main() {
    int tokens[2][77];
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 77; j++) {
            tokens[i][j] = rand() % 100;   
        }
    }
    CLIP clip = CLIP();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    Tensor output = clip.forward(&tokens[0][0], 2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time: " << milliseconds << "ms" << std::endl;
    std::cout << "Output: " << output.data_cpu[0] << std::endl;
    return 0;
}