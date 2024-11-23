#include "attention.h"
#include "kernels.h"
#include "tensor.h"
#include "linear.h"

int main() {
    Tensor a = Tensor(new int[4]{4, 64, 64, 3}, 4);
    a.load("output.bin");
    printf("%f\n", a.at(new int[4]{2, 3, 1, 0}));
    a = a.transpose(1, 2);
    printf("%f\n", a.at(new int[4]{2, 3, 1, 0}));
    return 0;
}