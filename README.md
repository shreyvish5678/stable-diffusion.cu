# What is this?

I made this project to become more familiar with CUDA, and have already and am currently learnt a lot about CUDA. My end goal is to first create a stable diffusion implementation in global memory, and then optimize my implementation to make it actually usable. You can see the checklist below for my progress, and maybe help out a little here and there. 

# Run

Clone this repository and inside the repo run
`!nvcc -o main $(find . -name "*.cu") && ./main && rm main`
It should compile run and clean everything with one simple command!

# Checklist

- [x] **Create a small Tensor library to handle computations**
  - [x] Initialization, Allocation, Freeing Memory
  - [x] Indexing, Basic Operations
  - [x] Reshaping, Transposing
  - [x] Matrix Multiplication, Softmax
  - [x] Saving, loading from Binary Files

- [ ] **Create kernels to use in the Tensor library**
  - [x] Initialization Kernels
  - [ ] Addition Kernels
    - [ ] Generalized [Same Size]
    - [x] Different Sizes (Do as needed)
  - [x] Multiplication Kernels
  - [x] Matmul Kernels
    - [x] 2Dx2D
    - [x] 3Dx2D
    - [x] 4Dx4D
  - [x] Transpose, Softmax kernels
  - [x] Mask kernel for attention
  - [x] Embedding Lookup kernel
  - [ ] Conv2d Kernel (4D input, 4D weight, kernel size, stride, padding), Upsampling
  - [ ] Mean, Variance Kernels
  - [ ] LayerNorm, Group Norm
  - [ ] Activations: Silu, Gelu

- [x] **Attention Class**
  - [x] Self Attention
  - [x] Cross Attention

- [x] Linear Class
- [x] Embedding Class
- [ ] CLIP Class
- [ ] Decoder, Encoder Classes
- [ ] Tokenizer
- [ ] Diffusion Model Class
  - [ ] Time embeddings

- [ ] **Sampling**
  - [ ] Backward Diffusion (Since only sampling, no forward needed)
    - [ ] Euler
    - [ ] Ancestral
    - [ ] LMS

- [ ] **Pipeline**
  - [ ] Model Loading from .bin
  - [ ] Final product finished