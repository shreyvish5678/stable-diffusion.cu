Work in progress for stable diffusion in CUDA

Plan:
- Implement basic forward pass (Done)
- Implement forward pass for 3D arrays, needed for Attention (Done)
- Implement Convolution (Done)
- Implement array manipulation:
    - Reshaping arrays
    - Adding dimensions
    - Splitting arrays by certain dimensions
    - Transposing, etc.
- Implement Attention
    - Self-attention
    - Cross-attention
- Implement Group normalization
- Implement VAE blocks
- Implement LayerNorm
- Implement Embedding layers
- Implement UNET Blocks
- Implement VAE, Clip and UNET
- Implement DDPM Scheduler
- Implement Pipeline

Thanks to:
https://github.com/hkproj/pytorch-stable-diffusion
https://github.com/yanconglin/Conv2d_Pytorch_from_scratch