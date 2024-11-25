import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        output = weight @ v
        output = output.transpose(1, 2)
        output = output.reshape(input_shape)
        output = self.out_proj(output)
        return output

attention = SelfAttention(12, 768, True, True)
input = torch.from_numpy(np.fromfile('input.bin', dtype=np.float32).reshape(16, 77, 768))
print(input[8, 8, 8].item())
attention.q_proj.weight.data = torch.from_numpy(np.fromfile('q_proj_weights.bin', dtype=np.float32).reshape(768, 768)).T
print(attention.q_proj.weight.data[8, 8].item())
attention.q_proj.bias.data = torch.from_numpy(np.fromfile('q_proj_bias.bin', dtype=np.float32))
print(attention.q_proj.bias.data[8].item())
attention.k_proj.weight.data = torch.from_numpy(np.fromfile('k_proj_weights.bin', dtype=np.float32).reshape(768, 768)).T
print(attention.k_proj.weight.data[8, 8].item())
attention.k_proj.bias.data = torch.from_numpy(np.fromfile('k_proj_bias.bin', dtype=np.float32))
print(attention.k_proj.bias.data[8].item())
attention.v_proj.weight.data = torch.from_numpy(np.fromfile('v_proj_weights.bin', dtype=np.float32).reshape(768, 768)).T
print(attention.v_proj.weight.data[8, 8].item())
attention.v_proj.bias.data = torch.from_numpy(np.fromfile('v_proj_bias.bin', dtype=np.float32))
print(attention.v_proj.bias.data[8].item())
attention.out_proj.weight.data = torch.from_numpy(np.fromfile('out_proj_weights.bin', dtype=np.float32).reshape(768, 768)).T
print(attention.out_proj.weight.data[8, 8].item())
attention.out_proj.bias.data = torch.from_numpy(np.fromfile('out_proj_bias.bin', dtype=np.float32))
print(attention.out_proj.bias.data[8].item())
output = attention(input)
output = attention(input, True)
print(output[8, 8, 8].item())