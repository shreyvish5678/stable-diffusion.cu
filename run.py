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
        print(q[0, 0, 0])
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(interim_shape).transpose(1, 2)
        print(q[0, 0, 0, 0])
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
    
self_attn = SelfAttention(12, 768, in_proj_bias=False, out_proj_bias=False)
input = np.fromfile('input.bin', dtype=np.float32).reshape(4, 64, 768)
self_attn.q_proj.weight.data = torch.from_numpy(np.fromfile('q_proj_weights.bin', dtype=np.float32).reshape(768, 768))
self_attn.k_proj.weight.data = torch.from_numpy(np.fromfile('k_proj_weights.bin', dtype=np.float32).reshape(768, 768))
self_attn.v_proj.weight.data = torch.from_numpy(np.fromfile('v_proj_weights.bin', dtype=np.float32).reshape(768, 768))
self_attn.out_proj.weight.data = torch.from_numpy(np.fromfile('out_proj_weights.bin', dtype=np.float32).reshape(768, 768))
output = self_attn(torch.tensor(input, dtype=torch.float32), causal_mask=False)
print(output[0, 0, 0])