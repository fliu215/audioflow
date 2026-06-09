from __future__ import annotations

import torch.nn as nn
from einops import rearrange
from torch import Tensor
import torch

from audioflow.layers.attention import Block
from audioflow.layers.embedders import TimestepEmbedder
from audioflow.layers.rope import RoPE


class Transformer(nn.Module):
    def __init__(
        self,
        dim=384,
        mlp_ratio=4.0,
        num_layers=12,
        num_heads=12,
        rope_len=8192,
        **kwargs
    ):
        super().__init__()

        # Time embedder
        self.t_embedder = TimestepEmbedder(dim=dim, freq_size=256, scale=1000.)

        self.blocks = nn.ModuleList(Block(dim, num_heads) for _ in range(num_layers))

        head_dim = dim // num_heads
        self.rope = RoPE(head_dim, max_len=rope_len)

    def forward(
        self, 
        t: Tensor, 
        x: Tensor, 
        controls: dict,
        **kwargs,
    ) -> Tensor:
        r"""DiT.

        b: batch_size
        d: dim
        t: time_step

        Args:
            t: (b,), random time steps between 0. and 1.
            x: (b, t, d)

        Outputs:
            out: (b, t, d)
        """
        c = controls["c"]  # (b, 1, d) | (b, t, d)
        seq = controls["seq"]  # (b, l, d)
        self_attn_mask = controls["self_attn_mask"]  # (b, t, d)
        cross_attn_mask = controls["cross_attn_mask"]  # (b, l, d)

        # Time embedding
        if t.dim() == 0:
            t = t.repeat(x.shape[0])  # (b,)

        c = c + self.t_embedder(t)[:, None, :]  # (b, 1, d) | (b, t, d)

        # Transformer
        for block in self.blocks:
            x = block(x, c, seq, self.rope, self_attn_mask, cross_attn_mask)
        
        return x