from __future__ import annotations

import torch.nn as nn
from einops import rearrange
from torch import Tensor

from audio_flow.models.attention2 import Block2
from audio_flow.models.embedders import TimestepEmbedder
from audio_flow.models.rope import RoPE


class Transformer2(nn.Module):
    def __init__(
        self,
        in_dim=16,
        dim=384,
        mlp_ratio=4.0,
        num_layers=12,
        num_heads=12,
        rope_len=8192,
        **kwargs
    ):
        
        super().__init__()

        self.fc_in = nn.Linear(in_dim, dim)
        self.fc_out = nn.Linear(dim, in_dim)
        
        # Time embedder
        self.t_embedder = TimestepEmbedder(dim=dim, freq_size=256, scale=100.)

        self.blocks = nn.ModuleList(Block2(dim, num_heads) for _ in range(num_layers))

        head_dim = dim // num_heads
        self.rope = RoPE(head_dim, max_len=rope_len)

    def forward(
        self, 
        t: Tensor, 
        x: Tensor, 
        c: Tensor,
        mask: Tensor,
    ) -> Tensor:
        r"""DiT.

        b: batch_size
        d: dim
        t: time_step

        Args:
            t: (b,), random time steps between 0. and 1.
            x: (b, d, t)

        Outputs:
            out: (b, d, t)
        """

        # Time embedding
        if t.dim() == 0:
            t = t.repeat(x.shape[0])  # (b,)
        c = c + self.t_embedder(t)[:, None, :]  # (b, 1, d)
        
        # Transformer
        x = self.fc_in(rearrange(x, 'b d t -> b t d'))
        for block in self.blocks:
            x = block(x, c, self.rope, mask)
        x = rearrange(self.fc_out(x), 'b t d -> b d t')

        return x