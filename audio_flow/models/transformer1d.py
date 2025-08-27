from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from audio_flow.models.attention import Block
from audio_flow.models.embedders import LabelEmbedder, MlpEmbedder, TimestepEmbedder
from audio_flow.models.rope import RoPE
from audio_flow.models.pad import pad1d


class Transformer1D(nn.Module):
    def __init__(
        self,
        in_dim=16,
        patch_size=1,
        dim=384,
        mlp_ratio=4.0,
        num_layers=12,
        num_heads=12,
        rope_len=8192,
        **kwargs
    ):
        
        super().__init__()

        self.patch_size = patch_size

        self.patch_x = nn.Conv1d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.unpatch_x = nn.ConvTranspose1d(dim, in_dim, kernel_size=patch_size, stride=patch_size)

        # Time embedder
        self.t_embedder = TimestepEmbedder(dim=dim, freq_size=256, scale=100.)

        self.blocks = nn.ModuleList(Block(dim, num_heads) for _ in range(num_layers))

        head_dim = dim // num_heads
        self.rope = RoPE(head_dim, max_len=rope_len)

    def forward(
        self, 
        t: Tensor, 
        x: Tensor, 
        emb_dict: dict
    ) -> Tensor:
        """Model

        Args:
            t: (b,), random time steps between 0. and 1.
            x: (b, d, t)
            cond_dict: dict

        Outputs:
            output: (b, d, t)
        """

        assert all(key in ["c", "ct", "cx"] for key in emb_dict.keys()), "Invalid key in emb_dict!"

        c = emb_dict.get("c", None)
        ct = emb_dict.get("ct", None)
        cx = emb_dict.get("cx", None)

        B, D, T = x.shape
        x = pad1d(x, self.patch_size)  # x: (b, d, t)
        x = self.patch_x(x)  # shape: (b, d, t, f)

        e = torch.zeros_like(x)

        # 2.1 Time embedder. Repeat B times for inference
        if t.dim() == 0:
            t = t.repeat(B)

        e += self.t_embedder(t)[:, :, None]
        
        if c is not None: 
            e += c[:, :, None]
        
        if ct is not None: 
            e += ct[:, :, :]

        if cx is not None:
            cx = rearrange(cx, 'b d t -> b t d')

        x = rearrange(x, 'b d t -> b t d')
        e = rearrange(e, 'b d t -> b t d')

        for block in self.blocks:
            x = block(x, e, cx, self.rope)

        x = rearrange(x, 'b t d -> b d t')

        x = self.unpatch_x(x)
        x = x[:, :, 0 : T]

        return x