import torch
import torch.nn as nn
from torch import Tensor

from audio_flow.models.attention2 import SelfAttention, RMSNorm
from audio_flow.models.rope import RoPE


class Aligner2(nn.Module):
    r"""Convert a variable-length representation into a fixed-length latent.
    """
    def __init__(self, in_dim: int, dim: int, max_length: int):
        super().__init__()

        self.fc = nn.Linear(in_dim, dim)
        self.register = nn.Parameter(torch.randn(max_length, dim))

        head_dim = 32
        num_heads = dim // head_dim
        self.blocks = nn.ModuleList(AttentionBlock(dim, num_heads) for _ in range(6))
        self.rope = RoPE(head_dim, max_len=8192)

    def forward(self, c: Tensor, mask: Tensor) -> Tensor:
        r"""Convert a variable-length representation into a fixed-length latent.

        b: batch_size
        l_in: input_seq_len
        l_out: output_seq_len
        d: dim

        Args:
            c: (b, l_in, d)
            length: int

        Returns:
            out: (b, l_out, d)
        """

        B, L1 = c.shape[0 : 2]
        L2 = mask.shape[-1] - L1
        c = self.fc(c)  # (b, l1, d)
        x = self.register[None, 0 : L2, :].repeat(B, 1, 1)  # (b, l2, d)
        x = torch.cat((c, x), dim=1)  # (b, l1+l2, d)

        for block in self.blocks:
            x = block(x, self.rope, mask)  # (b, l1+l2, d)

        return x[:, L1:, :]  # (b, l2, d)



class AttentionBlock(nn.Module):
    r"""Self attention block."""
    def __init__(self, dim, num_heads) -> None:
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        self.self_attn = SelfAttention(dim, num_heads)        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), 
            nn.GELU(approximate='tanh'),
            nn.Linear(dim * 4, dim)
        )

    def forward(
        self,
        x: Tensor,
        rope: RoPE,
        mask: Tensor
    ) -> torch.Tensor:
        r"""Self attention block.

        Args:
            x: (b, l, d)
            rope: (t, head_dim/2, 2)
            mask: None | (1, 1, l, l)
            emb: (b, l, d)

        Outputs:
            out: (b, l, d)
        """

        x = x + self.self_attn(self.norm2(x), rope, mask)
        x = x + self.ffn(x)
        return x