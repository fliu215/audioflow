import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from audio_flow.models.attention import SelfAttention, RMSNorm
from audio_flow.models.rope import RoPE


class Aligner_ljspeech_04b(nn.Module):
    r"""Convert a variable-length representation into a fixed-length latent.
    """
    def __init__(self, in_dim: int, dim: int, max_length: int):
        super().__init__()

        self.fc = nn.Linear(in_dim, dim)

        head_dim = 32
        num_heads = dim // head_dim
        self.blocks = nn.ModuleList(AlignerBlock(dim, num_heads) for _ in range(0))
        self.rope = RoPE(head_dim, max_len=8192)

    def forward(self, x: Tensor, length: int) -> Tensor:
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

        x = F.pad(x, (0, 0, 0, length - x.shape[1]))
        x = self.fc(x)

        for block in self.blocks:
            x = block(x, self.rope)

        return x



class AlignerBlock(nn.Module):
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
        
        x = x + self.self_attn(x, rope)
        x = x + self.ffn(x)
        return x