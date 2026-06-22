import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from audioflow.encoders.text.char import CharEncoder
from audioflow.utils.torch import check_masks_type
from audioflow.utils.xml import batch_parse_xml


class TTSAdapter(nn.Module): 
    def __init__(self, dim: int, **kwargs):
        super().__init__()

        # Character encoder
        self.char_encoder = CharEncoder()
        self.char_embedder = nn.Embedding(self.char_encoder.vocab_size, dim)
        self.char_conv = ConvNeXt(dim)

    def forward(self, data: dict) -> Tensor:

        # Prompt
        text = batch_parse_xml(data["prompt"])
        text, text_mask = self.char_encoder(text)  # (b, l_text, d), (b, l_text)
        text = self.char_embedder(text)  # (b, l_text, d)
        text = self.char_conv(text)

        # Sequence
        seq = text

        # Self attention mask
        tgt_mask = data["target_mask"]  # (b, l_q)
        self_mask = tgt_mask[:, :, None] * tgt_mask[:, None, :]  # (b, l_q, l_q)

        # Cross attention mask
        cross_mask = tgt_mask[:, :, None] * text_mask[:, None, :]  # (b, l_q, l_v)

        # Check
        assert check_masks_type([self_mask, cross_mask], torch.bool)

        controls = {
            "c": 0.,
            "seq": seq,  # (b, l, d)
            "self_attn_mask": self_mask.unsqueeze(1),  # (b, 1, l_q, l_q)
            "cross_attn_mask": cross_mask.unsqueeze(1)  # (b, 1, l_q, l_v)
        }
        
        return controls 



class ConvNeXt(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.blocks = nn.ModuleList([ConvNeXtBlock(dim) for _ in range(3)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        r"""

        Args:
            x: (b, l, d)

        Returns:
            out: (b, l, d)
        """

        residual = x
        x = rearrange(x, 'b l d -> b d l')
        x = self.dwconv(x)
        x = rearrange(x, 'b d l -> b l d')
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x + residual