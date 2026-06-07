import torch.nn as nn
from einops import rearrange
from torch import Tensor
import torch

from audioflow.encoders.text.flan_t5 import FlanT5
from audioflow.encoders.text.clap_text import CLAPTextEncoder

from audioflow.utils.torch import check_masks_type


class TTMAdapter(nn.Module): 
    def __init__(self, dim: int, **kwargs):
        super().__init__()

        # T5
        self.t5 = FlanT5()
        self.t5_proj = nn.Linear(self.t5.dim, dim)

        # CLAP
        self.clap_text = CLAPTextEncoder()
        self.clap_text_proj = nn.Linear(self.clap_text.dim, dim)

    def forward(self, data: dict) -> Tensor:

        prompt, prompt_mask = self.t5(data["prompt"])  # (b, l, d)
        prompt = self.t5_proj(prompt)  # (b, l, d)
        
        # Clap
        clap = self.clap_text(data["prompt"])  # (b, d)
        clap = self.clap_text_proj(clap)[:, None, :]  # (b, 1, d)
        
        # Build mask
        tgt_mask = data["target_mask"]  # (b, l_q)
        self_mask = tgt_mask[:, :, None] * tgt_mask[:, None, :]  # (b, l_q, l_q)
        cross_mask = tgt_mask[:, :, None] * prompt_mask[:, None, :]  # (b, l_q, l_v)
        assert check_masks_type([self_mask, cross_mask], torch.bool)

        controls = {
            "c": clap,  # (b, 1, d)
            "seq": prompt,  # (b, l, d)
            "self_attn_mask": self_mask.unsqueeze(1),  # (b, 1, l_q, l_q)
            "cross_attn_mask": cross_mask.unsqueeze(1)  # (b, 1, l_q, l_v)
        }

        return controls 
