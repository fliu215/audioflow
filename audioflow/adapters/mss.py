import torch
import torch.nn as nn
from torch import Tensor

from audioflow.encoders.text.clap_text import CLAPTextEncoder
from audioflow.encoders.text.flan_t5 import FlanT5
from audioflow.utils.torch import check_masks_type, ones
from audioflow.utils.xml import batch_parse_xml


class MSSAdapter(nn.Module): 
    def __init__(self, in_dim: int, dim: int, **kwargs):
        super().__init__()

        # T5
        self.t5 = FlanT5()
        self.t5_proj = nn.Linear(self.t5.dim, dim)

        # CLAP
        self.c_proj = nn.Linear(in_dim, dim)

    def forward(self, data: dict) -> Tensor:

        # Prompt
        prompt, prompt_mask = self.t5(data["prompt"])  # (b, l_text, d)
        prompt = self.t5_proj(prompt)  # (b, l_text, d)
        B = prompt.shape[0]

        in_feat = self.c_proj(data["input_feature"])

        # Sequence
        seq = torch.cat([in_feat, prompt], dim=1)

        # Self attention mask
        tgt_mask = data["target_mask"]  # (b, l_q)
        self_mask = tgt_mask[:, :, None] * tgt_mask[:, None, :]  # (b, l_q, l_q)

        # Cross attention mask
        cm1 = ones(
            n=data["target_latent"].shape[1],
            m=data["input_feature"].shape[1],
            device=in_feat.device
        )[None, :, :].expand(B, -1, -1)  # (b, l_q, l_v)
        
        cm2 = tgt_mask[:, :, None] * prompt_mask[:, None, :]  # (b, l_q, l_v)
        cross_mask = torch.cat([cm1, cm2], dim=-1)

        # Check
        assert check_masks_type([self_mask, cross_mask], torch.bool)

        controls = {
            "c": 0.,  # (b, 1, d)
            "seq": seq,  # (b, l, d)
            "self_attn_mask": self_mask.unsqueeze(1),  # (b, 1, l_q, l_q)
            "cross_attn_mask": cross_mask.unsqueeze(1)  # (b, 1, l_q, l_v)
        }

        return controls 