import torch.nn as nn
from torch import Tensor


class OnehotEncoder(nn.Module):
    def __init__(self, num_classes: int, dim: int):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Embedding(num_classes, dim),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )

    def forward(self, cond_dict: dict) -> Tensor:
        r"""Compute onehot embedding."""

        emb_dict = {
            "c": self.mlp(cond_dict["id"]),
        }

        return emb_dict