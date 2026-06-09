import torch.nn as nn
from torch import Tensor


class AudioFlow(nn.Module):
    def __init__(
        self, 
        in_: nn.Module, 
        base: nn.Module, 
        out: nn.Module, 
        adapter: nn.Module
    ) -> None:
        super().__init__()
        self.in_ = in_
        self.out = out
        self.base = base
        self.adapter = adapter

    def forward(self, t: Tensor, x: Tensor, data: dict) -> Tensor:
        controls = self.adapter(data)
        x = self.in_(x)
        x = self.base(t, x, controls)
        x = self.out(x)
        return x