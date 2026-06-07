import torch.nn as nn


def get_in(configs: dict) -> nn.Module:
    return nn.Linear(configs["in_dim"], configs["dim"])


def get_out(configs: dict) -> nn.Module:
    return nn.Linear(configs["dim"], configs["out_dim"])


def get_base(configs: dict) -> nn.Module:
    r"""Initialize base."""
    name = configs["name"]

    if name == "Transformer":
        from .transformer import Transformer
        return Transformer(**configs)

    else:
        raise ValueError(name)