import torch.nn as nn


def get_adapter(configs: dict) -> nn.Module:
    r"""Initialize adapter."""
    name = configs["name"]

    if name == "TTMAdapter":
        from .ttm import TTMAdapter
        return TTMAdapter(**configs)

    elif name == "TTAAdapter":
        from .tta import TTAAdapter
        return TTAAdapter(**configs)

    elif name == "V2AAdapter":
        from .v2a import V2AAdapter
        return V2AAdapter(**configs)

    else:
        raise ValueError(name)