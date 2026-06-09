import torch.nn as nn


def load_vae(name: str) -> nn.Module:
    
    if name == "levo_vae":
        from .levo import LevoVAE
        return LevoVAE()

    elif name == "architts_vae":
        from .architts_vae import ArchiTTSVAE
        return ArchiTTSVAE()

    elif name == "mmaudio_vae":
        from .mmaudio import MMAudioVAE
        return MMAudioVAE()

    else:
        raise ValueError(name)
