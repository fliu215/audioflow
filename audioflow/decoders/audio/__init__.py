import torch.nn as nn


def load_decoder(name: str) -> nn.Module:
    
    if name == "levo_vae":
        from audioflow.vaes.audio.levo import LevoVAE
        return LevoVAE()

    elif name == "architts_vae":
        from audioflow.vaes.audio.architts_vae import ArchiTTSVAE
        return ArchiTTSVAE()

    elif name == "mmaudio_vae":
        from audioflow.vaes.audio.mmaudio import MMAudioVAE
        return MMAudioVAE()

    else:
        raise ValueError(name)
