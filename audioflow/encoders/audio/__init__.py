import torch.nn as nn


def load_encoder(name: str) -> nn.Module:
    
    if name == "levo_vae":
        from audioflow.encoders.audio.levo_vae import LevoVAE
        return LevoVAE()

    elif name == "architts_vae":
        from audioflow.encoders.audio.architts_vae import ArchiTTSVAE
        return ArchiTTSVAE()

    elif name == "mmaudio_vae":
        from audioflow.encoders.audio.mmaudio_vae import MMAudioVAE
        return MMAudioVAE()

    else:
        raise ValueError(name)
