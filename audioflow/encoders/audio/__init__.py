import torch.nn as nn


def load_encoder(name: str) -> nn.Module:
    
    if name == "levo_vae":
        from audioflow.vaes.audio.levo import LevoVAE
        return LevoVAE()

    elif name == "architts_vae":
        from audioflow.vaes.audio.architts import ArchiTTSVAE
        return ArchiTTSVAE()

    elif name == "mmaudio_vae":
        from audioflow.vaes.audio.mmaudio import MMAudioVAE
        return MMAudioVAE(sr=16000)

    elif name == "mmaudio44k_vae":
        from audioflow.vaes.audio.mmaudio import MMAudioVAE
        return MMAudioVAE(sr=44100)

    else:
        raise ValueError(name)
