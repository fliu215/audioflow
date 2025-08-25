import torch
import torch.nn as nn
import torchaudio
from audidata.datasets import GTZAN
from torch import Tensor
from einops import rearrange

from audio_flow.vae.levo import LevoVAE


class MSSVAE(nn.Module):
    def __init__(self, target_stem: str):
        super().__init__()

        self.target_stem = target_stem
        self.vae = LevoVAE()
        self.sr = self.vae.sr

    def audio_to_latent(self, data: dict) -> tuple[Tensor, dict]:
        r"""Transform data into latent representations and conditions.

        b: batch_size
        c: channels_num
        l: audio_samples
        t: frames_num
        f: mel bins
        """
        
        device = next(self.parameters()).device

        # Mel spectrogram target
        mixture_latent = data["mixture_latent"].to(device)  # (b, d, t)
        target_latent = data["target_latent"].to(device)  # (b, d, t)

        # Condition
        cond_dict = {            
            "ct": mixture_latent
        }

        return target_latent, cond_dict

    def latent_to_audio(self, x: Tensor) -> Tensor:
        r"""Ues vocoder to convert mel spectrogram to audio.

        Args:
            x: (b, c, t, f)

        Outputs:
            y: (b, c, l)
        """
        x = self.vae.decode(x)
        return x

    def __call__(self, data: dict) -> tuple[Tensor, dict]:
        return self.audio_to_latent(data)