import torch
import torch.nn as nn
import torchaudio
from audidata.datasets import GTZAN
from torch import Tensor
from einops import rearrange

from audio_flow.vae.levo import LevoVAE


class Label2MusicVAE(nn.Module):
    def __init__(self):
        super().__init__()

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
        latent = data["latent"].to(device)  # (b, d, t)
        # latent = rearrange(latent, 'b d t -> b t d')

        ids = data["target"].to(device)  # (b,)
        captions = data["label"]  # (b,)

        # Condition
        cond_dict = {
            "id": ids,
            "caption": captions
        }

        return latent, cond_dict

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