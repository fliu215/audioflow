import torch
import torch.nn as nn
import torchaudio
from audidata.datasets import GTZAN
from torch import Tensor
from einops import rearrange

from audio_flow.vae.levo import LevoVAE
from audio_flow.encoders.dac import DAC
from audio_flow.utils import align_temporal_features


class Dac2StereoVAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.dac = DAC()
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

        
        dac_code = data["dac_code"].to(device)
        vae_latent = data["vae_latent"].to(device)

        dac_latent = self.dac.code_to_latent(dac_code)  # (b, d, t)

        dac_latent = align_temporal_features(
            input=dac_latent, 
            target=vae_latent, 
            input_fps=data["dac_fps"][0].item(), 
            target_fps=data["vae_fps"][0].item()
        )

        # Condition
        cond_dict = {            
            "ct": dac_latent
        }

        return vae_latent, cond_dict

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