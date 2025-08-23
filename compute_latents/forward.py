import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def forward_vae(vae: nn.Module, audio: np.array, clip_samples: int, sr: float) -> np.array:

    device = next(vae.parameters()).device
    latents = []
    i = 0

    while i < audio.shape[-1]:

        if audio.shape[-1] - i < int(1. * sr):
            continue
        
        x = Tensor(audio[None, :, i : i + clip_samples]).to(device)

        with torch.no_grad():
            vae.eval()
            latent = vae.encode_audio(x)[0].data.cpu().numpy()  # (d, t)

        latents.append(latent)
        i += clip_samples

    latents = np.concatenate(latents, axis=-1)

    return latents