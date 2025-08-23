from __future__ import annotations

import os
import sys
from collections import OrderedDict
from contextlib import contextmanager

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import Tensor


def parse_yaml(config_yaml: str) -> dict:
    r"""Parse yaml file."""
    
    with open(config_yaml, "r") as fr:
        return yaml.load(fr, Loader=yaml.FullLoader)


class LinearWarmUp:
    r"""Linear learning rate warm up scheduler."""

    def __init__(self, warm_up_steps: int) -> None:
        self.warm_up_steps = warm_up_steps

    def __call__(self, step: int) -> float:
        if step <= self.warm_up_steps:
            return step / self.warm_up_steps
        else:
            return 1.


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float = 0.999) -> None:
    """Update EMA model weights and buffers from model."""

    # Parameters
    for e, m in zip(ema.parameters(), model.parameters()):
        e.mul_(decay).add_(m.data.float(), alpha=1 - decay)

    # Buffers (BN running stats, etc)
    for e, m in zip(ema.buffers(), model.buffers()):
        if m.dtype in [torch.bool, torch.long]:
            continue
        e.mul_(decay).add_(m.data.float(), alpha=1 - decay)


def requires_grad(model: nn.Module, flag=True) -> None:
    for p in model.parameters():
        p.requires_grad = flag


class CombinedModel(nn.Module):
    def __init__(self, base: nn.Module, adaptor: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.adaptor = adaptor


def load_levo_vae() -> nn.Module:

    import json
    from huggingface_hub import hf_hub_download
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.autoencoders import AudioAutoencoder

    config_path = hf_hub_download(
        repo_id="tencent/SongGeneration", 
        filename="ckpt/vae/stable_audio_1920_vae.json"
    )

    model_path = hf_hub_download(
        repo_id="tencent/SongGeneration", 
        filename="ckpt/vae/autoencoder_music_1320k.ckpt"
    )
    
    with open(config_path, "r") as f:
        model_config = json.load(f)

    model = create_model_from_config(model_config)
    state_dict = torch.load(model_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict)

    vae_config = {
        "fps": 25,
        "sample_rate": model_config["sample_rate"]
    }

    return model, vae_config


def logmel(audio: np.ndarray, sr: float) -> np.ndarray:

    if audio.ndim == 2:
        audio = np.mean(audio, axis=0)

    return np.log10(librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_fft=2048, 
        hop_length=round(sr * 0.01), 
        n_mels=128
    )).T  # (t, f)


'''
def fix_length(x: Tensor, size: int) -> Tensor:

    if x.shape[-1] >= size:
        return x[:, :, 0 : size]
    else:
        pad_t = size - x.shape[-1]
        return F.pad(input=x, pad=(0, pad_t))


@contextmanager
def suppress_print():
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout


class Logmel:
    def __init__(self, sr: float):
        self.sr = sr
        self.n_fft = 2048
        self.hop_length = round(sr * 0.01)
        self.n_mels = 128

    def __call__(self, audio: np.array) -> np.array:
        
        logmel = np.log10(librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sr, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels
        )).T

        return logmel
'''
