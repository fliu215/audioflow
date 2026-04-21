from __future__ import annotations

import json
from pathlib import Path

import h5py
import librosa
import numpy as np
import re
import torch
import torch.nn as nn
import yaml


def parse_yaml(config_yaml: str) -> dict:
    r"""Parse yaml file."""
    
    with open(config_yaml, "r") as fr:
        return yaml.load(fr, Loader=yaml.FullLoader)


def load_jsonl(path) -> list[dict]:
    with open(path, "r") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    return lines


def write_jsonl(metas: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for meta in metas:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")


def get_single_value(lst: list):
    unique = list(set(lst))
    assert len(unique) == 1
    return unique[0]


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
    def __init__(self, base: nn.Module, adapter: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.adapter = adapter


def extract_latents_in_chunks(
    model: nn.Module, 
    audio: np.array, 
    chunk_samples: int,
    min_tail_samples: int = 10000
) -> np.array:
    r"""Convert audio into latents.

    c: audio_channels
    l: audio_samples
    d: dim
    t: time_steps

    Args:
        model (nn.Module)
        audio (np.ndarray): (c, l)
        chunk_samples (int)
        min_tail_samples (int)

    Returns:
        out: (d, t)
    """
    device = next(model.parameters()).device
    latents = []
    total_samples = audio.shape[-1]
    i = 0
    
    while i < total_samples:
        remaining_samples = total_samples - i
        if remaining_samples < min_tail_samples:
            break
        
        x = torch.from_numpy(audio[None, :, i : i + chunk_samples]).to(device)

        with torch.no_grad():
            model.eval()
            latent = model(x)[0].data.cpu().numpy()  # (d, t)

        latents.append(latent)
        i += chunk_samples

    return np.concatenate(latents, axis=-1)


def compute_and_save_latents(
    audio: np.ndarray, 
    aug_repeats: int, 
    chunk_samples: int, 
    vae: nn.Module, 
    latent_type: str,
    base_path: str
) -> None:
    r"""Convert audio into latents and write to HDF5.

    c: audio_channels
    l: audio_samples
    d: dim
    t: time_steps

    Args:
        audio (np.ndarray): (c, l)
        aug_repeats (int), number of jitter repetitions for data augmentation

    Returns:
        None
    """
    base_path.parent.mkdir(parents=True, exist_ok=True)

    for i in range(aug_repeats):
                
        jitter = round((i / aug_repeats) * (vae.sr / vae.fps))
        x = audio[:, jitter :]  # (2, l)
        latent = extract_latents_in_chunks(vae, x, chunk_samples)  # (t, d)

        out_path = str(base_path) + f"_{i:03d}_of_{aug_repeats:03d}.h5"
        with h5py.File(out_path, 'w') as hf:
            hf.create_dataset("latent", data=latent, dtype=np.float32)
            hf.attrs.create("fps", data=vae.fps, dtype=float)
            hf.attrs.create("duration", data=x.shape[-1] / vae.sr, dtype=float)
            hf.attrs.create("latent_type", data=latent_type)

        print(f"Write out to {out_path} {latent.shape}")


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


def load_stereo(path: str, sr: int) -> np.ndarray:
    audio, fs = librosa.load(path=path, sr=sr, mono=False)  # (l,)

    if audio.ndim == 1:
        return np.repeat(audio[None, :], repeats=2, axis=0)  # (2, l)
    
    elif audio.ndim == 2:
        if audio.shape[0] == 1:
            return np.repeat(audio[None, :], repeats=2, axis=0)  # (2, l)

        if audio.shape[0] == 2:
            return audio
            
        else:
            return np.repeat(np.mean(audio, axis=0, keepdims=True), repeats=2, axis=0)  # (2, l)


def build_attention_mask(mask):
    r"""Build mask."""
    return mask[:, None, :, None] * mask[:, None, None, :]  # (b, 1, l, l)


def euler_solver(
    model: nn.Module, 
    noise: Tensor, 
    controls: dict, 
    cfg_scale: float,
    n_steps: int
) -> Tensor:

    t = torch.linspace(0, 1, n_steps, device=noise.device)
    x = noise
    
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        dx = model(t[i], x, controls, cfg_scale)   # f(t, x)
        x = x + dt * dx              # Euler update

    return x



# def normalize_text(x: str) -> str:
#     from IPython import embed; embed(using=False); os._exit(0)
#     x = re.sub(r"[^\w\s]", " ", x.lower())  # Remain char and digit only
#     x = re.sub(r"\s+", " ", x)  # Remove extra spaces
#     return x.strip()