from __future__ import annotations

import h5py
import numpy as np
import random
import librosa
import torch
import torch.nn as nn

from audioflow.utils.misc import augment_path


def load_stereo(path: str, sr: int) -> np.ndarray:
    audio, fs = librosa.load(path=path, sr=sr, mono=False)

    if audio.ndim == 1:
        return np.repeat(audio[None, :], repeats=2, axis=0)  # (2, l)
    
    elif audio.ndim == 2:
        if audio.shape[0] == 1:
            return np.repeat(audio[None, :], repeats=2, axis=0)  # (2, l)

        elif audio.shape[0] == 2:
            return audio
            
        else:
            return np.repeat(np.mean(audio, axis=0, keepdims=True), repeats=2, axis=0)  # (2, l)


# def get_latent_length(path) -> int:
#     with h5py.File(path, 'r') as hf:
#         return hf["data"].shape[0]


# def sample_start_frame(total_frames: int, clip_frames: int) -> int:
#     r"""Random sample a frame index."""
#     max_start = max(total_frames - clip_frames, 0)
#     return random.randint(0, max_start)


# def load_latent(path, start: int, clip_frames: int) -> tuple[np.ndarray, np.ndarray]:
#     r"""Load latent from hdf5."""
#     with h5py.File(path, 'r') as hf:
#         x = hf["data"][start : start + clip_frames, :]  # (l, d)
#         length = x.shape[0]

#         x = librosa.util.fix_length(data=x, size=clip_frames, axis=0, constant_values=0.)  # (l, d)
#         mask = np.zeros(clip_frames, dtype=bool)  # (l,)
#         mask[:length] = True

#     return x, mask


def extract_and_save_audio_features(
    audio: np.ndarray, 
    aug_repeats: int, 
    chunk_samples: int, 
    model: nn.Module, 
    encoder_name: str,
    out_path: str,
    dtype=np.float32
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
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for i in range(aug_repeats):
                
        jitter = round((i / aug_repeats) * (model.sr / model.fps))
        x = audio[:, jitter :]  # (c, l)
        feat = extract_features_in_chunks(model, x, chunk_samples)  # (t, d)

        aug_path = augment_path(out_path, i)

        with h5py.File(aug_path, 'w') as hf:
            hf.create_dataset("data", data=feat, dtype=dtype)
            hf.attrs.create("fps", data=model.fps, dtype=float)
            hf.attrs.create("duration", data=x.shape[-1] / model.sr, dtype=float)
            hf.attrs.create("type", data=encoder_name)

        print(f"Write out to {aug_path} {feat.shape}")


def extract_features_in_chunks(
    model: nn.Module, 
    audio: np.ndarray, 
    chunk_samples: int,
    min_tail_samples: int = 1000
) -> np.array:
    r"""Convert audio into features.

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
        outs: (d, t)
    """
    device = next(model.parameters()).device
    outs = []
    i = 0
    
    while i <= audio.shape[-1] - min_tail_samples:
        
        x = torch.from_numpy(audio[None, :, i : i + chunk_samples]).to(device)  # (b, c, l)
        print(x.shape)

        with torch.no_grad():
            model.eval()
            out = model(x)[0].data.cpu().numpy()  # (d, t)

        outs.append(out)
        i += chunk_samples

    return np.concatenate(outs, axis=0)