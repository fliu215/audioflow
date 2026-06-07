from __future__ import annotations

import numpy as np
import librosa


def augment_path(path: Path, i: int) -> Path:
    if i == 0:
        return path
    else:
        return path.parent / f"{path.stem}.aug{i:04d}{path.suffix}"


def get_single_value(lst: list):
    unique = list(set(lst))
    assert len(unique) == 1
    return unique[0]


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