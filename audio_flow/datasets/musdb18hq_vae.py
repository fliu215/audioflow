r"""Code from https://github.com/AudioFans/audidata/blob/main/audidata/datasets/musdb18hq.py"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Union
import pickle
import random
import h5py

import librosa
import numpy as np
from audidata.io.audio import load
from audidata.io.crops import RandomCrop
from torch.utils.data import Dataset
from typing_extensions import Literal


class MUSDB18HqVAE(Dataset):
    r"""MUSDB18HQ [1] is a dataset containing 100 training audio files and 50 
    testing audio files, each with vocals, bass, drums, and other stems. The 
    total duration is 9.8 hours. The audio is stereo and sampled at 48,000 Hz. 
    After decompression, the dataset size is 30 GB.

    [1] https://zenodo.org/records/3338373

    The dataset looks like:

        musdb18hq (30 GB)
        ├── train (100 files)
        │   ├── A Classic Education - NightOwl
        │   │   ├── bass.wav
        │   │   ├── drums.wav
        │   │   ├── mixture.wav
        │   │   ├── other.wav
        │   │   └── vocals.wav
        │   ... 
        │   └── ...
        └── test (50 files)
            ├── Al James - Schoolboy Facination
            │   ├── bass.wav
            │   ├── drums.wav
            │   ├── mixture.wav
            │   ├── other.wav
            │   └── vocals.wav
            ... 
            └── ...
    """

    def __init__(
        self,
        root: str, 
        split: Literal["train", "test"] = "train",
        duration: float = 10.,
        target_stem: str = "vocals",
    ) -> None:
        r"""
        time_align: str. "strict" indicates all stems are aligned (from the 
            same song and have the same start time). "group" indictates 
            target stems / background stems are aligned. "random" indicates 
            all stems are from different songs with different start time.
        """

        self.root = root
        self.split = split
        self.duration = duration
        self.target_stem = target_stem

        self.latents_dir = Path(self.root, self.split)
        self.names = sorted(os.listdir(self.latents_dir))
        self.latents_num = len(self.names)

    def __getitem__(
        self, 
        index: int,
    ) -> dict:

        audio_names = {}
        audio_paths = {}
        start_times = {}
        clip_durations = {}

        full_data = {
            "dataset_name": "MUSDB18HqVAE",
        }

        # mixture_name = "{}.h5".format(self.names[index])
        mixture_path = Path(self.latents_dir, self.names[index], "mixture.h5")
        target_path = Path(self.latents_dir, self.names[index], f"{self.target_stem}.h5")
        
        data = self.load_latent_data(mixture_path, target_path)
        full_data.update(data)
        
        return full_data

    def __len__(self) -> int:
        return self.latents_num

    def load_latent_data(self, mixture_path: str, target_path) -> dict:

        with h5py.File(mixture_path, 'r') as hf:
            mixture_latent = hf["latent"][:]
            fps = hf.attrs["fps"]

        with h5py.File(target_path, 'r') as hf:
            target_latent = hf["latent"][:]

        total_frames = mixture_latent.shape[-1]
        clip_frames = int(self.duration * fps)
        bgn_frame = random.randint(0, total_frames - clip_frames)
        bgn_frame = max(0, bgn_frame)

        mixture_latent = mixture_latent[:, bgn_frame : bgn_frame + clip_frames]  # (d, t)
        target_latent = target_latent[:, bgn_frame : bgn_frame + clip_frames]  # (d, t)

        data = {
            "mixture_latent": mixture_latent,
            "target_latent": target_latent,
            "fps": fps
        }

        return data