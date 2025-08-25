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
   
    def __init__(
        self,
        root: str, 
        split: Literal["train", "test"] = "train",
        duration: float = 10.,
        target_stem: str = "vocals",
    ) -> None:

        self.root = root
        self.split = split
        self.duration = duration
        self.target_stem = target_stem

        self.meta_dict = self.load_meta()

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

        mixture_path = self.meta_dict["mixture_path"][index]
        target_path = self.meta_dict["target_path"][index]

        data = self.load_latent_data(mixture_path, target_path)
        full_data.update(data)

        return full_data

    def __len__(self) -> int:
        return len(self.meta_dict["mixture_path"])

    def load_meta(self):

        mixture_paths = sorted(list(Path(self.root, self.split).rglob("*mixture_*")))
        mixture_paths = [str(s) for s in mixture_paths]
        target_paths = [s.replace("mixture_", "{}_".format(self.target_stem)) for s in mixture_paths]

        meta_dict = {
            "mixture_path": mixture_paths,
            "target_path": target_paths
        }
        
        return meta_dict

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