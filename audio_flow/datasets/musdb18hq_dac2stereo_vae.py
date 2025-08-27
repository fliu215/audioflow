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


class MUSDB18HqDac2StereoVAE(Dataset):

    def __init__(
        self,
        root: str, 
        split: Literal["train", "test"] = "train",
        duration: float = 10.,
    ) -> None:
    
        self.root = root
        self.split = split
        self.duration = duration

        self.meta_dict = self.load_meta()

    def __getitem__(
        self, 
        index: int,
    ) -> dict:

        full_data = {
            "dataset_name": "MUSDB18HqMono2StereoVAE",
        }

        dac_path = self.meta_dict["dac_path"][index]
        vae_path = self.meta_dict["vae_path"][index]

        data = self.load_latent_data(dac_path, vae_path)
        full_data.update(data)
        
        return full_data

    def __len__(self) -> int:
        return len(self.meta_dict["vae_path"])

    def load_meta(self):

        vae_paths = sorted(list(Path(self.root, self.split).rglob('*vae.h5')))
        vae_paths = [str(s) for s in vae_paths]
        dac_paths = [s.replace("vae.h5", "dac.h5") for s in vae_paths]

        meta_dict = {
            "dac_path": dac_paths,
            "vae_path": vae_paths
        }
        
        return meta_dict
        
    def load_latent_data(self, mono_path: str, vae_path: str) -> dict:

        with h5py.File(mono_path, 'r') as hf:
            dac_code = hf["code"][:]
            dac_fps = hf.attrs["fps"]

        with h5py.File(vae_path, 'r') as hf:
            vae_latent = hf["latent"][:]
            vae_fps = hf.attrs["fps"]

        vae_frames = int(self.duration * vae_fps)
        vae_bgn = random.randint(0, vae_latent.shape[-1] - vae_frames)
        vae_bgn = max(0, vae_bgn)
        vae_latent = vae_latent[:, vae_bgn : vae_bgn + vae_frames]  # (d, t)

        dac_frames = round(vae_frames * (dac_fps / vae_fps))
        dac_bgn = round(vae_bgn * (dac_fps / vae_fps))
        dac_code = dac_code[:, dac_bgn : dac_bgn + dac_frames]  # (d, t)

        if dac_code.shape[-1] < dac_frames:
            dac_code = librosa.util.fix_length(dac_code, size=dac_frames, axis=-1, mode="edge")

        data = {
            "dac_code": dac_code,
            "vae_latent": vae_latent,
            "dac_fps": dac_fps,
            "vae_fps": vae_fps
        }

        return data