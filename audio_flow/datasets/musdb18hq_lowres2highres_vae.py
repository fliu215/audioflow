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


class MUSDB18HqLowres2HighresVAE(Dataset):

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
            "dataset_name": "MUSDB18HqLowres2HighresVAE",
        }

        mono_path = self.meta_dict["mono_path"][index]
        stereo_path = self.meta_dict["stereo_path"][index]

        data = self.load_latent_data(mono_path, stereo_path)
        full_data.update(data)
        
        return full_data

    def __len__(self) -> int:
        return len(self.meta_dict["mono_path"])

    def load_meta(self):

        from IPython import embed; embed(using=False); os._exit(0)

        input_paths = sorted(list(Path(self.root, self.split).rglob('*lowres_vae.h5')))
        target_paths = [str(s) for s in input_paths]
        target_paths = [s.replace("lowres_vae.h5", "highres_vae.h5") for s in input_paths]

        meta_dict = {
            "mono_path": mono_paths,
            "stereo_path": stereo_paths
        }
        
        return meta_dict
        
    def load_latent_data(self, mono_path: str, stereo_path: str) -> dict:

        with h5py.File(mono_path, 'r') as hf:
            mono_latent = hf["latent"][:]
            fps = hf.attrs["fps"]

        with h5py.File(stereo_path, 'r') as hf:
            stereo_latent = hf["latent"][:]

        total_frames = mono_latent.shape[-1]
        clip_frames = int(self.duration * fps)
        bgn_frame = random.randint(0, total_frames - clip_frames)
        bgn_frame = max(0, bgn_frame)

        mono_latent = mono_latent[:, bgn_frame : bgn_frame + clip_frames]  # (d, t)
        stereo_latent = stereo_latent[:, bgn_frame : bgn_frame + clip_frames]  # (d, t)

        data = {
            "mono_latent": mono_latent,
            "stereo_latent": stereo_latent,
            "fps": fps
        }

        return data