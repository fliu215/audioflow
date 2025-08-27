from __future__ import annotations

from pathlib import Path
import pickle
import os
import h5py

import librosa
import numpy as np
import pandas as pd
from audidata.io.audio import load
from audidata.io.crops import StartCrop
from audidata.transforms.audio import Mono
from audidata.utils import call
from torch.utils.data import Dataset
from typing_extensions import Literal


class LJSpeechVAE(Dataset):
    
    def __init__(
        self, 
        root: str = None, 
        split: Literal["train", "valid" "test"] = "train",
        duration: float = 10
    ) -> None:
    
        self.root = root
        self.split = split
        self.duration = duration

        self.meta_dict = self.load_meta()

    def __getitem__(self, index: int) -> dict:

        audio_path = self.meta_dict["audio_path"][index]
        caption = self.meta_dict["caption"][index]

        full_data = {
            "dataset_name": "LJSpeechVAE",
            "audio_path": audio_path,
        }

        # Load audio data
        audio_data = self.load_latent_data(path=audio_path)
        full_data.update(audio_data)

        # Load target data
        target_data = self.load_target_data(caption=caption)
        full_data.update(target_data)

        return full_data

    def __len__(self) -> int:
        return len(self.meta_dict["audio_name"])

    def load_meta(self) -> dict:
        r"""Load metadata of the GTZAN dataset.
        """

        # Load split file
        split_path = Path(self.root, "{}.txt".format(self.split))
        df = pd.read_csv(split_path, header=None)
        split_names = df[0].values

        # Load csv file
        csv_path = Path(self.root, "metadata.csv")
        df = pd.read_csv(csv_path, sep="|", header=None)
        names = df[0].values
        captions = df[1].values

        # Get split indexes
        idxes = []
        for i in range(len(names)):
            if names[i] in split_names:
                idxes.append(i)

        audios_num = len(os.listdir(Path(self.root, "wavs")))
        num_repeats = audios_num // len(captions)

        all_names = []
        all_captions = []
        all_paths = []

        for i in idxes:
            all_names.extend([names[i]] * num_repeats)
            all_captions.extend([captions[i]] * num_repeats)
            all_paths.extend([str(Path(self.root, "wavs", f"{names[i]}_{i:03d}_vae.h5")) for i in range(num_repeats)])
        
        meta_dict = {
            "audio_name": all_names,
            "audio_path": all_paths,
            "caption": all_captions
        }

        return meta_dict

    def load_latent_data(self, path: str) -> dict:

        with h5py.File(path, 'r') as hf:
            latent = hf["latent"][:]
            fps = hf.attrs["fps"]
        
        data = {
            "latent": latent,
            "fps": fps
        }
        
        return data

    def load_target_data(self, caption: str) -> dict:

        target = caption

        data = {
            "caption": caption,
            "target": target
        }

        return data