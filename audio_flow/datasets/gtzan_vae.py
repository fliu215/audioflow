r"""Code modified from: https://github.com/AudioFans/audidata/blob/main/audidata/datasets/gtzan.py"""
from __future__ import annotations

import os
import re
from pathlib import Path

import h5py
import random
import pickle
import librosa
import numpy as np
from audidata.io.audio import load
from audidata.io.crops import StartCrop
from audidata.transforms.audio import Mono
from audidata.transforms.onehot import OneHot
from audidata.utils import call
from torch.utils.data import Dataset
from typing_extensions import Literal


class GtzanVAE(Dataset):
    r"""GTZAN [1] is a music dataset containing 1,000 30-second audio clips. 
    The total duration is 8.3 hours. GTZAN includes 10 genres. All audio files 
    are mono sampled at 22,050 Hz. After decompression, the dataset size is 
    1.3 GB.

    [1] Tzanetakis, G., et al., Musical genre classification of audio signals. 2002

    The dataset looks like:

        gtzan (1.3 GB)
        └── genres
            ├── blues (100 files)
            ├── classical (100 files)
            ├── country (100 files)
            ├── disco (100 files)
            ├── hiphop (100 files)
            ├── jazz (100 files)
            ├── metal (100 files)
            ├── pop (100 files)
            ├── reggae (100 files)
            └── rock (100 files)
    """

    LABELS = ["blues", "classical", "country", "disco", "hiphop", "jazz", 
        "metal", "pop", "reggae", "rock"]

    CLASSES_NUM = len(LABELS)
    LB_TO_IX = {lb: ix for ix, lb in enumerate(LABELS)}
    IX_TO_LB = {ix: lb for ix, lb in enumerate(LABELS)}

    def __init__(
        self, 
        root: str = None, 
        split: Literal["train", "test"] = "train",
        test_fold: int = 0,  # E.g., fold 0 is used for testing. Fold 1 - 9 are used for training.
        duration: float = 10.
    ) -> None:
    
        self.root = root
        self.split = split
        self.test_fold = test_fold
        self.duration = duration
        
        self.labels = GtzanVAE.LABELS
        self.lb_to_ix = GtzanVAE.LB_TO_IX
        self.ix_to_lb = GtzanVAE.IX_TO_LB

        self.meta_dict = self.load_meta()

    def __getitem__(self, index: int) -> dict:

        path = str(self.meta_dict["path"][index])
        label = self.meta_dict["label"][index]

        full_data = {
            "dataset_name": "GtzanVAE",
            "path": path,
        }

        # Load audio data
        latent_data = self.load_latent_data(path=path)
        full_data.update(latent_data)

        # Load target data
        target_data = self.load_target_data(label=label)
        full_data.update(target_data)

        return full_data

    def __len__(self) -> int:
        return len(self.meta_dict["name"])

    def load_meta(self) -> dict:
        r"""Load metadata of the GTZAN dataset.
        """

        meta_dict = {
            "name": [],
            "path": [],
            "label": [],
        }

        out_dir = Path(self.root, "genres")

        for genre in self.labels:

            names = sorted(os.listdir(Path(out_dir, genre)))
            train_names, test_names = self.split_train_test(names)

            if self.split == "train":
                filtered_names = train_names

            elif self.split == "test":
                filtered_names = test_names

            else:
                raise ValueError(self.split)

            for name in filtered_names:
                path = str(Path(out_dir, genre, name))
                meta_dict["name"].append(name)
                meta_dict["path"].append(path)
                meta_dict["label"].append(genre)

        return meta_dict

    def split_train_test(self, names: list) -> tuple[list, list]:

        train_names = []
        test_names = []

        test_ids = range(self.test_fold * 10, (self.test_fold + 1) * 10)
        # E.g., if test_fold = 3, then test_ids = [30, 31, 32, ..., 39]

        for name in names:

            audio_id = int(re.search(r'\d+', name).group())
            # E.g., if name is "blues.00037.h5", then audio_id = 37

            if audio_id in test_ids:
                test_names.append(name)

            else:
                train_names.append(name)

        return train_names, test_names

    def load_latent_data(self, path: str) -> dict:

        with h5py.File(path, 'r') as hf:
            latent = hf["latent"][:]
            fps = hf.attrs["fps"]

        clip_frames = int(self.duration * fps)
        bgn_frame = random.randint(0, latent.shape[-1] - clip_frames)
        bgn_frame = max(0, bgn_frame)    
        latent = latent[:, bgn_frame : bgn_frame + clip_frames]  # (d, t)

        data = {
            "latent": latent,
            "fps": fps
        }

        return data

    def load_target_data(self, label: str) -> dict:

        target = self.lb_to_ix[label]

        data = {
            "label": label,
            "target": target  # shape: (classes_num,)
        }

        return data