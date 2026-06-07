import random

import h5py
import librosa
import numpy as np

from audioflow.utils.xml import dict_to_xml
from audioflow.utils.audio import get_latent_length, sample_start_frame, load_latent


class TTMDataset:
    def __init__(self, clip_duration: float) -> None:
        self.clip_duration = clip_duration

    def __getitem__(self, meta: dict) -> dict:
        r"""Get text to music data."""

        xml = dict_to_xml(meta["input"]["text"])
        prompt = "".join(xml)
        latent_path = meta["target"]["audio"]["path"]
        fps = meta["target"]["audio"]["fps"]
        
        clip_frames = round(self.clip_duration * fps)
        total_frames = get_latent_length(latent_path)
        start = sample_start_frame(total_frames, clip_frames)
        latent, mask, length = load_latent(latent_path, start, clip_frames)
        # latent: (l, d), mask: (l,)

        data = {
            "prompt": prompt, 
            "target_latent": latent,  # (l, d)
            "target_mask": mask,  # (l,)
            "target_length": length
        }
        
        return data


