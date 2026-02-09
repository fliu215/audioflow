import random

import h5py
import librosa
import numpy as np


class MetaDataset:
    def __init__(self, clip_duration: float):
        self.clip_duration = clip_duration

    def __getitem__(self, meta: dict) -> dict:
        r"""Load data from meta."""
        task = meta["task"]

        if task == "text_to_music":
            return get_ttm_data(meta, self.clip_duration)

        elif task == "text_to_speech":
            return get_tts_data(meta, self.clip_duration)

        elif task == "text_to_audio":
            return get_tta_data(meta, self.clip_duration)

        elif task in ["music_source_separation", "mono_to_stereo", "super-resolution", "codec_to_music"]:
            return get_aligned_data(meta, self.clip_duration)

        else:
            raise ValueError(task)


def get_ttm_data(meta: dict, clip_duration: float) -> dict:
    r"""Get text to music data."""
    task = meta["task"]
    prompt = random.choice(meta["input"]["text"]["prompt"])
    path = meta["target"]["audio"]["path"]
    fps = meta["target"]["audio"]["fps"]
    total_frames = meta["target"]["audio"]["num_frames"]
    clip_frames = round(clip_duration * fps)

    bgn = random_bgn_frame(total_frames, clip_frames)
    latent, length = load_latent(path, bgn, clip_frames)

    data = {
        "task": task,
        "prompt": prompt,
        "target_audio_latent": latent,
        "latent_length": length
    }
    return data


def get_tts_data(meta: dict, clip_duration: float) -> dict:
    r"""Get text to speech data."""
    task = meta["task"]
    content = meta["input"]["text"]["content"]
    path = meta["target"]["audio"]["path"]
    fps = meta["target"]["audio"]["fps"]
    total_frames = meta["target"]["audio"]["num_frames"]
    clip_frames = round(clip_duration * fps)

    bgn = random_bgn_frame(total_frames, clip_frames)
    latent, length = load_latent(path, bgn, clip_frames)

    data = {
        "task": task,
        "content": content,
        "target_audio_latent": latent,
        "latent_length": length
    }
    return data


def get_tta_data(meta: dict, clip_duration: float) -> dict:
    r"""Get text to audio data."""
    task = meta["task"]
    prompt = random.choice(meta["input"]["text"]["prompt"])
    path = meta["target"]["audio"]["path"]
    fps = meta["target"]["audio"]["fps"]
    total_frames = meta["target"]["audio"]["num_frames"]
    clip_frames = round(clip_duration * fps)

    bgn = random_bgn_frame(total_frames, clip_frames)
    latent, length = load_latent(path, bgn, clip_frames)

    data = {
        "task": task,
        "prompt": prompt,
        "target_audio_latent": latent,
        "latent_length": length
    }
    return data


def get_aligned_data(meta: dict, clip_duration: float) -> dict:
    r"""Get input and target aligned data."""
    assert meta["input"]["audio"]["latent_type"] == meta["target"]["audio"]["latent_type"]

    task = meta["task"]
    instruction = meta["input"]["text"]["instruction"]
    input_path = meta["input"]["audio"]["path"]
    target_path = meta["target"]["audio"]["path"]
    fps = meta["target"]["audio"]["fps"]
    total_frames = meta["target"]["audio"]["num_frames"]
    clip_frames = round(clip_duration * fps)

    bgn = random_bgn_frame(total_frames, clip_frames)
    input_latent, _ = load_latent(input_path, bgn, clip_frames)
    target_latent, length = load_latent(target_path, bgn, clip_frames)

    data = {
        "task": task,
        "instruction": instruction,
        "input_audio_latent": input_latent,
        "target_audio_latent": target_latent,
        "latent_length": length
    }
    return data


def random_bgn_frame(total_frames: int, clip_frames: int) -> int:
    r"""Random sample a frame index."""
    if clip_frames <= total_frames:
        return random.randint(0, total_frames - clip_frames)
    else:
        return 0


def load_latent(path: str, bgn: int, clip_frames: int) -> tuple[np.ndarray, int]:
    r"""Load latent from hdf5."""
    with h5py.File(path, 'r') as hf:
        latent = hf["latent"][:, bgn : bgn + clip_frames]
        length = latent.shape[-1]
        latent = librosa.util.fix_length(data=latent, size=clip_frames, axis=-1, mode="edge")

    return latent, length