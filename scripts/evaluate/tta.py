from __future__ import annotations

import argparse
from pathlib import Path

import os
import librosa
import soundfile
import torch
from torch import Tensor
import torchdiffeq
from torch.utils.data._utils.collate import default_collate
import numpy as np
import math
import pandas as pd
from audioflow.solvers import get_solver
from audioflow.inference.generate import generate_latent

from audioflow.utils.yaml import read_yaml
from audioflow.models import get_model
from audioflow.encoders.audio import load_encoder


def sample(args) -> None:
    r"""Train audio generation with flow matching."""

    # Arguments
    config_path = Path(args.config)
    ckpt_path = Path(args.ckpt_path)
    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.out_dir)
    duration = args.duration
    
    root = Path("/datasets/audiocaps")
    csv_path = Path("./assets/test-audiocaps.tsv")
    audios_dir = root / "test"

    # Configs
    configs = read_yaml(config_path)
    device = configs["train"]["device"]

    # Load model
    model = get_model(configs["model"], ckpt_path).to(device)
    
    # Load VAE
    vae = load_encoder(configs["valid"]["vae"]).to(device)

    # Solver
    solver = get_solver({"name": "euler", "steps": 100})

    # Cfg
    cfg_scale = configs["cfg"]["sample"]["scale"] if "cfg" in configs else None

    meta_dict = load_meta(csv_path)
    n_data = len(meta_dict["youtube_id"])

    for n in range(n_data):
        
        youtube_id = meta_dict["youtube_id"][n]
        audio_path = audios_dir / f"{youtube_id}.wav"

        # Noise
        length = round(duration * vae.fps)
        noise = torch.randn(1, length, vae.dim).to(device)
        
        caption = meta_dict["caption"][n]
        prompt = f"<audio>{caption}</audio>"
        data = get_data(prompt, length)
        data = default_collate([data])

        x_gen = generate_latent(model, noise, data, solver, cfg_scale)  # (b, l, d)

        # Decode audio from VAE latents
        audio_gen = vae.decode(x_gen).cpu().numpy()[0, 0]  # (c, l)
        audio_gen = librosa.util.fix_length(data=audio_gen, size=int(vae.sr * duration), axis=-1)
        
        # Write out
        out_path = out_dir / f"{youtube_id}.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(file=out_path, data=audio_gen, samplerate=vae.sr)
        print(f"Write out to {out_path}")

        audio_gt, _ = librosa.load(path=audio_path, sr=vae.sr, mono=True)
        audio_gt, orig_sr = librosa.load(path=audio_path)
        audio_gt = librosa.util.fix_length(data=audio_gt, size=int(orig_sr * duration), axis=-1)

        gt_path = gt_dir / f"{youtube_id}.wav"
        Path(gt_path).parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(file=gt_path, data=audio_gt, samplerate=orig_sr)
        print(f"Write out to {gt_path}")

        # if n == 10:
        #     break

        # from IPython import embed; embed(using=False); os._exit(0)

def load_meta(csv_path) -> dict:

    df = pd.read_csv(csv_path, sep='\t')
    meta_dict = {
        "youtube_id": df["id"].values,
        "caption": df["caption"].values,
    }
    return meta_dict


def get_data(prompt: str, length: int) -> dict:

    return {
        "prompt": prompt,
        "target_mask": np.ones(length, dtype=bool)
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--duration", type=float, default=10.)
    
    args = parser.parse_args()
    
    sample(args)