from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import soundfile
import torch
import torch.nn as nn
import torch.optim as optim
import torchdiffeq
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from tqdm import tqdm

import wandb
from audio_flow.adapters.adapter import Adapter
from audio_flow.datasets.dataset import MetaDataset
from audio_flow.encoders.audio.levo_vae import LevoVAE
from audio_flow.samplers.jsonl_sampler import JsonlSampler
from audio_flow.utils import (CombinedModel, LinearWarmUp, get_single_value,
                              load_jsonl, logmel, parse_yaml, requires_grad,
                              update_ema, normalize_text)
from train import get_model


def sample(args) -> None:
    r"""Train audio generation with flow matching."""

    # Arguments
    config_path = args.config
    ckpt_path = args.ckpt_path
    out_path = args.out_path
    duration = 10.
    
    # Configs
    configs = parse_yaml(config_path)
    device = configs["train"]["device"]

    # Load model
    model = get_model(configs, ckpt_path).to(device)
    
    # Load VAE
    vae = LevoVAE().to(device)

    # Prepare meta data
    data = get_data(args)
    data = default_collate([data])  # create a batch

    # Noise
    length = round(duration * vae.fps)
    noise = torch.randn(1, vae.dim, length).to(device)
    
    with torch.no_grad():
        model.eval()
        c = model.adapter(data, length).to(device)
        traj = torchdiffeq.odeint(
            lambda t, x: model.base(t, x, c),
            y0=noise,
            t=torch.linspace(0, 1, 2, device=device),
            atol=1e-4,
            rtol=1e-4,
            method="dopri5",
        )
        x_gen = traj[-1]  # (b, t, d)
    
    # Decode audio from VAE latents
    audio_gen = vae.decode(x_gen).data.cpu().numpy()[0]  # (c, l)

    # Write out
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(file=out_path, data=audio_gen.T, samplerate=vae.sr)
    print(f"Write out to {out_path}")


def get_data(args):

    task = args.task

    if task in ["text_to_speech"]:
        return {
            "task": task, 
            "content": args.prompt
        }

    elif task in ["text_to_music", "text_to_audio"]:
        return {
            "task": task, 
            "prompt": args.prompt
        }

    elif task == ["music_source_separation", "mono_to_stereo", "super-resolution", 
        "codec_to_music"]:
        pass
        # TODO
        # vae = LevoVAE().to(device)
        # audio, fs = librosa.load(path=path, sr=vae.sr, mono=False)  # (l,)
        # audio = np.repeat(audio[None, :], repeats=2, axis=0)  # (2, l)
        
        # chunk_samples = int(chunk_duration * vae.sr)
        # base_path = Path(out_dir, path.stem)
        
        # compute_and_save_latents(
        #     audio=audio, 
        #     aug_repeats=aug_repeats, 
        #     chunk_samples=chunk_samples, 
        #     vae=vae, 
        #     latent_type=latent_type, 
        #     base_path=base_path
        # )

        # data = {
        #     "task": task, 
        #     "prompt": args.prompt
        # }

    else:
        raise ValueError(task)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--audio", type=str)
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()

    sample(args)