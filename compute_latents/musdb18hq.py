import argparse
import os
import random
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn

from audio_flow.encoders.audio.levo_vae import LevoVAE
from audio_flow.utils import compute_and_save_latents


def compute_stereo_vae(args) -> None:

    # Arguments
    root = args.dataset_root
    stem = args.stem
    split = args.split
    latent_type = args.latent_type
    aug_repeats = args.augmentation_repeats
    chunk_duration = args.chunk_duration
    out_dir = args.out_dir
    
    # Parameters
    device = "cuda"

    # Load VAE
    if latent_type == "levo_vae":
        vae = LevoVAE().to(device)
    else:
        raise ValueError(latent_type)

    audio_names = sorted(os.listdir(Path(root, split)))

    for i, name in enumerate(audio_names):
        print("{}/{}, {}".format(i, len(audio_names), name))

        audio_path = Path(root, split, name, f"{stem}.wav")
        audio, _ = librosa.load(path=audio_path, sr=vae.sr, mono=False)  # (2, l)

        chunk_samples = int(chunk_duration * vae.sr)
        base_path = Path(out_dir, name)
        
        compute_and_save_latents(
            audio=audio, 
            aug_repeats=aug_repeats, 
            chunk_samples=chunk_samples, 
            vae=vae, 
            latent_type=latent_type, 
            base_path=base_path
        )


def compute_mono_vae(args) -> None:

    # Arguments
    root = args.dataset_root
    stem = args.stem
    split = args.split
    latent_type = args.latent_type
    aug_repeats = args.augmentation_repeats
    chunk_duration = args.chunk_duration
    out_dir = args.out_dir

    # Parameters
    device = "cuda"

    # Load VAE
    if latent_type == "levo_vae":
        vae = LevoVAE().to(device)
    else:
        raise ValueError(latent_type)

    audio_names = sorted(os.listdir(Path(root, split)))

    for i, name in enumerate(audio_names):
        print("{}/{}, {}".format(i, len(audio_names), name))

        audio_path = Path(root, split, name, f"{stem}.wav")
        audio, _ = librosa.load(path=audio_path, sr=vae.sr, mono=True)  # (2, l)
        audio = audio[None, :].repeat(repeats=2, axis=0)

        chunk_samples = int(chunk_duration * vae.sr)
        base_path = Path(out_dir, name)
        
        compute_and_save_latents(
            audio=audio, 
            aug_repeats=aug_repeats, 
            chunk_samples=chunk_samples, 
            vae=vae, 
            latent_type=latent_type, 
            base_path=base_path
        )


def compute_dac(args) -> None:

    from audio_flow.encoders.audio.dac import DAC

    # Arguments
    root = args.dataset_root
    stem = args.stem
    split = args.split
    latent_type = args.latent_type
    aug_repeats = args.augmentation_repeats
    chunk_duration = args.chunk_duration
    out_dir = args.out_dir

    # Parameters
    device = "cuda"

    # Loaod DAC
    model = DAC(n_quantizers=2).to(device)
    
    # Load VAE
    if latent_type == "levo_vae":
        vae = LevoVAE().to(device)
    else:
        raise ValueError(latent_type)

    audio_names = sorted(os.listdir(Path(root, split)))

    for i, name in enumerate(audio_names):
        print("{}/{}, {}".format(i, len(audio_names), name))

        audio_path = Path(root, split, name, f"{stem}.wav")
        audio, fs = librosa.load(path=audio_path, sr=model.sr, mono=False)  # (2, l)
        audio = audio.mean(axis=0, keepdims=True)

        chunk_samples = int(chunk_duration * model.sr)
        audio = _encode_decode(model, audio, chunk_samples)
        audio = librosa.resample(y=audio, orig_sr=model.sr, target_sr=vae.sr)
        base_path = Path(out_dir, name)

        compute_and_save_latents(
            audio=audio, 
            aug_repeats=aug_repeats, 
            chunk_samples=chunk_samples, 
            vae=vae, 
            latent_type=latent_type, 
            base_path=base_path
        )


def _encode_decode(
    model: nn.Module, 
    audio: np.array, 
    chunk_samples: int,
    min_tail_samples: int = 10000
) -> np.array:
    r"""Encode with DAC and decode."""
    device = next(model.parameters()).device
    outs = []
    total_samples = audio.shape[-1]
    i = 0
    
    while i < total_samples:
        remaining_samples = total_samples - i
        if remaining_samples < min_tail_samples:
            break
        
        x = torch.from_numpy(audio[None, :, i : i + chunk_samples]).to(device)

        with torch.no_grad():
            model.eval()
            latent = model.encode(x)  # (b, d, t)
            out = model.decode(latent)
            out = out[0].cpu().numpy().repeat(repeats=2, axis=0)

        outs.append(out)
        i += chunk_samples

    return np.concatenate(outs, axis=-1)


def compute_lowres_vae(args) -> None:

    # Arguments
    root = args.dataset_root
    stem = args.stem
    split = args.split
    latent_type = args.latent_type
    aug_repeats = args.augmentation_repeats
    chunk_duration = args.chunk_duration
    out_dir = args.out_dir
    
    # Parameters
    lowres_sr_min = 8000
    lowres_sr_max = 8000
    device = "cuda"

    # Load VAE
    if latent_type == "levo_vae":
        vae = LevoVAE().to(device)
    else:
        raise ValueError(latent_type)

    audio_names = sorted(os.listdir(Path(root, split)))

    for i, name in enumerate(audio_names):
        print("{}/{}, {}".format(i, len(audio_names), name))

        audio_path = Path(root, split, name, f"{stem}.wav")
        audio, _ = librosa.load(path=audio_path, sr=vae.sr, mono=False)  # (2, l)

        chunk_samples = int(chunk_duration * vae.sr)
        base_path = Path(out_dir, name)
        lowres_sr = random.uniform(lowres_sr_min, lowres_sr_max)

        audio = librosa.resample(y=audio, orig_sr=vae.sr, target_sr=lowres_sr)
        audio = librosa.resample(y=audio, orig_sr=lowres_sr, target_sr=vae.sr)

        compute_and_save_latents(
            audio=audio, 
            aug_repeats=aug_repeats, 
            chunk_samples=chunk_samples, 
            vae=vae, 
            latent_type=latent_type, 
            base_path=base_path
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    parser_stereo = subparsers.add_parser("stereo")
    parser_stereo.add_argument("--dataset_root", type=str, required=True)
    parser_stereo.add_argument("--stem", type=str, required=True)
    parser_stereo.add_argument("--split", type=str, required=True)
    parser_stereo.add_argument("--latent_type", type=str, default="levo_vae")
    parser_stereo.add_argument("--augmentation_repeats", type=int, default=10)
    parser_stereo.add_argument("--chunk_duration", type=float, default=60.)
    parser_stereo.add_argument("--out_dir", type=str, required=True)
    

    parser_mono = subparsers.add_parser("mono")
    parser_mono.add_argument("--dataset_root", type=str, required=True)
    parser_mono.add_argument("--stem", type=str, required=True)
    parser_mono.add_argument("--split", type=str, required=True)
    parser_mono.add_argument("--latent_type", type=str, default="levo_vae")
    parser_mono.add_argument("--augmentation_repeats", type=int, default=10)
    parser_mono.add_argument("--chunk_duration", type=float, default=60.)
    parser_mono.add_argument("--out_dir", type=str, required=True)

    parser_dac = subparsers.add_parser("dac")
    parser_dac.add_argument("--dataset_root", type=str, required=True, help="Path of config yaml.")
    parser_dac.add_argument("--stem", type=str, required=True)
    parser_dac.add_argument("--split", type=str, required=True)
    parser_dac.add_argument("--latent_type", type=str, default="levo_vae")
    parser_dac.add_argument("--augmentation_repeats", type=int, default=10)
    parser_dac.add_argument("--chunk_duration", type=float, default=60.)
    parser_dac.add_argument("--out_dir", type=str, required=True)

    parser_lowres = subparsers.add_parser("lowres")
    parser_lowres.add_argument("--dataset_root", type=str, required=True, help="Path of config yaml.")
    parser_lowres.add_argument("--stem", type=str, required=True)
    parser_lowres.add_argument("--split", type=str, required=True)
    parser_lowres.add_argument("--latent_type", type=str, default="levo_vae")
    parser_lowres.add_argument("--augmentation_repeats", type=int, default=10)
    parser_lowres.add_argument("--chunk_duration", type=float, default=60.)
    parser_lowres.add_argument("--out_dir", type=str)

    args = parser.parse_args()
    
    if args.mode == "stereo":
        compute_stereo_vae(args)

    elif args.mode == "mono":
        compute_mono_vae(args)

    elif args.mode == "dac":
        compute_dac(args)

    elif args.mode == "lowres":
        compute_lowres_vae(args)

    else:
        raise ValueError