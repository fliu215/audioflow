import argparse
import os
import random
import pandas as pd
from pathlib import Path

import h5py
import librosa
import numpy as np
import torch
import torch.nn as nn
from torchvision.io import read_video
from transformers import CLIPProcessor, CLIPModel

from audio_flow.encoders.audio.levo_vae import LevoVAE
from audio_flow.utils import compute_and_save_latents




def compute_audio_vae(args) -> None:

    # Arguments
    root = args.dataset_root
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

    csv_path = Path(root, f"{split}Set.txt")
    meta_dict = load_meta(csv_path)
    n_data = len(meta_dict["name"])
    
    for n in range(n_data):

        name = meta_dict["name"][n]
        path = Path(root, "AVE", f"{name}.mp4")

        audio, _ = librosa.load(path=path, sr=vae.sr, mono=False)  # (2, l)
        if audio.ndim == 1:
            audio = np.repeat(audio[None, :], repeats=2, axis=0)  # (2, l)

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
        

def compute_video_vae(args) -> None:

    # Arguments
    root = args.dataset_root
    split = args.split
    latent_type = args.latent_type
    aug_repeats = args.augmentation_repeats
    chunk_duration = args.chunk_duration
    out_dir = args.out_dir
    
    # Parameters
    fps = 25
    device = "cuda"

    # Load VAE
    if latent_type == "clip":
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    else:
        raise ValueError(latent_type)

    csv_path = Path(root, f"{split}Set.txt")
    meta_dict = load_meta(csv_path)
    n_data = len(meta_dict["name"])
    
    for n in range(n_data):

        i = 0
        name = meta_dict["name"][n]
        path = Path(root, "AVE", f"{name}.mp4")

        resampled_path = "_tmp.mp4"
        os.system(f'ffmpeg -y -i {path} -vf "minterpolate=fps={fps}" {resampled_path} -loglevel quiet') 
        video, _, info = read_video(resampled_path, output_format="TCHW", pts_unit="sec")

        latent = _compute_video_latent(processor, model, video)
        base_path = Path(out_dir, name)

        out_path = str(base_path) + f"_{i:03d}_of_{aug_repeats:03d}.h5"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out_path, 'w') as hf:
            hf.create_dataset("latent", data=latent, dtype=np.float32)
            hf.attrs.create("fps", data=fps, dtype=float)
            hf.attrs.create("duration", data=latent.shape[-1] / fps, dtype=float)
            hf.attrs.create("latent_type", data=latent_type)

        print(f"{n}/{n_data} Write out to {out_path} {latent.shape}")


def load_meta(csv_path: str) -> dict:
    df = pd.read_csv(csv_path, sep="&", header=None)
    meta_dict = {
        "label": df[0].values,
        "name": df[1].values
    }
    return meta_dict
        

def _compute_video_latent(processor, model, video):
    i = 0
    batch_size = 4
    total_frames = video.shape[0]
    device = next(model.parameters()).device
    latents = []
    while i < total_frames:
        x = video[i : i + batch_size]
        with torch.no_grad():
            model.eval()
            inputs = processor(
                text=[""],
                images=x,
                return_tensors="pt",
                padding=True
            ).to(device)
            outputs = model(**inputs)

        latent = outputs.image_embeds.cpu().numpy()
        latents.append(latent)
        i += batch_size
    latents = np.concatenate(latents)
    return latents


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    parser_audio = subparsers.add_parser("audio")
    parser_audio.add_argument("--dataset_root", type=str, required=True)
    parser_audio.add_argument("--split", type=str, required=True)
    parser_audio.add_argument("--latent_type", type=str, default="levo_vae")
    parser_audio.add_argument("--augmentation_repeats", type=int, default=1)
    parser_audio.add_argument("--chunk_duration", type=float, default=60.)
    parser_audio.add_argument("--out_dir", type=str, required=True)

    parser_video = subparsers.add_parser("video")
    parser_video.add_argument("--dataset_root", type=str, required=True)
    parser_video.add_argument("--split", type=str, required=True)
    parser_video.add_argument("--latent_type", type=str, default="clip")
    parser_video.add_argument("--augmentation_repeats", type=int, default=1)
    parser_video.add_argument("--chunk_duration", type=float, default=60.)
    parser_video.add_argument("--out_dir", type=str, required=True)
    
    args = parser.parse_args()
    
    if args.mode == "audio":
        compute_audio_vae(args)

    elif args.mode == "video":
        compute_video_vae(args)

    else:
        raise ValueError