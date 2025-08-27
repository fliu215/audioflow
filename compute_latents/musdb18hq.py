import argparse
import os
import random
import time
from pathlib import Path

import h5py
import librosa
import torch
import numpy as np

from audio_flow.utils import forward_in_chunks
from audio_flow.vae.levo import LevoVAE
from audio_flow.encoders.dac import DAC


def compute_stems_vae(args) -> None:

    root = args.dataset_root
    out_dir = args.out_dir
    aug_repeats = args.augmentation_repeats

    device = "cuda"
    clip_duration = 60.

    # Load VAE
    vae = LevoVAE().to(device)

    clip_samples = int(clip_duration * vae.sr)

    # Compuate VAE latent
    stems = ["vocals", "bass", "drums", "other", "mixture"]

    for split in ["train", "test"]:

        audio_names = sorted(os.listdir(Path(root, split)))

        for i, name in enumerate(audio_names):

            print("{}/{}, {}".format(i, len(audio_names), name))

            for stem in stems:

                path = Path(root, split, name, f"{stem}.wav")
                audio, fs = librosa.load(path=path, sr=vae.sr, mono=False)  # (2, l)

                for i in range(aug_repeats):
                
                    jitter = int((i / aug_repeats) * (vae.sr / vae.fps))
                    stereo = audio[:, jitter :]  # (2, l)

                    t1 = time.time()
                    latents = forward_in_chunks(vae, stereo, clip_samples)  # (d, t)
                    t = time.time() - t1

                    out_path = Path(out_dir, split, name, f"{stem}_{i:03d}_vae.h5")
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with h5py.File(out_path, 'w') as hf:
                        hf.create_dataset("latent", data=latents, dtype=np.float32)
                        hf.attrs.create("fps", data=vae.fps, dtype=float)

                    print(f"Write out to {out_path} time: {t:.2f} s")

    
def compute_mono_stereo_vae(args) -> None:

    root = args.dataset_root
    out_dir = args.out_dir
    aug_repeats = args.augmentation_repeats

    device = "cuda"
    clip_duration = 60.

    # Load VAE
    vae = LevoVAE().to(device)
    
    clip_samples = int(clip_duration * vae.sr)

    # Compuate VAE latent
    stems = ["mixture"]

    for split in ["train", "test"]:

        audio_names = sorted(os.listdir(Path(root, split)))

        for i, name in enumerate(audio_names):

            print("{}/{}, {}".format(i, len(audio_names), name))

            for stem in stems:

                path = Path(root, split, name, f"{stem}.wav")
                audio, fs = librosa.load(path=path, sr=vae.sr, mono=False)

                for i in range(aug_repeats):
                
                    jitter = int((i / aug_repeats) * (vae.sr / vae.fps))
                    stereo = audio[:, jitter :]  # (2, l)
                    mono = stereo.mean(axis=0, keepdims=True).repeat(repeats=2, axis=0)

                    t1 = time.time()
                    stereo_latents = forward_in_chunks(vae, stereo, clip_samples)
                    mono_latents = forward_in_chunks(vae, mono, clip_samples)
                    t = time.time() - t1

                    sub_dir = Path(out_dir, split, name)
                    sub_dir.mkdir(parents=True, exist_ok=True)

                    stereo_out_path = Path(sub_dir, f"{stem}_{i:03d}_stereo_vae.h5")
                    mono_out_path = Path(sub_dir, f"{stem}_{i:03d}_mono_vae.h5")
                    
                    with h5py.File(stereo_out_path, 'w') as hf:
                        hf.create_dataset("latent", data=stereo_latents, dtype=np.float32)
                        hf.attrs.create("fps", data=vae.fps, dtype=float)

                    with h5py.File(mono_out_path, 'w') as hf:
                        hf.create_dataset("latent", data=mono_latents, dtype=np.float32)
                        hf.attrs.create("fps", data=vae.fps, dtype=float)
                    
                    print(f"Write out to {stereo_out_path}")
                    print(f"Write out to {mono_out_path}")
                    print(f"time: {t:.2f} s")


def compute_dac_stereo_vae(args) -> None:

    import dac

    root = args.dataset_root
    out_dir = args.out_dir
    aug_repeats = args.augmentation_repeats

    device = "cuda"
    clip_duration = 60.
    n_quantizers = 2

    # Load VAE
    vae = LevoVAE().to(device)
    dac = DAC(n_quantizers=n_quantizers).to(device)
    
    clip_samples = int(clip_duration * vae.sr)

    # Compuate VAE latent
    stems = ["mixture"]
    
    for split in ["train", "test"]:

        audio_names = sorted(os.listdir(Path(root, split)))

        for i, name in enumerate(audio_names):

            print("{}/{}, {}".format(i, len(audio_names), name))

            for stem in stems:

                path = Path(root, split, name, f"{stem}.wav")
                audio, fs = librosa.load(path=path, sr=vae.sr, mono=False)
                
                for i in range(aug_repeats):
                
                    jitter = int((i / aug_repeats) * (vae.sr / vae.fps))
                    stereo = audio[:, jitter :]  # (2, l)

                    dac_audio = librosa.resample(y=stereo, orig_sr=vae.sr, target_sr=dac.sr)
                    dac_audio = dac_audio.mean(axis=0, keepdims=True)

                    t1 = time.time()
                    vae_latents = forward_in_chunks(vae, stereo, clip_samples)
                    dac_codes = forward_in_chunks(dac, dac_audio, clip_samples)
                    t = time.time() - t1
                    
                    sub_dir = Path(out_dir, split, name)
                    sub_dir.mkdir(parents=True, exist_ok=True)

                    vae_out_path = Path(sub_dir, f"{stem}_{i:03d}_vae.h5")
                    dac_out_path = Path(sub_dir, f"{stem}_{i:03d}_dac.h5")

                    with h5py.File(vae_out_path, 'w') as hf:
                        hf.create_dataset("latent", data=vae_latents, dtype=np.float32)
                        hf.attrs.create("fps", data=vae.fps, dtype=float)

                    with h5py.File(dac_out_path, 'w') as hf:
                        hf.create_dataset("code", data=dac_codes, dtype=int)
                        hf.attrs.create("fps", data=dac.fps, dtype=float)
                    
                    print(f"Write out to {vae_out_path}")
                    print(f"Write out to {dac_out_path}")
                    print(f"time: {t:.2f} s")


def compute_8khz_44khz_vae(args) -> None:

    root = args.dataset_root
    out_dir = args.out_dir
    aug_repeats = args.augmentation_repeats

    device = "cuda"
    clip_duration = 60.
    lowres_sr = 8000.

    # Load VAE
    vae = LevoVAE().to(device)
    
    clip_samples = int(clip_duration * vae.sr)

    # Compuate VAE latent
    stems = ["mixture"]

    for split in ["train", "test"]:

        audio_names = sorted(os.listdir(Path(root, split)))

        for i, name in enumerate(audio_names):

            print("{}/{}, {}".format(i, len(audio_names), name))

            for stem in stems:

                path = Path(root, split, name, f"{stem}.wav")
                audio, fs = librosa.load(path=path, sr=vae.sr, mono=False)

                for i in range(aug_repeats):
                
                    jitter = int((i / aug_repeats) * (vae.sr / vae.fps))
                    stereo = audio[:, jitter :]  # (2, l)
                    
                    lowres_audio = librosa.resample(y=stereo, orig_sr=vae.sr, target_sr=lowres_sr)
                    lowres_audio = librosa.resample(y=lowres_audio, orig_sr=lowres_sr, target_sr=vae.sr)

                    t1 = time.time()
                    stereo_latents = forward_in_chunks(vae, stereo, clip_samples)
                    lowres_latents = forward_in_chunks(vae, lowres_audio, clip_samples)
                    t = time.time() - t1

                    sub_dir = Path(out_dir, split, name)
                    sub_dir.mkdir(parents=True, exist_ok=True)

                    highres_out_path = Path(sub_dir, f"{stem}_{i:03d}_highres_vae.h5")
                    lowres_out_path = Path(sub_dir, f"{stem}_{i:03d}_lowres_vae.h5")
                    
                    with h5py.File(highres_out_path, 'w') as hf:
                        hf.create_dataset("latent", data=stereo_latents, dtype=np.float32)
                        hf.attrs.create("fps", data=vae.fps, dtype=float)

                    with h5py.File(lowres_out_path, 'w') as hf:
                        hf.create_dataset("latent", data=lowres_latents, dtype=np.float32)
                        hf.attrs.create("fps", data=vae.fps, dtype=float)
                    
                    print(f"Write out to {highres_out_path}")
                    print(f"Write out to {lowres_out_path}")
                    print(f"time: {t:.2f} s")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    parser_stems = subparsers.add_parser("stems")
    parser_stems.add_argument("--dataset_root", type=str, required=True, help="Path of config yaml.")
    parser_stems.add_argument("--out_dir", type=str)
    parser_stems.add_argument("--augmentation_repeats", type=int)

    parser_mono_stereo = subparsers.add_parser("mono_stereo")
    parser_mono_stereo.add_argument("--dataset_root", type=str, required=True, help="Path of config yaml.")
    parser_mono_stereo.add_argument("--out_dir", type=str)
    parser_mono_stereo.add_argument("--augmentation_repeats", type=int)

    parser_dac_stereo = subparsers.add_parser("dac_stereo")
    parser_dac_stereo.add_argument("--dataset_root", type=str, required=True, help="Path of config yaml.")
    parser_dac_stereo.add_argument("--out_dir", type=str)
    parser_dac_stereo.add_argument("--augmentation_repeats", type=int)

    parser_dac_stereo = subparsers.add_parser("8khz_44khz")
    parser_dac_stereo.add_argument("--dataset_root", type=str, required=True, help="Path of config yaml.")
    parser_dac_stereo.add_argument("--out_dir", type=str)
    parser_dac_stereo.add_argument("--augmentation_repeats", type=int)

    args = parser.parse_args()
    
    if args.mode == "stems":
        compute_stems_vae(args)

    elif args.mode == "mono_stereo":
        compute_mono_stereo_vae(args)

    elif args.mode == "dac_stereo":
        compute_dac_stereo_vae(args)

    elif args.mode == "8khz_44khz":
        compute_8khz_44khz_vae(args)

    else:
        raise ValueError