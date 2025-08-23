import argparse
import os
import random
import time
from pathlib import Path

import h5py
import librosa
import numpy as np

from audio_flow.utils import load_levo_vae
from compute_latents.forward import forward_vae


def compute_vae(args) -> None:

    root = args.dataset_root
    out_dir = args.out_dir
    aug_repeats = args.augmentation_repeats

    device = "cuda"
    clip_duration = 60.

    # Load VAE
    vae, vae_config = load_levo_vae()
    vae = vae.to(device)

    fps = vae_config["fps"]
    sr = vae_config["sample_rate"]
    clip_samples = int(clip_duration * sr)

    # Compuate VAE latent
    stems = ["vocals", "bass", "drums", "other", "mixture"]

    for split in ["train", "test"]:

        audio_names = sorted(os.listdir(Path(root, split)))

        for i, name in enumerate(audio_names):

            print(i, name)

            for stem in stems:

                path = Path(root, split, name, f"{stem}.wav")
                audio, fs = librosa.load(path=path, sr=sr, mono=False)

                for i in range(aug_repeats):
                
                    jitter = int((i / aug_repeats) * (sr / fps))
                    aug_audio = audio[:, jitter :]  # (2, l)

                    t1 = time.time()
                    latents = forward_vae(vae, aug_audio, clip_samples=clip_samples, sr=sr)
                    tm = time.time() - t1

                    out_path = Path(out_dir, split, f"{name}_{i:03d}", f"{stem}.h5")
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with h5py.File(out_path, 'w') as hf:
                        hf.create_dataset("latent", data=latents, dtype=np.float32)
                        hf.attrs.create("fps", data=fps, dtype=int)

                    print(f"Write out to {out_path} time: {tm:.2f} s")

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    parser_stems = subparsers.add_parser("stems")
    parser_stems.add_argument("--dataset_root", type=str, required=True, help="Path of config yaml.")
    parser_stems.add_argument("--out_dir", type=str)
    parser_stems.add_argument("--augmentation_repeats", type=int)

    parser_mono = subparsers.add_parser("mono")
    parser_mono.add_argument("--dataset_root", type=str, required=True, help="Path of config yaml.")
    parser_mono.add_argument("--out_dir", type=str)
    parser_mono.add_argument("--augmentation_repeats", type=int)

    args = parser.parse_args()
    
    if args.mode == "stems":
        compute_vae(args)

    else:
        raise ValueError