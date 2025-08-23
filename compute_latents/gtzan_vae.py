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
    clip_duration = 30.

    # Load VAE
    vae, vae_config = load_levo_vae()
    vae = vae.to(device)

    fps = vae_config["fps"]
    sr = vae_config["sample_rate"]
    clip_samples = int(clip_duration * sr)

    # Compuate VAE latent
    labels = sorted(os.listdir(Path(root, "genres")))

    for k, label in enumerate(labels):
        
        paths = sorted(list(Path(root, "genres", label).glob("*.au")))
        
        for path in paths:
            
            print(k, path)
            audio, fs = librosa.load(path=path, sr=sr, mono=False)
            audio = np.repeat(audio[None, :], repeats=2, axis=0)  # (2, l)

            for i in range(aug_repeats):
                
                jitter = int((i / aug_repeats) * (sr / fps))
                aug_audio = audio[:, jitter :]  # (2, l)
                aug_audio = librosa.util.fix_length(data=aug_audio, size=clip_samples, axis=-1)
                
                t1 = time.time()
                latents = forward_vae(vae, aug_audio, clip_samples=clip_samples, sr=sr)
                tm = time.time() - t1

                out_path = Path(out_dir, "genres", label, "{}_{:03d}.h5".format(Path(path).stem, i))
                out_path.parent.mkdir(parents=True, exist_ok=True)
                
                with h5py.File(out_path, 'w') as hf:
                    hf.create_dataset("latent", data=latents, dtype=np.float32)
                    hf.attrs.create("fps", data=fps, dtype=int)

                print(f"Write out to {out_path} time: {tm:.2f} s")

        
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True, help="Path of config yaml.")
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--augmentation_repeats", type=int)
    args = parser.parse_args()

    compute_vae(args)