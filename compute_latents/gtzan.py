import argparse
import os
import random
import time
from pathlib import Path

import h5py
import librosa
import numpy as np

from audio_flow.utils import forward_in_chunks
from audio_flow.vae.levo import LevoVAE


def compute_vae(args) -> None:

    root = args.dataset_root
    out_dir = args.out_dir
    aug_repeats = args.augmentation_repeats

    device = "cuda"
    clip_duration = 30.

    # Load VAE
    vae = LevoVAE().to(device)

    clip_samples = int(clip_duration * vae.sr)

    # Compuate VAE latent
    labels = sorted(os.listdir(Path(root, "genres")))

    for k, label in enumerate(labels):
        
        print("{}/{}".format(k, len(labels)))

        paths = sorted(list(Path(root, "genres", label).glob("*.au")))
        
        for path in paths:
            
            audio, fs = librosa.load(path=path, sr=vae.sr, mono=False)  # (l,)
            audio = np.repeat(audio[None, :], repeats=2, axis=0)  # (2, l)

            for i in range(aug_repeats):
                
                jitter = int((i / aug_repeats) * (vae.sr / vae.fps))
                aug_audio = audio[:, jitter :]  # (2, l)
                aug_audio = librosa.util.fix_length(data=aug_audio, size=clip_samples, axis=-1)
                
                t1 = time.time()
                latents = forward_in_chunks(vae, aug_audio, clip_samples)  # (d, t)
                t = time.time() - t1

                out_path = Path(out_dir, "genres", label, "{}_{:03d}_vae.h5".format(Path(path).stem, i))
                out_path.parent.mkdir(parents=True, exist_ok=True)
                
                with h5py.File(out_path, 'w') as hf:
                    hf.create_dataset("latent", data=latents, dtype=np.float32)
                    hf.attrs.create("fps", data=vae.fps, dtype=float)

                print(f"Write out to {out_path} time: {t:.2f} s")

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True, help="Path of config yaml.")
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--augmentation_repeats", type=int)
    args = parser.parse_args()

    compute_vae(args)