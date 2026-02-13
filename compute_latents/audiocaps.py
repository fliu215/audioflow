import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from audio_flow.encoders.audio.levo_vae import LevoVAE
from audio_flow.utils import compute_and_save_latents


def compute_vae(args) -> None:

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

    csv_path = Path(root, f"{split}.csv")
    meta_dict = load_meta(csv_path)
    n_audios = len(meta_dict["name"])

    for n, name in enumerate(meta_dict["name"]):
        print(f"{n}/{n_audios}")

        path = Path(root, "audiocaps_raw_audio", name)    
        audio, fs = librosa.load(path=path, sr=vae.sr, mono=False)  # (l,)

        if audio.ndim == 1:
            audio = np.repeat(audio[None, :], repeats=2, axis=0)  # (2, l)
        if audio.ndim == 2 and audio.shape[0] != 2:
            audio = np.repeat(np.mean(audio, axis=0, keepdims=True), repeats=2, axis=0)  # (2, l)

        chunk_samples = int(chunk_duration * vae.sr)
        base_path = Path(out_dir, path.stem)
        
        compute_and_save_latents(
            audio=audio, 
            aug_repeats=aug_repeats, 
            chunk_samples=chunk_samples, 
            vae=vae, 
            latent_type=latent_type, 
            base_path=base_path
        )
    

def load_meta(meta_csv: str) -> dict:
    df = pd.read_csv(meta_csv, sep=',')
    meta_dict = {"name": [], "caption": []}

    for n in range(len(df)):
        try:
            name = "{}_{}.wav".format(df["youtube_id"][n], round(df["start_time"][n]))
            meta_dict["name"].append(name)
            meta_dict["caption"].append(df["caption"][n])
        except:
            pass

    return meta_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--latent_type", type=str, default="levo_vae")
    parser.add_argument("--augmentation_repeats", type=int, default=1)
    parser.add_argument("--chunk_duration", type=float, default=60.)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    compute_vae(args)