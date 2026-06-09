import torch
import torch.nn as nn
from functools import partial
import soundfile
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import re

from audioflow.decoders.audio import load_decoder
from audioflow.datasets import get_dataset
from audioflow.solvers.euler import euler_solver
from audioflow.utils.json import read_jsonl
from torch.utils.data._utils.collate import default_collate
from audioflow.utils.torch import requires_grad, trim_target_latent, to_device, mean, save, load
from copy import deepcopy
from audioflow.guidance.cfg import cfg_drop, cfg_forward
from audioflow.utils.misc import logmel
from audioflow.solvers import get_solver
from audioflow.inference.generate import generate_latent


class Validator:
    def __init__(self, configs: dict, model: nn.Module, device: torch.device) -> None:
        
        self.configs = configs
        self.model = model
        self.device = device

        self.decoder_name = self.configs["vae"]["name"]
        self.dataset = get_dataset(configs["dataset"])
        self.cfg_scale = configs["cfg"]["sample"]["scale"] if "cfg" in configs else None
        self.solver = get_solver(configs["solver"])

        decode_to_audio = configs["validate"]["decode_to_audio"]
        self.decoder = load_decoder(self.decoder_name).to(device) if decode_to_audio else None

    def __call__(self, split: str, out_dir: str) -> None:

        Path(out_dir).mkdir(parents=True, exist_ok=True)

        for json_dict in self.configs["validate"][split]:
        
            jsonl_path = json_dict["path"]
            n_valid = json_dict["num"]
            
            metas = read_jsonl(jsonl_path)
            indices = np.linspace(0, len(metas) - 1, n_valid, dtype=int)
            metas = [metas[i] for i in indices]

            for i in range(len(metas)):

                # ------ 1. Data preparation ------
                # 1.1 Get Data
                data = self.dataset[metas[i]]
                data = default_collate([data])  # list to batch
                data = trim_target_latent(data)  # Cut silense
                data = to_device(data, self.device)
                
                x_real = data["target_latent"]  # (1, t, d)
                noise = torch.randn_like(x_real)  # (1, l, d)
                
                x_gen = generate_latent(self.model, noise, data, self.solver, self.cfg_scale)
                    
                name = f"{split},idx={i},prompt="
                name += re.sub(r"</\w+>", "", data["prompt"][0])

                if self.decoder:
                    # Decode audio from VAE latents
                    audio_gen = self.decoder.decode(x_gen).cpu().numpy()[0]  # (c, l)
                    audio_gt = self.decoder.decode(x_real).cpu().numpy()[0]  # (c, l)
                    
                    # ------ 3. Plot and Visualization ------
                    logmel_gen = logmel(audio_gen, self.decoder.sr)
                    logmel_gt = logmel(audio_gt, self.decoder.sr)

                    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
                    vmin, vmax = -10, 5

                    axs[1].matshow(logmel_gen.T, origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
                    axs[2].matshow(logmel_gt.T, origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
                    axs[0].set_title("Input")
                    axs[1].set_title("Generation")
                    axs[2].set_title("Ground truth")
                    axs[2].xaxis.tick_bottom()

                    out_path = out_dir / (name + ".png")
                    plt.savefig(out_path)
                    print(f"Write out to {out_path}")

                    out_path = out_dir / (name + ",gen.wav")
                    self._write_audio(audio_gen, out_path)

                    out_path = out_dir / (name + ",gt.wav")
                    self._write_audio(audio_gt, out_path)
                    
                else:
                    out_path = out_dir / (name + ",gen.h5")
                    self._write_hdf5(x_gen.cpu().numpy()[0], decoder_name, out_path)
                    
                    out_path = out_dir / (name + ",gt.h5")
                    self._write_hdf5(x_real.cpu().numpy()[0], decoder_name, out_path)

    def _write_audio(self, audio: np.ndarray, path) -> None:
        soundfile.write(file=path, data=audio.T, samplerate=self.decoder.sr)
        print(f"Write out to {path}")

    def _write_hdf5(self, data, name, path) -> None:
        with h5py.File(path, 'w') as hf:
            hf.create_dataset("data", data=data, dtype=np.float32)
            hf.attrs.create_dataset("type", data=name)
        print(f"Write out to {path}")