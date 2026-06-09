import torch
import torch.nn as nn
from functools import partial
import soundfile
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import re

from audioflow.encoders.audio.factory import load_encoder
from audioflow.datasets.factory import get_dataset
from audioflow.solvers.euler import euler_solver
from audioflow.utils.json import read_jsonl
from torch.utils.data._utils.collate import default_collate
from audioflow.utils.torch import requires_grad, trim_target_latent, to_device, mean, save, load
from copy import deepcopy
from audioflow.guidance.cfg import cfg_drop, cfg_forward
from audioflow.utils.misc import logmel
from audioflow.solvers.factory import get_solver


class Validator:
    def __init__(self, configs: dict, model: nn.Module, device: torch.device) -> None:
        
        self.configs = configs
        self.model = model
        self.device = device

        self.vae = self._build_vae(configs)
        self.dataset = get_dataset(configs)
        self.solver = get_solver(configs["solver"])

        self.vae_name = self.configs["vae"]["name"]

    def _build_vae(self, configs: dict):
        if configs["valid"]["save_audio"] is True:
            return load_encoder(configs["vae"]["name"]).to(self.device)
        else:
            return None

    def __call__(self, split, out_dir):

        Path(out_dir).mkdir(parents=True, exist_ok=True)

        jsonl_path = self.configs["valid"][split]["path"]
        n_valid = self.configs["valid"][split]["num"]
        metas = read_jsonl(jsonl_path)
        
        indices = np.linspace(0, len(metas) - 1, n_valid, dtype=int)
        metas = [metas[i] for i in indices]

        for i in range(len(metas)):

            # ------ 1. Data preparation ------
            # 1.1 Get Data
            data = self.dataset[metas[i]]
            data = default_collate([data])
            data = trim_target_latent(data)

            data_c = deepcopy(data)
            data_u = deepcopy(data)
            data_u = cfg_drop(data_u, p_full=1.0, p_partial=0.0)

            data_c = to_device(data_c, self.device)
            data_u = to_device(data_u, self.device)

            # 2.1 Sample noise
            x_real = data_c["target_latent"]  # (1, t, d)
            noise = torch.randn_like(x_real)  # (1, l, d)

            # ------ 2. Forward with ODE ------
            # 2.1 Iteratively forward
            with torch.no_grad():
                self.model.eval()

                if self.configs["cfg"]:
                    fn = partial(
                        cfg_forward,  # fn(model, t, x, data_c, data_u, cfg_scale)
                        model=self.model,
                        data_c=data_c,
                        data_u=data_u,
                        cfg_scale=self.configs["cfg"]["sample"]["scale"]
                    )  # New function: fn(t, x)
                else:
                    fn = partial(
                        self.model,  # fn(model, t, x, data)
                        data=data_c
                    )  # New function: fn(t, x)
            
                x_gen = self.solver(fn, noise, n_steps=100)  # (b, l, d)
                
            name = f"{split},idx={i},prompt="
            name += re.sub(r"</\w+>", "", data["prompt"][0])

            if self.vae:
                # Decode audio from VAE latents
                audio_gen = self.vae.decode(x_gen).cpu().numpy()[0]  # (c, l)
                audio_gt = self.vae.decode(x_real).cpu().numpy()[0]  # (c, l)
                
                # ------ 3. Plot and Visualization ------
                logmel_gen = logmel(audio_gen, self.vae.sr)
                logmel_gt = logmel(audio_gt, self.vae.sr)

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
                self._write_hdf5(x_gen.cpu().numpy()[0], self.vae_name, out_path)
                
                out_path = out_dir / (name + ",gt.h5")
                self._write_hdf5(x_real.cpu().numpy()[0], self.vae_name, out_path)

    def _write_audio(self, audio: np.ndarray, path) -> None:
        soundfile.write(file=path, data=audio.T, samplerate=self.vae.sr)
        print(f"Write out to {path}")

    def _write_hdf5(self, data, name, path) -> None:
        with h5py.File(path, 'w') as hf:
            hf.create_dataset("data", data=data, dtype=np.float32)
            hf.attrs.create_dataset("type", data=name)
        print(f"Write out to {path}")