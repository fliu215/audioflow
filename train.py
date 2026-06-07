from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.utils.data._utils.collate import default_collate
from functools import partial
import matplotlib.pyplot as plt
import soundfile
from copy import deepcopy
import wandb

from audioflow.utils.yaml import read_yaml
from audioflow.utils.json import read_jsonl
from audioflow.encoders.audio import load_encoder
from audioflow.utils.torch import requires_grad, trim_target_latent, to_device, mean, save, load
from audioflow.utils.ema import update_ema
from audioflow.utils.optim import get_optimizer_and_scheduler
from audioflow.utils.misc import get_single_value, logmel
from audioflow.flows.factory import get_flow
from audioflow.utils.cfg import cfg_drop, cfg_forward
from audioflow.solvers.euler import euler_solver
from audioflow.samplers.factory import get_batch_sampler
from audioflow.datasets.factory import get_dataset
from audioflow.models.factory import get_in, get_out, get_base
from audioflow.adapters.factory import get_adapter


def train(args) -> None:
    r"""Train audio generation with flow matching."""

    # Arguments
    wandb_log = not args.no_log
    config_path = Path(args.config)
    filename = Path(__file__).stem
    
    # Configs
    configs = read_yaml(config_path)
    device = configs["train"]["device"]
    ckpt_path = configs["train"]["resume_ckpt_path"]

    # Checkpoints directory
    config_name = config_path.stem
    ckpts_dir = Path("./checkpoints") / filename / config_name
    ckpts_dir.mkdir(parents=True, exist_ok=True)

    # Sampler
    batch_sampler = get_batch_sampler(configs)

    # Dataset
    train_dataset = get_dataset(configs)

    # Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_sampler=batch_sampler,
        num_workers=configs["train"]["num_workers"], 
        pin_memory=True,
    )

    # Model
    model = get_model(configs["model"], ckpt_path).to(device)

    # EMA (optional)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    ema.eval()  # EMA model should always be in eval mode

    # Optimizer
    optimizer, scheduler = get_optimizer_and_scheduler(
        configs=configs, 
        params=model.parameters()
    )

    # Flow matcher
    fm = get_flow(configs["flow"])
    
    # Logger
    if wandb_log:
        wandb.init(project="audio_flow", name=f"{filename}_{config_name}")

    validator = Validator(configs, ema, device)
    
    for step, data in enumerate(tqdm(train_dataloader)):

        # ------ 1. Data preparation ------
        # 1.1 CFG drop
        data = cfg_drop(
            data=data, 
            p_full=configs["cfg"]["train"]["p_full"], 
            p_partial=configs["cfg"]["train"]["p_partial"]
        )

        # 1.2 Trim data
        data = trim_target_latent(data)
        data = to_device(data, device)

        # 1.3 Sample noise
        x_real = data["target_latent"]
        noise = torch.randn_like(x_real)  # (b, ...)

        # 1.4 Sample t. Compute input, and velocity
        t, x, u = fm.sample(x0=noise, x1=x_real)  # t: (b,), x: (b, ...), u: (b, ...)

        # ------ 2. Training ------
        # 2.1 Forward
        model.train()
        v = model(t, x, data)  # (b, ...)

        # 2.2 Loss
        loss = mean((v - u) ** 2, mask=data["target_mask"])

        # 2.3 Optimize
        optimizer.zero_grad()  # Reset all parameter.grad to 0
        loss.backward()  # Update all parameter.grad
        optimizer.step()  # Update all parameters based on all parameter.grad
        update_ema(ema, model, decay=0.999)

        # 2.4 Learning rate scheduler
        if scheduler:
            scheduler.step()

        if step % 100 == 0:
            print("train loss: {:.4f}".format(loss.item()))
        
        # ------ 3. Evaluation ------
        # 3.1 Evaluate
        if step % configs["train"]["test_every_n_steps"] == 0:

            # for split in ["train", "test"]:
            for split in ["test"]:
                validator(
                    split=split,
                    out_dir=Path("./results", filename, config_name, f"steps={step}_ema"),
                )

            if wandb_log:
                wandb.log(
                    data={
                        "train_loss": loss.item()
                    },
                    step=step
                )
        
        # 3.2 Save model
        if step % configs["train"]["save_every_n_steps"] == 0:
            ckpt_path = Path(ckpts_dir, f"step={step}_ema.pth")
            save(ema, ckpt_path)
            print(f"Save model to {ckpt_path}")
        
        if step == configs["train"]["training_steps"]:
            break
        
        step += 1


def get_model(configs: dict, ckpt_path: str) -> nn.Module:

    in_ = get_in(configs["in"])
    out = get_out(configs["out"])
    base = get_base(configs["base"])
    adapter = get_adapter(configs["adapter"])

    model = AudioFlow(in_, base, out, adapter)

    if ckpt_path:
        model = load(model, ckpt_path)
        print(f"Load checkpoint from {ckpt_path}")

    return model


class AudioFlow(nn.Module):
    def __init__(self, in_: nn.Module, base: nn.Module, out: nn.Module, adapter: nn.Module) -> None:
        super().__init__()
        self.in_ = in_
        self.out = out
        self.base = base
        self.adapter = adapter

    def forward(self, t, x, data):
        controls = self.adapter(data)
        x = self.in_(x)
        x = self.base(t, x, controls)
        x = self.out(x)
        return x


class Validator:
    def __init__(self, configs: dict, model: nn.Module, device) -> nn.Module:
        
        self.configs = configs
        self.model = model
        self.device = device

        if "validate" in configs:
            self.vae = load_encoder(configs["validate"]["vae"]["name"]).to(self.device)
        else:
            self.vae = None

        self.dataset = get_dataset(self.configs)

    def __call__(self, split, out_dir):

        jsonl_path = self.configs[f"validate_{split}_jsonl"]["path"]
        n_valid = self.configs[f"validate_{split}_jsonl"]["num"]
        metas = read_jsonl(jsonl_path)

        skip_n = max(1, len(metas) // n_valid)
        metas = metas[0 :: skip_n]

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
                        cfg_forward, 
                        model=self.model,
                        data_c=data_c,
                        data_u=data_u,
                        cfg_scale=self.configs["cfg"]["sample"]["scale"]
                    )  # Usage: fn(t, x)
                else:
                    fn = partial(
                        self.model, 
                        data=data_c
                    )  # Usage: fn(t, x)
            
                x_gen = euler_solver(fn, noise, n_steps=100)  # (b, l, d)
                

            # data["gen"] = x_gen
            # out_path = Path(out_dir, stem + ".png")
            # write_to_hdf5(data, )

            # from IPython import embed; embed(using=False); os._exit(0)

            if self.vae:
                # Decode audio from VAE latents
                audio_gen = self.vae.decode(x_gen).data.cpu().numpy()[0]  # (c, l)
                audio_gt = self.vae.decode(x_real).data.cpu().numpy()[0]  # (c, l)
                
                # task = data["task"][0]
                # if task in ["music source separation", "vocals to music", 
                #     "mono to stereo", "super-resolution", "codec to music"]:
                #     x_in = data["input_latent"].to(device)
                #     audio_in = vae.decode(x_in).data.cpu().numpy()[0]  # (c, l)
                # else:
                #     audio_in = None
                # audio_in = self.vae.decode(x_in).data.cpu().numpy()[0]  # (c, l)
                audio_in = None

                # ------ 3. Plot and Visualization ------
                # 3.1 Plot mel spectrogram
                if audio_in is not None:
                    logmel_in = logmel(audio_in, self.vae.sr)
                logmel_gen = logmel(audio_gen, self.vae.sr)
                logmel_gt = logmel(audio_gt, self.vae.sr)

                fig, axs = plt.subplots(3, 1, figsize=(10, 10))
                vmin, vmax = -10, 5

                if audio_in is not None:
                    axs[0].matshow(logmel_in.T, origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)

                # elif task in ["midi to audio"]:
                #     x_in = data["input_latent"].cpu().numpy()[0]
                #     axs[0].matshow(x_in.T, origin='lower', aspect='auto', cmap='jet', vmin=0., vmax=1.)

                axs[1].matshow(logmel_gen.T, origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
                axs[2].matshow(logmel_gt.T, origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
                axs[0].set_title("Input")
                axs[1].set_title("Generation")
                axs[2].set_title("Ground truth")
                axs[2].xaxis.tick_bottom()

                strs = [split, f"idx={i}"]
                for key in ["task", "prompt"]:
                    if key in data.keys():
                        # text = get_single_value(data[key])[0 : 150]
                        text = data[key][0][0 : 150]
                        # strs.append("{}={}".format(key, text))
                        # strs.append("{}={}".format(key, metas[i]["input"]["text"]["music"]["caption"]))
                        strs.append("{}={}".format(key, metas[i]["input"]["text"]["audio"]["caption"]))
                stem = ",".join(strs)
                
                # if task in ["video to audio"]:
                #     stem += ",id={}".format(Path(data["input_latent_path"][0]).stem)

                Path(out_dir).mkdir(parents=True, exist_ok=True)
                out_path = Path(out_dir, stem + ".png")
                plt.savefig(out_path)
                print(f"Write out to {out_path}")

                # 3.2 Save audio
                if audio_in is not None:
                    out_path = Path(out_dir, stem + ",input.wav")
                    soundfile.write(file=out_path, data=audio_in.T, samplerate=self.vae.sr)
                    print(f"Write out to {out_path}") 

                out_path = Path(out_dir, stem + ",gen.wav")
                soundfile.write(file=out_path, data=audio_gen.T, samplerate=self.vae.sr)
                print(f"Write out to {out_path}")

                out_path = Path(out_dir, stem + ",gt.wav")
                soundfile.write(file=out_path, data=audio_gt.T, samplerate=self.vae.sr)
                print(f"Write out to {out_path}")
                # from IPython import embed; embed(using=False); os._exit(0)


def write_to_hdf5(data, path):
    with h5py.File(path, 'w') as hf:
        for k, v in data.items():
            if isinstance(v[0], Tensor):
                dtype = v[0].cpu().numpy()
            else:
                dtype = type(v[0])
            hf.create_dataset(k, v[0], dtype=dtype)


'''
def validate(
    configs: dict,
    model: nn.Module,
    vae: nn.Module,
    split: Literal["train", "test"],
    out_dir: str
) -> float:
    r"""Validate the model on part of data."""

    device = next(model.parameters()).device
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = configs[f"validate_{split}_jsonl"]["path"]
    n_valid = configs[f"validate_{split}_jsonl"]["num"]
    metas = load_jsonl(jsonl_path)

    skip_n = max(1, len(metas) // n_valid)
    metas = metas[0 :: skip_n]

    # Dataset
    dataset = get_dataset(configs)
    
    for i in range(len(metas)):

        # ------ 1. Data preparation ------
        # 1.1 Get Data
        data = dataset[metas[i]]
        data = default_collate([data, data])  # (2,)

        # Drop controls for cfg
        data = cfg_drop(data, model.adapter.null_c, mask=Tensor([0, 1]), device=device)  # (2,)

        data = truncate_latent(data)  # (2,)
        data = to_device(data, device)  # (2,)

        with torch.no_grad():
            model.eval()
            controls = model.adapter(data)  # (2,)

        # 2.1 Sample noise
        x_real = data["target_latent"][0 : 1]  # (1, t, d)
        noise = torch.randn_like(x_real)  # (1, l, d)

        # ------ 2. Forward with ODE ------
        # 2.1 Iteratively forward
        with torch.no_grad():
            model.eval()
            x_gen = euler_solver_cfg(model.base, noise, controls, n_steps=100)  # (b, l, d)

        # Decode audio from VAE latents
        audio_gen = vae.decode(x_gen).data.cpu().numpy()[0]  # (c, l)
        audio_gt = vae.decode(x_real).data.cpu().numpy()[0]  # (c, l)
        
        task = data["task"][0]
        if task in ["music source separation", "vocals to music", 
            "mono to stereo", "super-resolution", "codec to music"]:
            x_in = data["input_latent"].to(device)
            audio_in = vae.decode(x_in).data.cpu().numpy()[0]  # (c, l)
        else:
            audio_in = None

        # ------ 3. Plot and Visualization ------
        # 3.1 Plot mel spectrogram
        if audio_in is not None:
            logmel_in = logmel(audio_in, vae.sr)
        logmel_gen = logmel(audio_gen, vae.sr)
        logmel_gt = logmel(audio_gt, vae.sr)

        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        vmin, vmax = -10, 5

        if audio_in is not None:
            axs[0].matshow(logmel_in.T, origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)

        elif task in ["midi to audio"]:
            x_in = data["input_latent"].cpu().numpy()[0]
            axs[0].matshow(x_in.T, origin='lower', aspect='auto', cmap='jet', vmin=0., vmax=1.)

        axs[1].matshow(logmel_gen.T, origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
        axs[2].matshow(logmel_gt.T, origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
        axs[0].set_title("Input")
        axs[1].set_title("Generation")
        axs[2].set_title("Ground truth")
        axs[2].xaxis.tick_bottom()

        strs = [split, f"idx={i}"]
        for key in ["task", "prompt"]:
            if key in data.keys():
                # text = get_single_value(data[key])[0 : 150]
                text = data[key][0][0 : 150]
                strs.append("{}={}".format(key, text))
        stem = ",".join(strs)
        
        if task in ["video to audio"]:
            stem += ",id={}".format(Path(data["input_latent_path"][0]).stem)

        out_path = Path(out_dir, stem + ".png")
        plt.savefig(out_path)
        print(f"Write out to {out_path}")

        # 3.2 Save audio
        if audio_in is not None:
            out_path = Path(out_dir, stem + ",input.wav")
            soundfile.write(file=out_path, data=audio_in.T, samplerate=vae.sr)
            print(f"Write out to {out_path}") 

        out_path = Path(out_dir, stem + ",gen.wav")
        soundfile.write(file=out_path, data=audio_gen.T, samplerate=vae.sr)
        print(f"Write out to {out_path}")

        out_path = Path(out_dir, stem + ",gt.wav")
        soundfile.write(file=out_path, data=audio_gt.T, samplerate=vae.sr)
        print(f"Write out to {out_path}")


# @torch.no_grad()
# def cfg_pred(model, t, x, controls, cfg_scale=4.0):
#     x = torch.cat([x, x], dim=0)
#     u, c = model(t=t, x=x, controls=controls).chunk(2, dim=0)
#     pred = u + cfg_scale * (c - u)
#     return pred


@torch.no_grad()
def cfg_pred(model, t, x, controls, cfg_scale=4.0):
    # from IPython import embed; embed(using=False); os._exit(0)
    x = torch.cat([x, x], dim=0)
    c, u = model(t=t, x=x, controls=controls).chunk(2, dim=0)
    pred = u + cfg_scale * (c - u)
    return pred


def euler_solver_cfg(
    model: nn.Module, 
    noise: Tensor, 
    controls: dict, 
    n_steps: int,
) -> Tensor:

    t = torch.linspace(0, 1, n_steps, device=noise.device)
    x = noise
    
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        dx = cfg_pred(model, t[i], x, controls)   # f(t, x)
        x = x + dt * dx              # Euler update

    return x
'''

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path of config yaml.")
    parser.add_argument("--no_log", action="store_true", default=False)
    args = parser.parse_args()

    train(args)