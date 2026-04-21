from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import soundfile
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torchdiffeq
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from tqdm import tqdm

import wandb
from audio_flow.datasets.dataset3 import MetaDataset3
from audio_flow.encoders.audio.levo_vae import LevoVAE
from audio_flow.samplers.jsonl_sampler import JsonlSampler, BatchJsonlSampler
from audio_flow.utils import (CombinedModel, LinearWarmUp, get_single_value,
                              load_jsonl, logmel, parse_yaml, requires_grad,
                              update_ema)


def train(args) -> None:
    r"""Train audio generation with flow matching."""

    # Arguments
    wandb_log = not args.no_log
    config_path = args.config
    filename = Path(__file__).stem
    
    # Configs
    configs = parse_yaml(config_path)
    device = configs["train"]["device"]
    ckpt_path = configs["train"]["resume_ckpt_path"]

    # Checkpoints directory
    config_name = Path(config_path).stem
    ckpts_dir = Path("./checkpoints", filename, config_name)
    Path(ckpts_dir).mkdir(parents=True, exist_ok=True)

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
    model = get_model(configs, ckpt_path).to(device)

    # VAE for validation
    vae = LevoVAE().to(device)

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

    # Flow matching data processor
    fm = ConditionalFlowMatcher(sigma=0.)
    
    # Logger
    if wandb_log:
        wandb.init(project="audio_flow", name=f"{filename}_{config_name}")

    for step, data in enumerate(tqdm(train_dataloader)):

        # ------ 1. Data preparation ------
        # 1.1 Data
        # data = truncate_latent(data)
        data = to_device(data, device)

        x_real = data["target_latent"]
        noise = torch.randn_like(x_real)
        
        # 1.2 Get input and velocity
        t, xt, ut = fm.sample_location_and_conditional_flow(x0=noise, x1=x_real)

        # ------ 2. Training ------
        # 2.1 Forward
        model.train()
        controls = model.adapter(data)
        vt = model.base(t=t, x=xt, controls=controls)

        # 2.2 Loss
        loss = torch.mean((vt - ut) ** 2)

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

            for split in ["train", "test"]:
                validate(
                    configs=configs,
                    model=ema,
                    vae=vae,
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
            torch.save(get_saveable_state_dict(ema), ckpt_path)
            print(f"Save model to {ckpt_path}")

        if step == configs["train"]["training_steps"]:
            break

        step += 1


# def truncate_latent(data):
#     data["target_audio_latent"] = data["target_audio_latent"][:, :, 0 : max(data["latent_length"])]
#     return data

def to_device(data: dict, device) -> dict:
    for key, value in data.items():
        if isinstance(value, Tensor):
            data[key] = value.to(device)
    return data


def get_saveable_state_dict(model):
    save_dict = {}

    for k, v in model.state_dict().items():
        keep = True

        for n, m in model.named_modules():
            if k.startswith(n) and getattr(m, "saveable", True) is False:
                keep = False
                break

        if keep:
            save_dict[k] = v

    return save_dict
     

def get_dataset(configs: dict) -> Dataset:
    r"""Get dataset."""
    name = configs["dataset"]["name"]
    
    if name == "MetaDataset":
        return MetaDataset3(configs["clip_duration"])

    else:
        raise ValueError(name)


# def get_sampler(configs: dict) -> Iterable:
#     r"""Get sampler."""
#     name = configs["sampler"]["name"]

#     if name == "JsonlSampler":
#         paths = [meta["path"] for meta in configs["train_jsonls"]]
#         weights = [meta["weight"] for meta in configs["train_jsonls"]]
#         return JsonlSampler(paths, weights)

#     else:
#         raise ValueError(name)


def get_batch_sampler(configs: dict) -> Iterable:
    r"""Get sampler."""
    name = configs["sampler"]["name"]
    batch_size = configs["train"]["batch_size_per_device"]

    if name == "BatchJsonlSampler":
        paths = [meta["path"] for meta in configs["train_jsonls"]]
        weights = [meta["weight"] for meta in configs["train_jsonls"]]
        return BatchJsonlSampler(paths, weights, batch_size)

    else:
        raise ValueError(name)


def get_model(configs: dict, ckpt_path: str) -> nn.Module:
    base = get_base(configs=configs)
    adapter = get_adapter(configs=configs)
    model = CombinedModel(base, adapter)

    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt, strict=False)
        print(f"Load checkpoint from {ckpt_path}")

    return model


def get_base(
    configs: dict, 
) -> nn.Module:
    r"""Initialize base model."""
    name = configs["base"]["name"]

    if name == "Transformer":
        from audio_flow.models.transformer import Transformer
        return Transformer(**configs["base"])

    elif name == "Transformer3":
        from audio_flow.models.transformer3 import Transformer3
        return Transformer3(**configs["base"])

    else:
        raise ValueError(name)    


def get_adapter(
    configs: dict, 
) -> nn.Module:
    r"""Initialize adapter."""
    name = configs["adapter"]["name"]
    
    if name == "Adapter":
        from audio_flow.adapters.adapter import Adapter
        return Adapter(**configs["adapter"])

    elif name == "Adapter2":
        from audio_flow.adapters.adapter2 import Adapter2
        return Adapter2(**configs["adapter"])

    if name == "AdapterFinetune":
        from audio_flow.adapters.adapter_ft import AdapterFinetune
        return AdapterFinetune(**configs["adapter"])

    elif name == "Adapter_ljspeech_02":
        from audio_flow.adapters.adapter_ljspeech_02 import Adapter_ljspeech_02
        return Adapter_ljspeech_02(**configs["adapter"])

    elif name == "Adapter_ljspeech_03":
        from audio_flow.adapters.adapter_ljspeech_03 import Adapter_ljspeech_03
        return Adapter_ljspeech_03(**configs["adapter"])

    elif name == "Adapter_ljspeech_04":
        from audio_flow.adapters.adapter_ljspeech_04 import Adapter_ljspeech_04
        return Adapter_ljspeech_04(**configs["adapter"])

    elif name == "Adapter_ljspeech_04b":
        from audio_flow.adapters.adapter_ljspeech_04b import Adapter_ljspeech_04b
        return Adapter_ljspeech_04b(**configs["adapter"])

    elif name == "Adapter_ljspeech_04b2":
        from audio_flow.adapters.adapter_ljspeech_04b2 import Adapter_ljspeech_04b2
        return Adapter_ljspeech_04b2(**configs["adapter"])

    elif name == "Adapter_ljspeech_04c":
        from audio_flow.adapters.adapter_ljspeech_04c import Adapter_ljspeech_04c
        return Adapter_ljspeech_04c(**configs["adapter"])

    elif name == "Adapter3":
        from audio_flow.adapters.adapter3 import Adapter3
        return Adapter3(**configs["adapter"])

    else:
        raise ValueError(name)    


def get_optimizer_and_scheduler(
    configs: dict, 
    params: list[torch.Tensor]
) -> tuple[optim.Optimizer, None | optim.lr_scheduler.LambdaLR]:
    r"""Get optimizer and scheduler."""

    lr = float(configs["train"]["lr"])
    warm_up_steps = configs["train"]["warm_up_steps"]
    optimizer_name = configs["train"]["optimizer"]

    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(params=params, lr=lr)

    if warm_up_steps:
        lr_lambda = LinearWarmUp(warm_up_steps)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = None

    return optimizer, scheduler


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
        data = default_collate([data])
        data = to_device(data, device)

        # 2.1 Sample noise
        x_real = data["target_latent"]
        noise = torch.randn_like(x_real)
        
        # ------ 2. Forward with ODE ------
        # 2.1 Iteratively forward
        with torch.no_grad():
            model.eval()
            controls = model.adapter(data)
            traj = torchdiffeq.odeint(
                lambda t, x: model.base(t, x, controls),
                y0=noise,
                t=torch.linspace(0, 1, 2, device=device),
                atol=1e-4,
                rtol=1e-4,
                method="dopri5",
            )

        x_gen = traj[-1]  # (b, t, d)
        
        # Decode audio from VAE latents
        audio_gen = vae.decode(x_gen.permute(0, 2, 1)).data.cpu().numpy()[0]  # (c, l)
        audio_gt = vae.decode(x_real.permute(0, 2, 1)).data.cpu().numpy()[0]  # (c, l)
        
        if "input_audio_latent" in data.keys(): 
            x_in = data["input_audio_latent"].to(device)
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
        axs[1].matshow(logmel_gen.T, origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
        axs[2].matshow(logmel_gt.T, origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
        axs[0].set_title("Input")
        axs[1].set_title("Generation")
        axs[2].set_title("Ground truth")
        axs[2].xaxis.tick_bottom()

        strs = [split, f"idx={i}"]
        for key in ["prompt", "content"]:
            if key in data.keys():
                text = get_single_value(data[key])[0 : 150]
                strs.append("{}={}".format(key, text))
        stem = ",".join(strs)
        
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path of config yaml.")
    parser.add_argument("--no_log", action="store_true", default=False)
    args = parser.parse_args()

    train(args)