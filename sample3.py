from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import soundfile
import torch
from torch import Tensor
import torchdiffeq
from torch.utils.data._utils.collate import default_collate

from audio_flow.encoders.audio.levo_vae import LevoVAE
from audio_flow.utils import parse_yaml
# from train import get_model
from train3_trunc import get_model, to_device


def sample(args) -> None:
    r"""Train audio generation with flow matching."""

    # Arguments
    config_path = args.config
    ckpt_path = args.ckpt_path
    out_path = args.out_path
    duration = args.duration
    
    # Configs
    configs = parse_yaml(config_path)
    device = configs["train"]["device"]

    # Load model
    model = get_model(configs, ckpt_path).to(device)
    
    # Load VAE
    vae = LevoVAE().to(device)

    # Prepare meta data
    data = get_data(args, vae)
    # data = default_collate([data])  # create a batch
    duration = get_duration(args)

    # Noise
    length = round(duration * vae.fps)
    noise = torch.randn(1, length, vae.dim).to(device)
    
    with torch.no_grad():
        model.eval()
        data = default_collate([data])
        data = to_device(data, device)
        controls = model.adapter(data)

        select = "euler_manual"
        if select == "dopri5":
            traj = torchdiffeq.odeint(
                lambda t, x: model.base(t, x, controls),
                y0=noise,
                t=torch.linspace(0, 1, 2, device=device),
                atol=1e-4,
                rtol=1e-4,
                method="dopri5",
            )
            x_gen = traj[-1]  # (b, t, d)
        elif select == "euler":
            n_steps = 100
            traj = torchdiffeq.odeint(
                lambda t, x: model.base(t, x, controls),
                y0=noise,
                t=torch.linspace(0, 1, n_steps, device=device),
                method="euler",
            )
            x_gen = traj[-1]  # (b, t, d)

        elif select == "euler_manual":
            x_gen = euler_solver(model.base, noise, controls, 100)
    
    # Decode audio from VAE latents
    audio_gen = vae.decode(x_gen.permute(0, 2, 1)).data.cpu().numpy()[0]  # (c, l)

    # Write out
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(file=out_path, data=audio_gen.T, samplerate=vae.sr)
    print(f"Write out to {out_path}")


def euler_solver(
    model: nn.Module, 
    noise: Tensor, 
    controls: dict, 
    n_steps: int
) -> Tensor:

    t = torch.linspace(0, 1, n_steps, device=noise.device)
    x = noise
    
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        dx = model(t[i], x, controls)   # f(t, x)
        x = x + dt * dx              # Euler update

    return x


def get_data(args, vae):

    task = args.task

    if task in ["text_to_speech"]:
        return {
            "task": task, 
            "content": args.prompt,
        }

    elif task in ["text_to_music", "text_to_audio"]:
        return {
            "task": task, 
            "prompt": args.prompt
        }

    elif task in ["music_source_separation", "mono_to_stereo", "super-resolution", 
        "codec_to_music"]:

        audio_path = args.audio_path
        instruction = args.instruction

        device = next(vae.parameters()).device
        audio, fs = librosa.load(path=audio_path, sr=vae.sr, mono=False)
        audio = Tensor(audio).to(device)  # (c, l)
        latent = vae.encode(audio[None, :, :])[0].T  # (t, d)

        return {
            "task": task,
            "instruction": instruction,
            "input_audio_latent": latent,
            "latent_length": latent.shape[-1]
        }

    else:
        raise ValueError(task)


def get_duration(args):
    task = args.task

    if task == "text_to_speech":
        return len(args.prompt) / 16.30

    else:
        return args.duration


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--instruction", type=str)
    parser.add_argument("--audio_path", type=str)
    parser.add_argument("--duration", type=float, default=10.)
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()

    sample(args)