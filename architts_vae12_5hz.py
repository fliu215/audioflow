from __future__ import annotations

import argparse
from pathlib import Path

import torch.nn as nn
import torch
import torch.nn.functional as F
import torchaudio
from torch.package import PackageImporter


DEFAULT_REPO_ID = "archimickey/architts-vae12_5hz"
CKPT_FILENAME = "architts_vae12_5hz.pt"


class ArchiTTSVAE12Hz(nn.Module):
    """Small loader around the packaged ArchiTTS 24 kHz VAE.

    The checkpoint is a torch.package archive containing the VAE source files,
    the resolved model config, and the state_dict. Latents exposed by this
    wrapper use shape (B, T, D), where fps = 12.5 and D = 64.
    """

    def __init__(self, ckpt_path: str | Path | None = None, device: str | torch.device | None = None):
        super().__init__()
        self.ckpt_path = resolve_ckpt_path(ckpt_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        importer = PackageImporter(str(self.ckpt_path))
        package = importer.load_pickle("architts_vae", "checkpoint.pkl")
        autoencoder_module = importer.import_module("architts.model.vae.autoencoder")
        pretransform_module = importer.import_module("architts.model.vae.pretransform")

        self.metadata = package["metadata"]
        self.model = pretransform_module.AutoencoderPretransform(
            autoencoder_module.create_autoencoder_from_config(package["config"]),
            scale=float(self.metadata["scale"]),
        )
        self.model.load_state_dict(package["state_dict"])
        self.model.requires_grad_(False).eval().to(self.device)

        self.sample_rate = int(self.metadata["sample_rate"])
        self.downsampling_ratio = int(self.metadata["downsampling_ratio"])
        self.fps = float(self.metadata["fps"])
        self.dim = int(self.metadata["encoded_channels"])

    @torch.inference_mode()
    def encode(self, audio: torch.Tensor, sample_posterior: bool = False) -> torch.Tensor:
        """Encode audio shaped (B, C, samples) to latents shaped (B, T, 64)."""
        if audio.ndim != 3:
            raise ValueError(f"Expected audio shape (B, C, samples), got {tuple(audio.shape)}")
        audio = audio.to(self.device)
        if audio.shape[1] > 1:
            audio = audio.mean(dim=1, keepdim=True)
        remainder = audio.shape[-1] % self.downsampling_ratio
        if remainder:
            audio = F.pad(audio, (0, self.downsampling_ratio - remainder))

        if sample_posterior:
            latent = self.model.encode(audio)
        else:
            latent = self._encode_posterior_mean(audio)
        return latent.transpose(1, 2).contiguous()

    @torch.inference_mode()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latents shaped (B, T, 64) or (T, 64) to audio shaped (B, 1, samples)."""
        if latent.ndim == 2:
            latent = latent.unsqueeze(0)
        if latent.ndim != 3:
            raise ValueError(f"Expected latent shape (B, T, 64) or (T, 64), got {tuple(latent.shape)}")
        latent = latent.to(self.device).transpose(1, 2).contiguous()
        return self.model.decode(latent)

    def _encode_posterior_mean(self, audio: torch.Tensor) -> torch.Tensor:
        autoencoder = self.model.model
        latent = autoencoder.encoder(audio) if autoencoder.encoder is not None else audio
        bottleneck = autoencoder.bottleneck
        if bottleneck is not None:
            bottleneck_name = type(bottleneck).__name__
            if bottleneck_name == "VAEBottleneck2":
                latent = latent[:, :-1, :]
            elif bottleneck_name == "VAEBottleneck":
                latent = latent.chunk(2, dim=1)[0]
            else:
                latent = bottleneck.encode(latent)
        return latent / self.model.scale

    def load_audio(self, audio_path: str | Path) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(str(audio_path))
        waveform = waveform.float()
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        return waveform.unsqueeze(0)

    def save_audio(self, audio: torch.Tensor, audio_path: str | Path) -> None:
        audio_path = Path(audio_path)
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio = audio.detach().cpu()
        if audio.ndim == 3:
            audio = audio[0]
        torchaudio.save(str(audio_path), audio.clamp(-1, 1), self.sample_rate)


def resolve_ckpt_path(ckpt_path: str | Path | None = None, repo_id: str = DEFAULT_REPO_ID) -> Path:
    if ckpt_path is not None:
        path = Path(ckpt_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    local_path = Path(__file__).resolve().with_name(CKPT_FILENAME)
    if local_path.exists():
        return local_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise FileNotFoundError(
            f"Could not find {CKPT_FILENAME} next to this script, and huggingface_hub is not installed."
        ) from exc
    return Path(hf_hub_download(repo_id=repo_id, filename=CKPT_FILENAME))


def load_latent(path: str | Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("latent", "latents", "z"):
            if key in obj:
                return torch.as_tensor(obj[key], dtype=torch.float32)
        raise KeyError(f"No latent tensor found in {path}; available keys: {sorted(obj.keys())}")
    return torch.as_tensor(obj, dtype=torch.float32)


def save_latent(path: str | Path, latent: torch.Tensor, vae: ArchiTTSVAE12Hz) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latent": latent.detach().cpu(),
            "sample_rate": vae.sample_rate,
            "fps": vae.fps,
            "dim": vae.dim,
            "downsampling_ratio": vae.downsampling_ratio,
        },
        path,
    )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode/decode audio with the packaged ArchiTTS 12.5 Hz VAE.")
    parser.add_argument("--ckpt", help=f"Path to {CKPT_FILENAME}. Defaults to local file or Hugging Face download.")
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID)
    parser.add_argument("--device", default=None)

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("info")

    encode_parser = subparsers.add_parser("encode")
    encode_parser.add_argument("audio_path")
    encode_parser.add_argument("latent_path")
    encode_parser.add_argument("--sample_posterior", action="store_true")

    decode_parser = subparsers.add_parser("decode")
    decode_parser.add_argument("latent_path")
    decode_parser.add_argument("audio_path")

    recon_parser = subparsers.add_parser("reconstruct")
    recon_parser.add_argument("audio_path")
    recon_parser.add_argument("out_path")
    recon_parser.add_argument("--sample_posterior", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = get_args()
    ckpt = resolve_ckpt_path(args.ckpt, args.repo_id)
    vae = ArchiTTSVAE12Hz(ckpt, device=args.device)

    if args.command == "info":
        print(f"checkpoint: {ckpt}")
        print(f"sample_rate: {vae.sample_rate}")
        print(f"fps: {vae.fps}")
        print(f"latent_dim: {vae.dim}")
        print(f"downsampling_ratio: {vae.downsampling_ratio}")
    elif args.command == "encode":
        audio = vae.load_audio(args.audio_path)
        latent = vae.encode(audio, sample_posterior=args.sample_posterior)
        save_latent(args.latent_path, latent[0], vae)
        print(f"wrote latent {tuple(latent[0].shape)} to {args.latent_path}")
    elif args.command == "decode":
        latent = load_latent(args.latent_path)
        audio = vae.decode(latent)
        vae.save_audio(audio, args.audio_path)
        print(f"wrote audio {tuple(audio.shape)} to {args.audio_path}")
    elif args.command == "reconstruct":
        audio = vae.load_audio(args.audio_path)
        latent = vae.encode(audio, sample_posterior=args.sample_posterior)
        recon = vae.decode(latent)
        vae.save_audio(recon, args.out_path)
        print(f"wrote reconstruction {tuple(recon.shape)} to {args.out_path}")


if __name__ == "__main__":
    main()
