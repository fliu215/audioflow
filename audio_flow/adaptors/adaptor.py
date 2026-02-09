import torch.nn as nn
from einops import rearrange
from torch import Tensor

from audio_flow.encoders.audio.clap import CLAPTextEncoder
from audio_flow.encoders.text.char import CharEncoder
from audio_flow.encoders.text.t5 import T5
from audio_flow.models.aligner import Aligner
from audio_flow.utils import get_single_value


class Adaptor(nn.Module): 
    def __init__(self, dim: int, max_length: int, **kwargs):
        super().__init__()

        # Text encoder
        self.t5_encoder = T5()
        self.t5_aligner = Aligner(self.t5_encoder.dim, dim, max_length)

        # TTS encoder
        self.tts_encoder = CharEncoder()
        self.tts_embedder = nn.Embedding(self.tts_encoder.vocab_size, dim)
        self.tts_aligner = Aligner(dim, dim, max_length)
        
        # CLAP
        self.clap_encoder = CLAPTextEncoder()
        self.clap_fc = nn.Linear(self.clap_encoder.dim, dim)

        # VAE encoder
        self.audiovae_fc = nn.Linear(64, dim)

    def forward(self, data: dict) -> Tensor:
        r"""Get fixed length condition."""
        task = get_single_value(data["task"])
        length = data["target_audio_latent"].shape[-1]

        if task == "text_to_music":
            return self.get_ttm_condition(data, length)
        
        elif task == "text_to_speech":
            return self.get_tts_condition(data, length)

        elif task == "text_to_audio":
            return self.get_tta_condition(data, length)

        elif task in ["music_source_separation", "mono_to_stereo", "super-resolution", "codec_to_music"]:
            return self.get_aligned_condition(data, length)

        else:
            raise ValueError(task)

    def get_ttm_condition(self, data: dict, length: int) -> Tensor:
        r"""Get text to music condition."""
        task = self.t5_encoder(data["task"])
        task = task.mean(dim=1, keepdims=True).repeat(1, length, 1)  # (b, l, d)

        prompt = self.t5_encoder(data["prompt"])  # (b, l', d)
        prompt = self.t5_aligner(prompt, length)  # (b, l, d)

        clap = self.clap_encoder(data["prompt"])  # (b, d)
        clap = self.clap_fc(clap)[:, None, :].repeat(1, length, 1)  # (b, l, d)

        return task + prompt + clap

    def get_tts_condition(self, data: dict, length: int) -> Tensor:
        r"""Get text to speech condition."""
        task = self.t5_encoder(data["task"])
        task = task.mean(dim=1, keepdims=True).repeat(1, length, 1)  # (b, l, d)

        content = self.tts_encoder(data["content"])  # (b, l', d)
        content = self.tts_embedder(content)
        content = self.tts_aligner(content, length)  # (b, l, d)

        return task + content

    def get_tta_condition(self, data: dict, length: int) -> Tensor:
        r"""Get text to audio condition."""
        task = self.t5_encoder(data["task"])
        task = task.mean(dim=1, keepdims=True).repeat(1, length, 1)  # (b, l, d)

        prompt = self.t5_encoder(data["prompt"])  # (b, l', d)
        prompt = self.t5_aligner(prompt, length)  # (b, l, d)

        clap = self.clap_encoder(data["prompt"])  # (b, d)
        clap = self.clap_fc(clap)[:, None, :].repeat(1, length, 1)  # (b, l, d)

        return task + prompt + clap

    def get_aligned_condition(self, data: dict, length: int) -> Tensor:
        r"""Get aligned condition."""
        task = self.t5_encoder(data["task"])
        task = task.mean(dim=1, keepdims=True).repeat(1, length, 1)  # (b, l, d)

        instruction = self.t5_encoder(data["instruction"])
        instruction = instruction.mean(dim=1, keepdims=True).repeat(1, length, 1)  # (b, l, d)

        device = next(self.parameters()).device
        vae = rearrange(data["input_audio_latent"], 'b d t -> b t d').to(device)
        vae = self.audiovae_fc(vae)

        return task + instruction + vae