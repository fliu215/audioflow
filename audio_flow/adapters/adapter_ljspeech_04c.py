import torch.nn as nn
from einops import rearrange
from torch import Tensor
import torch.nn.functional as F

from audio_flow.encoders.audio.clap import CLAPTextEncoder
from audio_flow.encoders.text.char import CharEncoder
from audio_flow.encoders.text.t5 import T5
from audio_flow.models.aligner_ljspeech_04c import Aligner_ljspeech_04c
from audio_flow.utils import get_single_value


class Adapter_ljspeech_04c(nn.Module): 
    def __init__(self, dim: int, max_length: int, **kwargs):
        super().__init__()

        # Text encoder
        self.t5_encoder = T5()
        
        # TTS encoder
        self.char_encoder = CharEncoder()
        self.char_embedder = nn.Embedding(self.char_encoder.vocab_size, dim)
        
        # CLAP
        self.clap_text_encoder = CLAPTextEncoder()
        self.clap_fc = nn.Linear(self.clap_text_encoder.dim, dim)
        self.char_aligner = Aligner_ljspeech_04c(dim, dim, max_length)

        # VAE encoder
        self.audiovae_fc = nn.Linear(64, dim)

    def forward(self, data: dict, length: int) -> Tensor:
        r"""Get fixed length condition."""
        task = get_single_value(data["task"])
        
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

        clap = self.clap_text_encoder(data["prompt"])  # (b, d)
        clap = self.clap_fc(clap)[:, None, :].repeat(1, length, 1)  # (b, l, d)

        return task + prompt + clap

    def get_tts_condition(self, data: dict, length: int) -> Tensor:
        r"""Get text to speech condition."""

        task = self.t5_encoder(data["task"])  # (b, l_in, d)
        task = task.mean(dim=1, keepdims=True).repeat(1, length, 1)  # (b, l, d)
        
        content = self.char_encoder(data["content"])  # (b, l')
        content = self.char_embedder(content)  # (b, l', d)
        content = self.char_aligner(content, length)

        return task + content

    def get_tta_condition(self, data: dict, length: int) -> Tensor:
        r"""Get text to audio condition."""
        task = self.t5_encoder(data["task"])
        task = task.mean(dim=1, keepdims=True).repeat(1, length, 1)  # (b, l, d)

        prompt = self.t5_encoder(data["prompt"])  # (b, l', d)
        prompt = self.t5_aligner(prompt, length)  # (b, l, d)

        clap = self.clap_text_encoder(data["prompt"])  # (b, d)
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