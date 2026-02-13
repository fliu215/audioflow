import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


class CLIPTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.dim = self.model.config.vision_config.projection_dim

    def forward(self, text: list[str]) -> Tensor:
        r"""Convert text into CLAP embedding.

        b: batch_size
        d: dim

        Args:
            text: list[str]

        Returns:
            latent: (b, d)
        """
        device = next(self.parameters()).device
        # inputs = self.tokenizer(text, padding=True, return_tensors="pt").to(device)

        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)

        outputs = self.model(**inputs)

        # with torch.no_grad():
        #     self.encoder.eval()
        #     latent = self.encoder.get_text_features(**inputs)  # (b, d)

        # return latent


class CLIPVideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.dim = self.model.config.vision_config.projection_dim

    def forward(self, video: list[str]) -> Tensor:
        r"""Convert text into CLAP embedding.

        b: batch_size
        d: dim
        t: frames
        h: height
        w: weight
        c: channels

        Args:
            video: (b, t, h, w, c)

        Returns:
            latent: (b, d)
        """
        device = next(self.parameters()).device
        inputs = self.tokenizer(text, padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            self.encoder.eval()
            latent = self.encoder.get_text_features(**inputs)  # (b, d)

        return latent
