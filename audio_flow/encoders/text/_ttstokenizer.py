from ttstokenizer import IPATokenizer

import torch
from torch import Tensor
import torch.nn as nn


# class TTSTokenizer(nn.Module):
class TTSTokenizer:
    def __init__(self):
        super().__init__()
        self.tokenizer = IPATokenizer()
        self.vocab_size = len(self.tokenizer.ipatokens())
        # self.register_buffer(name="_dummy", tensor=torch.zeros(1))
        # self._dummy = nn.Linear(1, 1)
        
    def forward(self, text: list[str]) -> Tensor:

        for t in text:
            self.tokenizer(t)
            from IPython import embed; embed(using=False); os._exit(0)

        
        device = next(self.parameters()).device
        
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        from IPython import embed; embed(using=False); os._exit(0)
        
        with torch.no_grad():
            self.encoder.eval()
            latent = self.encoder(**inputs).last_hidden_state  # (b, l, d)

        return latent