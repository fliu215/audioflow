from phonemizer import phonemize

import torch
from torch import LongTensor
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


VOCAB = ["<pad>", "<bos>", "<eos>", "<sil>", "<unk>", '$', ';', ':', ',', '.', 
    '!', '?', '¡', '¿', '—', '…', '"', '«', '»', '“', '”', ' ', 'A', 'B', 'C', 
    'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 
    'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 
    'w', 'x', 'y', 'z', 'ɑ', 'ɐ', 'ɒ', 'æ', 'ɓ', 'ʙ', 'β', 'ɔ', 'ɕ', 'ç', 'ɗ', 
    'ɖ', 'ð', 'ʤ', 'ə', 'ɘ', 'ɚ', 'ɛ', 'ɜ', 'ɝ', 'ɞ', 'ɟ', 'ʄ', 'ɡ', 'ɠ', 'ɢ', 
    'ʛ', 'ɦ', 'ɧ', 'ħ', 'ɥ', 'ʜ', 'ɨ', 'ɪ', 'ʝ', 'ɭ', 'ɬ', 'ɫ', 'ɮ', 'ʟ', 'ɱ', 
    'ɯ', 'ɰ', 'ŋ', 'ɳ', 'ɲ', 'ɴ', 'ø', 'ɵ', 'ɸ', 'θ', 'œ', 'ɶ', 'ʘ', 'ɹ', 'ɺ', 
    'ɾ', 'ɻ', 'ʀ', 'ʁ', 'ɽ', 'ʂ', 'ʃ', 'ʈ', 'ʧ', 'ʉ', 'ʊ', 'ʋ', 'ⱱ', 'ʌ', 'ɣ', 
    'ɤ', 'ʍ', 'χ', 'ʎ', 'ʏ', 'ʑ', 'ʐ', 'ʒ', 'ʔ', 'ʡ', 'ʕ', 'ʢ', 'ǀ', 'ǁ', 'ǂ', 
    'ǃ', 'ˈ', 'ˌ', 'ː', 'ˑ', 'ʼ', 'ʴ', 'ʰ', 'ʱ', 'ʲ', 'ʷ', 'ˠ', 'ˤ', '˞', '↓', 
    '↑', '→', '↗', '↘', "'", '̩', 'ᵻ'
]



class CharEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.vocab = VOCAB
        self.phoneme2id = {p: i for i, p in enumerate(self.vocab)}
        self.id2phoneme = {i: p for i, p in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self._dummy = nn.Parameter(torch.zeros(1))
        
    def forward(self, text: list[str]) -> LongTensor:

        device = next(self.parameters()).device

        phones = phonemize(
            text,
            language="en-us",
            backend="espeak",
            strip=True,
            preserve_punctuation=False,
            with_stress=True
        )

        ids = [LongTensor([self.phoneme2id[p] for p in phone]) for phone in phones]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.phoneme2id["<pad>"]).to(device)
        
        return ids