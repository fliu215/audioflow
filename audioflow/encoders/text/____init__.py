import torch.nn as nn


def build_text_encoder(name: str) -> nn.Module:
    if name == "flan_t5":
        from audioflow.encoders.text.flan_t5 import FlanT5
        return FlanT5()

    else:
        raise ValueError(name)
