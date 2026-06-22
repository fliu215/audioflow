import torch.nn as nn


def get_adapter(configs: dict) -> nn.Module:
    r"""Initialize adapter."""
    name = configs["name"]

    if name == "TTMAdapter":
        from .ttm import TTMAdapter
        return TTMAdapter(**configs)

    elif name == "TTAAdapter":
        from .tta import TTAAdapter
        return TTAAdapter(**configs)

    elif name == "TTSAdapter":
        from .tts import TTSAdapter
        return TTSAdapter(**configs)

    elif name == "Midi2AudioAdapter":
        from .midi2audio import Midi2AudioAdapter
        return Midi2AudioAdapter(**configs)

    elif name in ["MSSAdapter", "Vocals2MusicAdapter", "Mono2StereoAdapter", "SuperResolutionAdapter"]:
        from .mss import MSSAdapter
        return MSSAdapter(**configs)

    elif name == "MSSDiagAttAdapter":
        from .mss_diag_att import MSSDiagAttAdapter
        return MSSDiagAttAdapter(**configs)

    elif name == "Midi2AudioFullAttAdapter":
        from .midi2audio_full_att import Midi2AudioFullAttAdapter
        return Midi2AudioFullAttAdapter(**configs)

    elif name == "V2AAdapter":
        from .v2a import V2AAdapter
        return V2AAdapter(**configs)

    else:
        raise ValueError(name)