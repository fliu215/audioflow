from torch.utils.data import Dataset


def get_dataset(configs: dict) -> Dataset:
    r"""Get dataset.
    """
    name = configs["name"]

    if name == "TTMDataset":
        from .ttm import TTMDataset
        return TTMDataset(configs["crop_duration"])

    elif name == "TTADataset":
        from .tta import TTADataset
        return TTADataset(configs["crop_duration"])

    elif name == "TTSDataset":
        from .tts import TTSDataset
        return TTSDataset(configs["crop_duration"])

    elif name in ["MSSDataset", "Vocals2MusicDataset", "Mono2StereoDataset", "SuperResolutionDataset"]:
        from .mss import MSSDataset
        return MSSDataset(configs["crop_duration"])

    elif name == "Midi2AudioDataset":
        from .midi2audio import Midi2AudioDataset
        return Midi2AudioDataset(configs["crop_duration"])

    elif name == "V2ADataset":
        from .v2a import V2ADataset
        return V2ADataset(configs["crop_duration"])

    else:
        raise ValueError(name)
