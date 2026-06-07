from torch.utils.data import Dataset


def get_dataset(configs: dict) -> Dataset:
    r"""Get dataset."""
    name = configs["dataset"]["name"]

    if name == "TTMDataset":
        from audioflow.datasets.ttm import TTMDataset
        return TTMDataset(configs["clip_duration"])

    elif name == "TTADataset":
        from audioflow.datasets.tta import TTADataset
        return TTADataset(configs["clip_duration"])

    else:
        raise ValueError(name)
