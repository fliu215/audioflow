from torch import Tensor
import torch.nn as nn
import torch


def requires_grad(model: nn.Module, flag=True) -> None:
    for p in model.parameters():
        p.requires_grad = flag


def trim_target_latent(data: dict) -> dict:
    for k in ["target_latent", "target_mask"]:
        data[k] = data[k][:, 0 : max(data["target_length"])]
    return data


def to_device(data: dict, device) -> dict:
    for k, v in data.items():
        if isinstance(v, Tensor):
            data[k] = v.to(device)
    return data


def mean(x: Tensor, mask: Tensor) -> torch.float:
    r"""

    Args:
        x: (b, l, d)
        mask: (b, l)

    Returns:
        out: (b, d)
    """
    out = (x * mask[:, :, None]).sum(1) / mask.sum(dim=1, keepdims=True)  # (b, d)
    return out.mean()


def check_masks_type(masks: list[Tensor], dtype) -> bool:
    return all(mask.dtype == dtype for mask in masks)


def save(model: nn.Module, path) -> None:
    r"""Save model into a checkpoint."""
    ckpt = {}
    for k, v in model.named_children():
        if k in ["adapter"]:
            ckpt[k] = get_saveable_state_dict(v)
        else:
            ckpt[k] = v.state_dict()

    torch.save(ckpt, path)


def get_saveable_state_dict(model: nn.Module) -> dict:
    
    excluded = [n for n, m in model.named_modules() if hasattr(m, "saveable") and m.saveable is False]

    return {
        k: v for k, v in model.state_dict().items()
        if not any(k == p or k.startswith(p + ".") for p in excluded)
    }



def load(model: nn.Module, path) -> nn.Module:
    r"""Load model from a checkpoint."""
    ckpt = torch.load(path, map_location="cpu")

    for k, v in model.named_children():
        v.load_state_dict(ckpt[k], strict=False)

    return model