from torch import Tensor
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LambdaLR


def get_optimizer_and_scheduler(
    configs: dict, 
    params: list[Tensor]
) -> tuple[Optimizer, LambdaLR | None]:
    r"""Get optimizer and scheduler."""

    lr = float(configs["train"]["lr"])
    warm_up_steps = configs["train"]["warm_up_steps"]
    optimizer_name = configs["train"]["optimizer"]

    if optimizer_name == "AdamW":
        optimizer = AdamW(params=params, lr=lr)

    if warm_up_steps:
        lr_lambda = LinearWarmUp(warm_up_steps)
        scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = None

    return optimizer, scheduler


class LinearWarmUp:
    r"""Linear learning rate warm up scheduler."""

    def __init__(self, warm_up_steps: int) -> None:
        self.warm_up_steps = warm_up_steps

    def __call__(self, step: int) -> float:
        if step <= self.warm_up_steps:
            return step / self.warm_up_steps
        else:
            return 1.