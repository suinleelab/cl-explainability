"""Attribution methods that simply assign random noises."""
from typing import Callable, Union

import torch
import torch.nn as nn
from captum.attr import Attribution


class RandomBaseline(Attribution):
    """Attribution class that returns noises as attribution scores."""

    def __init__(self, forward_func: Union[Callable, nn.Module]) -> None:
        super().__init__(forward_func)

    def attribute(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.randn(inputs.shape).to(inputs.device)
