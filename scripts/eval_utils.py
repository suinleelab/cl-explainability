"""Utility functions."""
import random
from typing import Optional

import numpy as np
import torch


def get_device(use_gpu: bool, gpu_num: Optional[int] = None) -> str:
    """Get device name."""
    if use_gpu:
        if gpu_num is not None:
            device = f"cuda:{gpu_num}"
        else:
            device = "cuda:0"
    else:
        device = "cpu"
    return device


def make_reproducible(seed: int = 123) -> None:
    """Make evaluation reproducible."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
