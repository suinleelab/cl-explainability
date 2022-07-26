"""Utility functions."""
import os
import random
from typing import List, Optional, Tuple

import constants
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from cl_explain.encoders.simclr.resnet_wider import resnet50x1, resnet50x2, resnet50x4


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


def load_data(
    dataset_name: str, batch_size: int
) -> Tuple[Dataset, DataLoader, List[str]]:
    """Load data."""
    if dataset_name == "imagenette2":
        dataset_path = os.path.join(constants.DATA_PATH, dataset_name, "val")
        dataset = torchvision.datasets.ImageFolder(
            dataset_path,
            transform=transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ]
            ),
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        class_map = dataset.find_classes(dataset_path)[0]
    else:
        raise NotImplementedError(f"{dataset_name} loading is not implemented!")
    return dataset, dataloader, class_map


def _encoder_not_implemented_error(encoder_name: str) -> None:
    raise NotImplementedError(f"{encoder_name} loading is not implemented!")


def load_encoder(encoder_name: str) -> nn.Module:
    """Load encoder."""
    encoder = None
    if "simclr" in encoder_name:
        state_dict_path = os.path.join(constants.ENCODER_PATH, "simclr")
        if encoder_name == "simclr_x1":
            encoder = resnet50x1()
            state_dict_path = os.path.join(state_dict_path, "resnet50-1x.pth")
        elif encoder_name == "simclr_x2":
            encoder = resnet50x2()
            state_dict_path = os.path.join(state_dict_path, "resnet50-2x.pth")
        elif encoder_name == "simclr_x4":
            encoder = resnet50x4()
            state_dict_path = os.path.join(state_dict_path, "resnet50-4x.pth")
        else:
            _encoder_not_implemented_error(encoder_name)
        state_dict = torch.load(state_dict_path, map_location="cpu")
        encoder.load_state_dict(state_dict["state_dict"])
    else:
        _encoder_not_implemented_error(encoder_name)
    return encoder
