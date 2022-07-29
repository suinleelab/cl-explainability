"""Utility functions."""
import argparse
import os
import random
import sys
from typing import List, Optional, Tuple

import constants
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from cl_explain.encoders.simclr.resnet_wider import resnet50x1, resnet50x2, resnet50x4


def parse_args(evaluate: bool = False, meta: bool = False):
    """Parse command line input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "encoder_name",
        type=str,
        choices=["simclr_x1", "simclr_x2", "simclr_x4"],
        help="name of pre-trained encoder to explain",
    )
    parser.add_argument(
        "attribution_name",
        type=str,
        choices=["vanilla_grad", "int_grad", "kernel_shap", "random_baseline"],
        help="name of feature attribution method to use",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="imagenette2",
        choices=["imagenette2"],
        help="name of dataset to use",
        dest="dataset_name",
    )
    parser.add_argument(
        "--explicand-size",
        type=int,
        default=100,
        help="number of explicands per class",
        dest="explicand_size",
    )
    parser.add_argument(
        "--corpus-size",
        type=int,
        default=100,
        help="number of corpus examples per class",
        dest="corpus_size",
    )
    parser.add_argument(
        "--contrast",
        action="store_true",
        help="flag to use contrastive explanations with a corpus and foil set",
        dest="contrast",
    )
    parser.add_argument(
        "--foil-size",
        type=int,
        default=500,
        help="number of foil examples",
        dest="foil_size",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="batch size for all data loaders",
        dest="batch_size",
    )
    parser.add_argument(
        "--superpixel-dim",
        type=int,
        default=1,
        help="superpixel width and height for removing image pixels",
        dest="superpixel_dim",
    )
    parser.add_argument(
        "--blur-strength",
        type=float,
        default=5.0,
        help="strength of blurring when removing features by blurring",
        dest="blur_strength",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="flag to enable GPU usage",
        dest="use_gpu",
    )
    parser.add_argument(
        "--gpu-num",
        type=int,
        help="if --use-gpu is enabled, controls which GPU to use",
        dest="gpu_num",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="seed for random processes",
        dest="seed",
    )
    if evaluate:
        parser.add_argument(
            "--take-attribution-abs",
            action="store_true",
            help="flag to take absolute value of attributions during evaluation",
            dest="take_attribution_abs",
        )
        parser.add_argument(
            "--eval-superpixel-dim",
            type=int,
            default=1,
            help="superpixel width and height for removing pixels during evaluation",
            dest="eval_superpixel_dim",
        )
        parser.add_argument(
            "--eval-foil-size",
            type=int,
            default=500,
            help="number of foil samples for evaluating contrastive metrics",
            dest="eval_foil_size",
        )
    if meta:
        parser.add_argument(
            "--mode",
            type=str,
            default="attribute_and_eval",
            choices=["attribute_only", "eval_only", "attribute_and_eval"],
            help="for meta script, whether to run attribute.py and or eval.py",
            dest="mode",
        )
    args = parser.parse_args()
    print(f"Running {sys.argv[0]} with arguments")
    for arg in vars(args):
        print(f"\t{arg}={getattr(args, arg)}")
    return args


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
    dataset_name: str,
    subset: str,
    batch_size: int,
) -> Tuple[Dataset, DataLoader, List[str]]:
    """Load data."""
    if dataset_name == "imagenette2":
        dataset_path = os.path.join(constants.DATA_PATH, dataset_name, subset)
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


def get_result_path(
    dataset_name: str,
    encoder_name: str,
    attribution_name: str,
    seed: int,
    contrast: bool,
) -> str:
    """Generate path for storing results."""
    if contrast:
        attribution_name = "contrastive_" + attribution_name
    return os.path.join(
        constants.RESULT_PATH,
        dataset_name,
        encoder_name,
        attribution_name,
        f"{seed}",
    )


def get_output_filename(
    corpus_size: int,
    contrast: bool,
    foil_size: int,
    explicand_size: int,
    attribution_name: str,
    superpixel_dim: int,
    removal: str,
    blur_strength: float,
) -> str:
    """Get output filename for saving attribution results."""
    output_filename = "outputs"
    output_filename += f"_corpus_size={corpus_size}"
    if contrast:
        output_filename += f"_foil_size={foil_size}"
    output_filename += f"_explicand_size={explicand_size}"
    if attribution_name in constants.SUPERPIXEL_ATTRIBUTION_METHODS:
        output_filename += f"_superpixel_dim={superpixel_dim}"
    output_filename += f"_removal={removal}"
    if removal == "blurring":
        output_filename += f"_blur_strength={blur_strength:.1f}"
    output_filename += ".pkl"
    return output_filename


def get_image_dataset_meta(dataset_name: str) -> Tuple[int, int, str]:
    """Get meta information about an image dataset."""
    if dataset_name == "imagenette2":
        img_h = 224
        img_w = 224
        removal = "blurring"  # Appropriate pixel removal operation.
    else:
        raise NotImplementedError(f"dataset={dataset_name} is not implemented!")
    return img_h, img_w, removal
