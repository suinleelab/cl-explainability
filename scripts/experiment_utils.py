"""Utility functions for attribution experiments."""
import argparse
import os
import random
import sys
from typing import List, Optional, Tuple, Union

import constants
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from cl_explain.data.datasets import MURAImageDataset
from cl_explain.encoders.simclr.resnet_wider import resnet50x1, resnet50x2, resnet50x4
from cl_explain.encoders.simsiam.resnet import resnet18
from cl_explain.modules.classifier import LitClassifier


def parse_args(evaluate: bool = False, meta: bool = False):
    """Parse command line input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "encoder_name",
        type=str,
        choices=["simclr_x1", "simsiam_18", "classifier_18"],
        help="name of pre-trained encoder to explain",
    )
    parser.add_argument(
        "explanation_name",
        type=str,
        choices=[
            "self_weighted",
            "contrastive_self_weighted",
            "corpus",
            "contrastive_corpus",
        ],
        help="explanation behavior for feature attribution methods",
    )
    parser.add_argument(
        "attribution_name",
        type=str,
        choices=[
            "vanilla_grad",
            "int_grad",
            "smooth_vanilla_grad",
            "smooth_int_grad",
            "kernel_shap",
            "gradient_shap",
            "rise",
            "random_baseline",
        ],
        help="name of feature attribution method to use",
    )
    parser.add_argument(
        "--normalize-similarity",
        action="store_true",
        help="flag to normalize dot product similarity to use cosine similarity",
        dest="normalize_similarity",
    )
    parser.add_argument(
        "--different-classes",
        action="store_true",
        help="flag to have explicands and corpus from different classes",
        dest="different_classes",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="imagenet",
        choices=["imagenet", "cifar", "mura"],
        help="name of dataset to use",
        dest="dataset_name",
    )
    parser.add_argument(
        "--explicand-size",
        type=int,
        default=25,
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
        "--foil-size",
        type=int,
        default=1500,
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
            "--resample-eval-foil",
            action="store_true",
            help="whether to resample the foil set for evaluation",
            dest="resample_eval_foil",
        )
        parser.add_argument(
            "--eval-foil-size",
            type=int,
            default=1500,
            help="if resampling foil samples, the foil size to resample",
            dest="eval_foil_size",
        )
        parser.add_argument(
            "--comprehensive",
            action="store_true",
            help="whether to evaluate with a comprehensive list of metrics",
            dest="comprehensive",
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
        parser.add_argument(
            "--one-seed",
            action="store_true",
            help="whether to run experiment with only the specified seed",
            dest="one_seed",
        )
    args = parser.parse_args()
    print(f"Running {sys.argv[0]} with arguments")
    for arg in vars(args):
        print(f"\t{arg}={getattr(args, arg)}")
    return args


def get_device(
    use_gpu: bool, gpu_num: Optional[Union[int, List[int]]] = None
) -> Union[None, int, List[int]]:
    """Get device name or indices."""
    if use_gpu:
        if gpu_num is not None:
            device = gpu_num
        else:
            device = 1
    else:
        device = None  # None for cpu.
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
    normalize: bool = False,
    augment: bool = False,
) -> Tuple[Dataset, DataLoader, List[str]]:
    """Load image data."""
    transform_list = []
    if dataset_name in ["imagenette2", "imagenet", "mura"]:
        transform_list.append(transforms.Resize(256))
        transform_list.append(transforms.CenterCrop(224))
    elif dataset_name in ["cifar"]:
        transform_list.append(
            transforms.Resize(
                int(32 * (8 / 7)), interpolation=transforms.InterpolationMode.BICUBIC
            )
        )
        transform_list.append(transforms.CenterCrop(32))
    else:
        raise NotImplementedError(f"{dataset_name} loading is not implemented!")

    if augment:
        if dataset_name in ["cifar"]:
            raise NotImplementedError(
                f"{dataset_name} augmentations are not implemented!"
            )
        transform_list.append(transforms.RandomVerticalFlip())
        transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.RandomRotation(degrees=30))
    transform_list.append(transforms.ToTensor())
    if normalize:
        if dataset_name in ["imagenette2", "imagenet", "mura"]:
            transform_list.append(
                transforms.Normalize(
                    mean=constants.IMAGENET_MEAN,
                    std=constants.IMAGENET_STD,
                )
            )
        elif dataset_name in ["cifar"]:
            transform_list.append(
                transforms.Normalize(
                    mean=constants.CIFAR_MEAN,
                    std=constants.CIFAR_STD,
                )
            )
    transform = transforms.Compose(transform_list)

    if dataset_name == "imagenette2":
        dataset_path = os.path.join(constants.DATA_PATH, dataset_name, subset)
        dataset = torchvision.datasets.ImageFolder(
            dataset_path,
            transform=transform,
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        class_map = dataset.find_classes(dataset_path)[0]
    elif dataset_name == "imagenet":
        dataset_path = os.path.join(
            constants.DATA_PATH, dataset_name, "ILSVRC/Data/CLS-LOC", subset
        )
        dataset = torchvision.datasets.ImageFolder(
            dataset_path,
            transform=transform,
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        class_map = dataset.find_classes(dataset_path)[0]
    elif dataset_name == "mura":
        if subset == "val":
            subset = "valid"
        dataset_path = os.path.join(constants.DATA_PATH, dataset_name, "MURA-v1.1")
        dataset = MURAImageDataset(dataset_path, subset, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        class_map = dataset.classes
    elif dataset_name == "cifar":
        is_train = True
        if subset == "val":
            is_train = False
        dataset_path = os.path.join(constants.DATA_PATH, dataset_name + "10")
        dataset = torchvision.datasets.CIFAR10(
            dataset_path, train=is_train, transform=transform
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        class_map = dataset.classes
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
    elif "simsiam" in encoder_name:
        state_dict_path = os.path.join(constants.ENCODER_PATH, "simsiam")
        if encoder_name == "simsiam_18":
            encoder = resnet18(low_dim=10)
            state_dict_path = os.path.join(state_dict_path, "resnet18.pth.tar")
        else:
            _encoder_not_implemented_error(encoder_name)
        state_dict = torch.load(state_dict_path, map_location="cpu")
        encoder.load_state_dict(state_dict["state_dict"])
    elif "classifier" in encoder_name:
        ckpt_path = os.path.join(constants.ENCODER_PATH, "classifier")
        if encoder_name == "classifier_18":
            model = LitClassifier.load_from_checkpoint(
                os.path.join(ckpt_path, "resnet18.ckpt")
            )
            model.freeze()  # Essential to ensure model is usable for inference.
            encoder = model.network
        else:
            _encoder_not_implemented_error(encoder_name)
    else:
        _encoder_not_implemented_error(encoder_name)
    return encoder


def get_result_path(
    dataset_name: str,
    encoder_name: str,
    normalize_similarity: bool,
    explanation_name: str,
    attribution_name: str,
    seed: int,
) -> str:
    """Generate path for storing results."""
    if normalize_similarity:
        method_name = "normalized_"
    else:
        method_name = "unnormalized_"
    method_name += explanation_name + "_" + attribution_name
    return os.path.join(
        constants.RESULT_PATH,
        dataset_name,
        encoder_name,
        method_name,
        f"{seed}",
    )


def get_output_filename(
    different_classes: bool,
    corpus_size: int,
    explanation_name: str,
    foil_size: int,
    explicand_size: int,
    attribution_name: str,
    superpixel_dim: int,
    removal: str,
    blur_strength: float,
) -> str:
    """Get output filename for saving attribution results."""
    if different_classes:
        output_filename = "diff_class_outputs"
    else:
        output_filename = "same_class_outputs"

    output_filename += f"_explicand_size={explicand_size}"
    if "corpus" in explanation_name:
        output_filename += f"_corpus_size={corpus_size}"
    if "contrastive" in explanation_name:
        output_filename += f"_foil_size={foil_size}"
    if attribution_name in constants.SUPERPIXEL_ATTRIBUTION_METHODS:
        output_filename += f"_superpixel_dim={superpixel_dim}"
    output_filename += f"_removal={removal}"
    if removal == "blurring":
        output_filename += f"_blur_strength={blur_strength:.1f}"
    output_filename += ".pkl"
    return output_filename


def get_image_dataset_meta(dataset_name: str) -> Tuple[int, int, str]:
    """Get meta information about an image dataset."""
    if dataset_name in ["imagenet", "imagenette2"]:
        img_h = 224
        img_w = 224
        removal = "blurring"  # Appropriate pixel removal operation.
    elif dataset_name in ["cifar"]:
        img_h = 32
        img_w = 32
        removal = "blurring"
    elif dataset_name in ["mura"]:
        img_h = 224
        img_w = 224
        removal = "blurring"
    else:
        raise NotImplementedError(f"dataset={dataset_name} is not implemented!")
    return img_h, img_w, removal


def get_black_baseline(
    explicand: torch.Tensor, dataset_name: str, normalize: bool = True
) -> torch.Tensor:
    """
    Get black baseline values for a batch of explicand inputs.

    Args:
    ----
        explicand: A batch of explicands with shape `(batch_size, *)`, where `*`
            indicates the model input size for one sample.
        dataset_name: Name of the dataset where the explicands come from. This is to
            determine the channel means and standard deviations for normalization.
        normalize: Whether to normalize the black baseline values.

    Returns
    -------
        Black baseline values with shape `(batch_size, *)`, on the same device as
        `explicand`.
    """
    black_baseline = torch.zeros_like(explicand)
    if normalize:
        if dataset_name in ["imagenette2", "imagenet", "mura"]:
            normalization_function = transforms.Normalize(
                mean=constants.IMAGENET_MEAN,
                std=constants.IMAGENET_STD,
            )
        elif dataset_name in ["cifar"]:
            normalization_function = transforms.Normalize(
                mean=constants.CIFAR_MEAN,
                std=constants.CIFAR_STD,
            )
        else:
            raise NotImplementedError(
                f"dataset_name={dataset_name} is not implemented!"
            )
        black_baseline = normalization_function(black_baseline)
    return black_baseline
