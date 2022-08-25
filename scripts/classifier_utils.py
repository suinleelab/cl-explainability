"""Utility functions for training and evaluating classifiers."""
import argparse
import os
import sys
from typing import List, Tuple

import constants
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def parse_classifier_args(evaluate: bool = False):
    """Parse command line arguments for a classifier."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "classifier_name",
        type=str,
        choices=["resnet18", "resnet50"],
        help="name of pre-trained classifier to fine tune",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="mura",
        choices=["mura"],
        help="name of dataset to use for training the classifier",
        dest="dataset_name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="batch size for all data loaders",
        dest="batch_size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
        help="weight decay parameter",
        dest="weight_decay",
    )
    parser.add_argument(
        "--lr-step-size",
        type=int,
        default=10,
        help="learning rate decay step size in epochs",
        dest="lr_step_size",
    )
    parser.add_argument(
        "--lr-gamma",
        type=float,
        default=0.1,
        help="multiplicative factor for learning rate decay",
        dest="lr_gamma",
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
        nargs="+",
        help="if --use-gpu is enabled, controls which GPUs to use",
        dest="gpu_num",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="seed for random processes",
        dest="seed",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="flag for using small number of samples and epochs for debugging",
        dest="debug",
    )
    if evaluate:
        parser.add_argument(
            "--ckpt-filename",
            type=str,
            default="last.ckpt",
            help="model checkpoint filename",
            dest="ckpt_filename",
        )
        parser.add_argument(
            "--eval-batch-size",
            type=int,
            default=256,
            help="evaluation batch size for all data loaders",
            dest="eval_batch_size",
        )
    args = parser.parse_args()
    print(f"Running {sys.argv[0]} with arguments")
    for arg in vars(args):
        print(f"\t{arg}={getattr(args, arg)}")
    return args


def get_classifier_output_path(
    dataset_name: str,
    classifier_name: str,
    seed: int,
    lr: float,
    weight_decay: float,
    lr_step_size: int,
    lr_gamma: float,
    batch_size: int,
    max_epochs: int,
) -> str:
    """Generate path for storing classifier outputs."""
    hyperparams = f"lr={lr}_weight_decay={weight_decay}"
    hyperparams += f"_lr_step_size={lr_step_size}_lr_gamma={lr_gamma}"
    hyperparams += f"_batch_size={batch_size}"
    hyperparams += f"_max_epochs={max_epochs}"
    return os.path.join(
        constants.MODEL_OUTPUT_PATH,
        dataset_name,
        classifier_name,
        f"{seed}",
        hyperparams,
    )


def split_train_val_dataset(
    train_val_dataset: Dataset, val_size: float = 0.2, seed: int = 42
) -> Tuple[List, List]:
    """
    Split a dataset into a training and a validation subset.

    Args:
    ----
        train_val_dataset: A PyTorch Dataset with an attribute `samples`, which is a
            sequence of tuple where the second element of each tuple is the data label.
        val_size: Proportion of dataset to split into the validation subset.
        seed: Random seed for splitting the training and validation subset.

    Returns
    -------
        A tuple of two lists. The first element is the list of training data indices.
        The second is the list of validation data indices.
    """
    train_val_idx = [i for i in range(len(train_val_dataset.samples))]
    train_val_targets = [sample[1] for sample in train_val_dataset.samples]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        random_state=seed,
        shuffle=True,
        stratify=train_val_targets,
    )
    return train_idx, val_idx
