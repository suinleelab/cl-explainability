"""PyTorch dataset classes."""
import os
from typing import Dict, List, Tuple

from torchvision.datasets import ImageFolder


class ImageSubsetFolder(ImageFolder):
    """
    PyTorch Dataset class for a subset of image classes.

    Args:
    ----
        root: Root directory path to Imagenet data folder.
        subset_folders: List of subset folders to include.
        **kwargs: Other valid arguments for torchvision.datasets.ImageFolder.
    """

    def __init__(self, root: str, subset_folders: List[str], **kwargs) -> None:
        self.subset_folders = subset_folders  # Needed for super().__init__().
        super().__init__(root=root, **kwargs)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Find only classes in the subset folders."""
        classes = sorted(
            entry.name
            for entry in os.scandir(directory)
            if entry.is_dir() and entry.name in self.subset_folders
        )
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
