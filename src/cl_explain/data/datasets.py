"""PyTorch dataset classes."""
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from torchvision.datasets import ImageFolder, VisionDataset
from torchvision.datasets.folder import default_loader


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


class MURAImageDataset(VisionDataset):
    """
    Dataset class for MURA where each image is considered a classification instance.

    The MURA dataset can be obtained from
    https://stanfordmlgroup.github.io/competitions/mura/.

    Args:
    ----
        root: MURA root directory containing all data subsets and meta information
            files.
        subset: The dataset subset to load. Either "train" or "valid".
        transform: A function/transform that takes in an PIL image and returns a
            transformed version. E.g, `transforms.RandomCrop`.
        target_transform: A function/transform that takes in the target and
            transforms it.
        loader: A function to load an image given its path.
    """

    def __init__(
        self,
        root: str,
        subset: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
    ) -> None:
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
        )
        available_subsets = ["train", "valid"]
        assert (
            subset in available_subsets
        ), f"subset = {subset} is not one of {available_subsets}!"
        self.subset = subset
        self.loader = loader

        self.img_path_file = os.path.join(root, f"{subset}_image_paths.csv")
        self.classes = ["negative", "positive"]
        self.class_to_idx = {"negative": 0, "positive": 1}
        self.samples = self.make_dataset()

    def make_dataset(self) -> List[Tuple[str, int]]:
        """Generate a list of samples of the form `(path_to_sample, class)`."""
        with open(self.img_path_file, "r") as handle:
            img_paths = handle.readlines()
        img_paths = [path.replace("\n", "") for path in img_paths]  # Remove end lines.
        img_paths = ["/".join(path.split("/")[1:]) for path in img_paths]
        # Remove MURA root directory name (e.g., "MURA-v1.1") because it's already part
        # of self.root.

        class_indices = [
            self.class_to_idx[path.split("/")[-2].split("_")[-1]] for path in img_paths
        ]
        img_paths = [os.path.join(self.root, path) for path in img_paths]
        return list(zip(img_paths, class_indices))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Get a data instance."""
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self) -> int:
        """Count number of data instances."""
        return len(self.samples)
