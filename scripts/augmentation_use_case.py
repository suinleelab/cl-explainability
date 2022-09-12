"""Contrastive corpus attribution for understanding data augmentation in SimCLR."""
import argparse
import os
import sys
from typing import Dict, List

import constants
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms as transforms
from captum.attr import GradientShap, IntegratedGradients
from experiment_utils import get_device, load_encoder
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from cl_explain.attributions.rise import RISE
from cl_explain.explanations.contrastive_weighted_score import ContrastiveWeightedScore


def parse_augmentation_use_case_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "attribution_name",
        type=str,
        choices=["int_grad", "gradient_shap", "rise"],
        help="name of feature attribution method to use",
    )
    parser.add_argument(
        "--synset",
        type=str,
        default="n02102040",  # English springer as default.
        help="synset for the Imagenet class to study",
        dest="synset",
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="whether to run debug mode with only a few explicands",
        dest="debug",
    )
    args = parser.parse_args()
    print(f"Running {sys.argv[0]} with arguments")
    for arg in vars(args):
        print(f"\t{arg}={getattr(args, arg)}")
    return args


def get_synset_indices(
    dataset: torchvision.datasets.ImageFolder, synset: str
) -> List[int]:
    """
    Get indices in an Imagenet ImageFolder dataset that correspond to a synset class.

    Args:
    ----
        dataset: An ImageFolder dataset for Imagenet.
        synset: An Imagenet synset class label.

    Returns
    -------
        A list of integer indices corresponding to the samples in `dataset` that are
        in the `synset` class.
    """
    labels = [sample[0].split("/")[-2] for sample in dataset.samples]
    sample_size = len(labels)
    synset_indices = [i for i in range(sample_size) if labels[i] == synset]
    return synset_indices


def get_transform_dict() -> Dict[str, transforms.Compose]:
    """Get a dictionary of image transformations for Imagenet without normalization."""
    basic_transform_list = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    original_transform_list = [  # Original image.
        *basic_transform_list,
        transforms.ToTensor(),
    ]
    crop_transform_list = [  # Random cropping.
        transforms.RandomResizedCrop(224, scale=(0.2, 0.5)),
        transforms.ToTensor(),
    ]
    gray_transform_list = [  # Grayscale transformation.
        *basic_transform_list,
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]
    jitter_transform_list = [  # Color jitter transformation.
        *basic_transform_list,
        transforms.ColorJitter(  # SimCLR parameters for color jitter.
            brightness=0.8,
            contrast=0.8,
            saturation=0.8,
            hue=0.2,
        ),
        transforms.ToTensor(),
    ]
    erase_transform_list = [  # Cut out a random portion of the image.
        *basic_transform_list,
        transforms.ToTensor(),
        transforms.RandomErasing(p=1.0, scale=(0.1, 0.25), value=0.5),
    ]
    return {
        "original": transforms.Compose(original_transform_list),
        "crop": transforms.Compose(crop_transform_list),
        "gray": transforms.Compose(gray_transform_list),
        "jitter": transforms.Compose(jitter_transform_list),
        "erase": transforms.Compose(erase_transform_list),
    }


def main():
    """Main function."""
    args = parse_augmentation_use_case_args()
    pl.seed_everything(args.seed, workers=True)
    device = get_device(args.use_gpu, args.gpu_num)
    get_baseline = transforms.GaussianBlur(21, sigma=args.blur_strength).to(device)

    print("Loading encoder...")
    encoder = load_encoder("simclr_x1")
    encoder.eval()
    encoder.to(device)

    synset_mapping = {}
    with open(
        os.path.join(constants.DATA_PATH, "imagenet", "LOC_synset_mapping.txt"), "r"
    ) as handle:
        lines = handle.readlines()
    for line in lines:
        synset = line.split(" ")[0]
        label = " ".join(line.split(" ")[1:])
        label = label.replace("\n", "")
        label = label.split(",")[0]  # Only take the first among all equivalent labels.
        synset_mapping[synset] = label

    val_dataset_path = os.path.join(
        constants.DATA_PATH, "imagenet", "ILSVRC/Data/CLS-LOC", "val"
    )
    val_synset_indices = get_synset_indices(
        torchvision.datasets.ImageFolder(val_dataset_path), args.synset
    )
    if args.debug:
        val_synset_indices = val_synset_indices[:5]
    train_dataset_path = os.path.join(
        constants.DATA_PATH, "imagenet", "ILSVRC/Data/CLS-LOC", "train"
    )

    print("Setting up explicand loader...")
    # Get original and augmented images for the validation synset images.
    flip = transforms.RandomHorizontalFlip(p=1.0)  # Callable for flipping an image.
    transform_dict = get_transform_dict()
    val_dataloader_dict = {}
    for key, transform in transform_dict.items():
        val_transform_dataset = Subset(
            torchvision.datasets.ImageFolder(val_dataset_path, transform=transform),
            indices=val_synset_indices,
        )
        val_dataloader_dict[key] = DataLoader(
            val_transform_dataset, batch_size=len(val_synset_indices), shuffle=False
        )
    original_img, _ = next(iter(val_dataloader_dict["original"]))
    flip_img = flip(original_img)
    crop_img, _ = next(iter(val_dataloader_dict["crop"]))
    crop_flip_img = flip(crop_img)
    gray_img, _ = next(iter(val_dataloader_dict["gray"]))
    jitter_img, _ = next(iter(val_dataloader_dict["jitter"]))
    erase_img, _ = next(iter(val_dataloader_dict["erase"]))
    rotate_img = transforms.functional.rotate(original_img, angle=90)

    # Set up data loader for coupled original and augmented images.
    aug_dataset = torch.utils.data.TensorDataset(
        original_img,
        flip_img,
        crop_img,
        crop_flip_img,
        gray_img,
        jitter_img,
        erase_img,
        rotate_img,
    )
    aug_dataloader = DataLoader(aug_dataset, batch_size=args.batch_size, shuffle=False)
    aug_name_list = [
        "original",
        "flip",
        "crop",
        "crop_flip",
        "gray",
        "jitter",
        "erase",
        "rotate",
    ]

    print("Setting up foil loader...")
    train_dataset = torchvision.datasets.ImageFolder(
        train_dataset_path,
        transform=transform_dict["original"],
    )
    foil_indices = torch.arange(len(train_dataset.samples))
    foil_indices = foil_indices[torch.randperm(foil_indices.size(0))][: args.foil_size]
    foil_dataloader = DataLoader(
        Subset(train_dataset, indices=foil_indices),
        batch_size=args.batch_size,
        shuffle=False,
    )

    print("Running and saving attribution...")
    # Run and save feature attribution and the corresponding image.
    explanation_model = ContrastiveWeightedScore(
        encoder=encoder,
        foil_dataloader=foil_dataloader,
        normalize=True,
        batch_size=args.batch_size,
    )
    result_path = os.path.join(
        constants.RESULT_PATH, "augmentation_use_case", f"{args.seed}", args.synset
    )
    os.makedirs(result_path, exist_ok=True)

    overall_img_counter = 0
    pred_list = []
    for img_list in tqdm(aug_dataloader):
        batch_size = img_list[0].size(0)
        explanation_model.generate_weight(
            img_list[0].detach().clone().to(device)
        )  # Original image as corpus.

        pred = encoder(img_list[0].to(device), apply_eval_head=True).argmax(dim=-1)
        pred = pred.detach().cpu()
        pred_list.append(pred)

        for j, img in enumerate(img_list):
            img = img.to(device)  # Original or augmented image as explicand.
            baseline = get_baseline(img)
            img.requires_grad = True

            if args.attribution_name == "int_grad":
                attribution_model = IntegratedGradients(explanation_model)
                attribution = attribution_model.attribute(img, baselines=baseline)
            elif args.attribution_name == "gradient_shap":
                attribution_model = GradientShap(explanation_model)
                attribution = attribution_model.attribute(
                    img,
                    baselines=baseline,
                    n_samples=50,
                    stdevs=0.2,
                )
            elif args.attribution_name == "rise":
                attribution_model = RISE(explanation_model)
                attribution = attribution_model.attribute(
                    img,
                    grid_shape=(7, 7),
                    baselines=baseline,
                    mask_prob=0.5,
                    n_samples=5000,
                    normalize_by_mask_prob=True,
                )
            else:
                raise NotImplementedError(
                    f"{args.attribution_name} attribution is not implemented!"
                )

            attribution = attribution.detach().cpu()
            aug_name = aug_name_list[j]
            img_counter = overall_img_counter
            for i in range(batch_size):
                img_result_path = os.path.join(result_path, f"img_{img_counter}")
                os.makedirs(img_result_path, exist_ok=True)
                torch.save(
                    img[i].detach().cpu(),
                    os.path.join(img_result_path, f"{aug_name}_img.pt"),
                )
                torch.save(
                    attribution[i],
                    os.path.join(
                        img_result_path, f"{aug_name}_{args.attribution_name}.pt"
                    ),
                )
                img_counter += 1
        overall_img_counter += batch_size
    pred_list = torch.cat(pred_list)

    print("Plotting results...")
    with PdfPages(
        os.path.join(result_path, f"all_{args.attribution_name}_results.pdf")
    ) as pdf:
        num_img_per_page = 5
        page_img_counter = 0
        fig, axes = plt.subplots(
            ncols=len(aug_name_list),
            nrows=2 * num_img_per_page,
            figsize=(24, 6 * num_img_per_page),
        )

        for i in tqdm(range(overall_img_counter)):
            img_result_path = os.path.join(result_path, f"img_{i}")
            pred_synset = train_dataset.classes[pred_list[i]]
            pred_label = synset_mapping[pred_synset]
            if pred_synset == args.synset:
                pred_info = "Correct classification."
            else:
                pred_info = "Misclassification."
            pred_info += f" Predicted: {pred_synset} ({pred_label})."

            for j, aug_name in enumerate(aug_name_list):
                aug_img = torch.load(
                    os.path.join(img_result_path, f"{aug_name}_img.pt"),
                    map_location="cpu",
                )
                aug_attribution = torch.load(
                    os.path.join(
                        img_result_path, f"{aug_name}_{args.attribution_name}.pt"
                    ),
                    map_location="cpu",
                ).mean(dim=0)

                aug_img_plot_idx = page_img_counter * 2
                axes[aug_img_plot_idx, j].imshow(aug_img.permute(1, 2, 0))
                axes[aug_img_plot_idx, j].set_xticks([])
                axes[aug_img_plot_idx, j].set_yticks([])
                if j == 0:
                    axes[aug_img_plot_idx, j].set_title(pred_info)

                aug_attribution = torch.nn.functional.relu(
                    aug_attribution
                )  # Focus on positive attributions for this use case.
                aug_attribution_scale = aug_attribution.max()
                axes[aug_img_plot_idx + 1, j].imshow(
                    aug_attribution,
                    cmap="seismic",
                    vmin=-aug_attribution_scale,
                    vmax=aug_attribution_scale,
                )
                axes[aug_img_plot_idx + 1, j].set_xticks([])
                axes[aug_img_plot_idx + 1, j].set_yticks([])

            page_img_counter += 1
            if page_img_counter == num_img_per_page:
                page_img_counter = 0
                pdf.savefig()
                plt.close(fig)
                fig, axes = plt.subplots(
                    ncols=len(aug_name_list),
                    nrows=2 * num_img_per_page,
                    figsize=(24, 6 * num_img_per_page),
                )
    print("Done!")


if __name__ == "__main__":
    main()
