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
        "--plot-only",
        action="store_true",
        help="flag to plot existing attribution results without running attribution",
        dest="plot_only",
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
    classes = val_dataloader_dict["original"].dataset.dataset.classes
    class_to_idx = val_dataloader_dict["original"].dataset.dataset.class_to_idx

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
    num_explicands = aug_dataset.tensors[0].size(0)
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

    result_path = os.path.join(
        constants.RESULT_PATH, "augmentation_use_case", f"{args.seed}", args.synset
    )
    os.makedirs(result_path, exist_ok=True)

    if not args.plot_only:
        print("Setting up foil loader...")
        train_dataset = torchvision.datasets.ImageFolder(
            train_dataset_path,
            transform=transform_dict["original"],
        )
        foil_indices = torch.arange(len(train_dataset.samples))
        foil_indices = foil_indices[torch.randperm(foil_indices.size(0))][
            : args.foil_size
        ]
        foil_dataloader = DataLoader(
            Subset(train_dataset, indices=foil_indices),
            batch_size=args.batch_size,
            shuffle=False,
        )
        explanation_model = ContrastiveWeightedScore(
            encoder=encoder,
            foil_dataloader=foil_dataloader,
            normalize=True,
            batch_size=args.batch_size,
        )

        print("Running and saving attribution...")
        overall_img_counter = 0
        for img_list in tqdm(aug_dataloader):
            batch_size = img_list[0].size(0)
            explanation_model.generate_weight(
                img_list[0].detach().clone().to(device)
            )  # Original image as corpus.

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

    true_idx = class_to_idx[args.synset]
    true_pred_prob_list = [[] for _ in range(len(aug_name_list))]
    max_pred_prob_list = [[] for _ in range(len(aug_name_list))]
    pred_list = [[] for _ in range(len(aug_name_list))]
    for img_list in tqdm(aug_dataloader):
        for j, img in enumerate(img_list):
            pred_prob = (
                encoder(img.to(device), apply_eval_head=True).softmax(dim=-1).detach()
            )
            true_pred_prob_list[j].append(pred_prob[:, true_idx].cpu())
            max_pred_prob, pred = pred_prob.max(dim=-1)
            max_pred_prob_list[j].append(max_pred_prob.cpu())
            pred_list[j].append(pred.cpu())
    true_pred_prob_list = [
        torch.cat(true_pred_prob) for true_pred_prob in true_pred_prob_list
    ]
    max_pred_prob_list = [
        torch.cat(max_pred_prob) for max_pred_prob in max_pred_prob_list
    ]
    pred_list = [torch.cat(pred) for pred in pred_list]

    print("Plotting results...")
    with PdfPages(
        os.path.join(result_path, f"all_{args.attribution_name}_results.pdf")
    ) as pdf:
        num_img_per_page = 4
        page_img_counter = 0
        fig, axes = plt.subplots(
            ncols=len(aug_name_list),
            nrows=3 * num_img_per_page,
            figsize=(28, 11 * num_img_per_page),
        )

        for i in tqdm(range(num_explicands)):
            img_result_path = os.path.join(result_path, f"img_{i}")
            for j, aug_name in enumerate(aug_name_list):
                true_pred_prob = true_pred_prob_list[j][i]
                max_pred_prob = max_pred_prob_list[j][i]
                pred_label = synset_mapping[classes[pred_list[j][i]]]
                pred_info = f"Pred: {pred_label}"
                pred_info += f"\nPred prob: {max_pred_prob:.3f}"
                pred_info += f"\nLabel prob: {true_pred_prob:.3f}"

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

                aug_img_plot_idx = page_img_counter * 3

                # Raw images.
                axes[aug_img_plot_idx, j].imshow(aug_img.permute(1, 2, 0))
                axes[aug_img_plot_idx, j].set_xticks([])
                axes[aug_img_plot_idx, j].set_yticks([])
                axes[aug_img_plot_idx, j].set_title(pred_info, fontsize=24)

                if args.attribution_name in ["int_grad", "gradient_shape"]:
                    aug_attribution = torch.nn.functional.relu(
                        aug_attribution
                    )  # Focus on positive attributions.
                    vmin = -aug_attribution.max()
                    vmax = aug_attribution.max()
                    cmap = "seismic"
                else:
                    vmin = aug_attribution.min()
                    vmax = aug_attribution.max()
                    cmap = "jet"

                # Overlay attributions on raw images.
                axes[aug_img_plot_idx + 1, j].imshow(aug_img.permute(1, 2, 0))
                axes[aug_img_plot_idx + 1, j].set_xticks([])
                axes[aug_img_plot_idx + 1, j].set_yticks([])
                axes[aug_img_plot_idx + 1, j].imshow(
                    aug_attribution,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    alpha=0.8,
                )

                # Attributions only.
                axes[aug_img_plot_idx + 2, j].imshow(
                    aug_attribution,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                axes[aug_img_plot_idx + 2, j].set_xticks([])
                axes[aug_img_plot_idx + 2, j].set_yticks([])

            page_img_counter += 1
            if page_img_counter == num_img_per_page:
                page_img_counter = 0
                plt.tight_layout()
                pdf.savefig()
                plt.close(fig)
                fig, axes = plt.subplots(
                    ncols=len(aug_name_list),
                    nrows=3 * num_img_per_page,
                    figsize=(28, 11 * num_img_per_page),
                )
        if num_explicands % num_img_per_page:
            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    main()
