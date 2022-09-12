"""Contrastive corpus attribution for understanding CLIP."""
import argparse
import os
import sys

import clip
import constants
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import skimage
import torch
import torchvision.transforms as transforms
from captum.attr import GradientShap, IntegratedGradients
from experiment_utils import get_device
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR100

from cl_explain.attributions.rise import RISE
from cl_explain.explanations.contrastive_corpus_similarity import (
    ContrastiveCorpusSimilarity,
)
from cl_explain.explanations.weighted_score import WeightedScore


def parse_augmentation_use_case_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target_name",
        type=str,
        choices=["self_weighted", "contrastive_corpus"],
        help="name of the explanation target (model behavior)",
    )
    parser.add_argument(
        "attribution_name",
        type=str,
        choices=["int_grad", "gradient_shap", "rise"],
        help="name of feature attribution method to use",
    )
    parser.add_argument(
        "explicand_name",
        type=str,
        choices=["astronaut", "camera"],
        help="name of explicand image",
    )
    parser.add_argument(
        "corpus_name",
        type=str,
        help="name of corpus text to use",
    )
    parser.add_argument(
        "foil_name",
        type=str,
        help="name of foil text to use",
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
    args = parser.parse_args()
    print(f"Running {sys.argv[0]} with arguments")
    for arg in vars(args):
        print(f"\t{arg}={getattr(args, arg)}")
    return args


def plot_results(explicand_raw, attribution, is_zero_center, fname):
    """
    Visualize explicand and attributions.

    Args:
    ----
        explicand_raw: An unprocessed explicand image.
        attribution: An attribution tensor with the same shape as the explicand.
        is_zero_center: A boolean describing whether to center the color map.
        fname: A filename where the figure will be saved.
    """
    flat_attribution = attribution.cpu()[0].mean(0)
    n_col, n_row = 2, 1
    plt.figure(figsize=(4 * n_col, 4 * n_row))

    # Plot explicand
    plt.subplot(n_row, n_col, 1)
    plt.imshow(explicand_raw)
    plt.xticks([])
    plt.yticks([])

    # Plot label-free attributions
    plt.subplot(n_row, n_col, 2)
    if is_zero_center:
        m = flat_attribution.abs().max()
        plt.imshow(flat_attribution, vmin=-m, vmax=m, cmap="seismic")
    else:
        m1, m2 = flat_attribution.min(), flat_attribution.max()
        plt.imshow(flat_attribution, vmin=m1, vmax=m2, cmap="seismic")
    plt.xticks([])
    plt.yticks([])

    plt.savefig(fname)


def main():
    """Main function."""
    args = parse_augmentation_use_case_args()
    pl.seed_everything(args.seed, workers=True)
    device = get_device(args.use_gpu, args.gpu_num)

    # Set up result path
    result_path = os.path.join(constants.RESULT_PATH, "clip_use_case", f"{args.seed}")
    os.makedirs(result_path, exist_ok=True)
    explicand_result_path = os.path.join(
        result_path, f"explicand_{args.explicand_name}"
    )
    os.makedirs(explicand_result_path, exist_ok=True)
    fname = f"{args.corpus_name}_vs_{args.foil_name}_"
    fname += f"{args.target_name}_{args.attribution_name}"
    result_fname = os.path.join(explicand_result_path, fname + ".pt")
    fig_fname = os.path.join(explicand_result_path, fname + ".pdf")

    print("Loading encoder...")
    encoder, preprocess = clip.load("ViT-B/32")
    encoder.cuda(device).eval()

    print("Loading explicand and baseline...")
    if args.explicand_name == "astronaut":
        explicand_fname = "astronaut.png"
    elif args.explicand_name == "camera":
        explicand_fname = "camera.png"
    else:
        raise NotImplementedError(
            f"{args.explicand_name} explicand_name is not implemented!"
        )
    explicand_raw = Image.open(os.path.join(skimage.data_dir, explicand_fname)).convert(
        "RGB"
    )
    explicand = torch.unsqueeze(preprocess(explicand_raw).cuda(device), 0)
    get_baseline = transforms.GaussianBlur(21, sigma=args.blur_strength).to(device)
    baseline = get_baseline(explicand)

    # Skip if attributions exist
    if os.path.exists(result_fname):
        print("Loading attribution...")
        attribution = torch.load(result_fname, map_location="cpu")
    else:
        print("Setting up explanation target...")
        if args.target_name == "self_weighted":
            # Create the explanation target
            explanation_target = WeightedScore(encoder.encode_image, normalize=True)
            explanation_target.generate_weight(explicand)

        elif args.target_name == "contrastive_corpus":
            # Set up corpus and foil
            corpus_tokens = [f"This is a photo of a {args.corpus_name}"]
            if args.foil_name == "cifar100":
                cifar100 = CIFAR100(
                    os.path.expanduser("~/.cache"), transform=preprocess, download=True
                )
                foil_tokens = []
                for label in cifar100.classes:
                    if label != args.corpus_name:
                        foil_tokens.append(f"This is a photo of a {label}")
            else:
                foil_tokens = [f"This is a photo of a {args.foil_name}"]

            # Convert to dataloaders
            corpus_tokens = clip.tokenize(corpus_tokens).cuda(device)
            corpus_dataloader = DataLoader(
                TensorDataset(corpus_tokens, torch.ones(corpus_tokens.shape[0]))
            )
            foil_tokens = clip.tokenize(foil_tokens).cuda(device)
            foil_dataloader = DataLoader(
                TensorDataset(foil_tokens, torch.ones(foil_tokens.shape[0]))
            )

            # Create the explanation target
            explanation_target = ContrastiveCorpusSimilarity(
                encoder.encode_text,
                corpus_dataloader,
                foil_dataloader,
                normalize=True,
                explicand_encoder=encoder.encode_image,
                device=device,
            )

        else:
            raise NotImplementedError(
                f"{args.target_name} target_name is not implemented!"
            )

        print("Running and saving attributions...")
        if args.attribution_name == "int_grad":
            attribution_model = IntegratedGradients(explanation_target)
            attribution = attribution_model.attribute(explicand, baselines=baseline)
        elif args.attribution_name == "gradient_shap":
            attribution_model = GradientShap(explanation_target)
            attribution = attribution_model.attribute(
                explicand,
                baselines=baseline,
                n_samples=50,
                stdevs=0.2,
            )
        elif args.attribution_name == "rise":
            attribution_model = RISE(explanation_target)
            attribution = attribution_model.attribute(
                explicand,
                grid_shape=(7, 7),
                baselines=baseline,
                mask_prob=0.5,
                n_samples=20000,  # Higher n_samples
                normalize_by_mask_prob=True,
            )
        else:
            raise NotImplementedError(
                f"{args.attribution_name} attribution is not implemented!"
            )

        attribution = attribution.detach().cpu()
        torch.save(attribution, result_fname)

    print("Plotting results...")
    is_zero_center = True
    if args.attribution_name in ["rise"]:
        is_zero_center = False
    plot_results(explicand_raw, attribution, is_zero_center, fig_fname)

    print("Done!")


if __name__ == "__main__":
    main()
