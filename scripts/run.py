"""Run explanation methods for encoder representations."""

import argparse
import os
import pickle
import sys
from functools import partial

import constants
import torch
import torchvision.transforms as transforms
from captum.attr import IntegratedGradients, KernelShap, Saliency
from experiment_utils import get_device, load_data, load_encoder, make_reproducible
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from cl_explain.attributions.random_baseline import RandomBaseline
from cl_explain.explanations.corpus_similarity import CorpusSimilarity
from cl_explain.utils import make_superpixel_map


def parse_args():
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
        help="superpixel width and height for attributions",
        dest="superpixel_dim",
    )
    parser.add_argument(
        "--blur-strength",
        type=float,
        default=5,
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


def main():
    """Main function."""
    args = parse_args()
    make_reproducible(args.seed)
    device = get_device(args.use_gpu, args.gpu_num)
    print("Loading encoder...")
    encoder = load_encoder(args.encoder_name)
    encoder.eval()
    encoder.to(device)
    print("Loading dataset...")
    dataset, dataloader, class_map = load_data(args.dataset_name, args.batch_size)
    if args.dataset_name == "imagenette2":
        img_w = 224
        img_h = 224
        removal = "blurring"
        get_baseline = transforms.GaussianBlur(21, sigma=args.blur_strength).to(device)
    else:
        raise NotImplementedError(
            f"--dataset-name={args.dataset_name} is not implemented!"
        )

    labels = []
    for _, label in dataloader:
        labels.append(label)
    labels = torch.cat(labels)
    unique_labels = labels.unique().numpy()
    outputs = {target: {"source_label": class_map[target]} for target in unique_labels}

    print("Computing feature attributions for each class...")
    for target in tqdm(unique_labels):
        idx = (labels == target).nonzero().flatten()
        idx = idx[torch.randperm(idx.size(0))]

        explicand_idx = idx[: args.explicand_size]
        corpus_idx = idx[args.explicand_size : (args.explicand_size + args.corpus_size)]
        outputs[target]["explicand_idx"] = explicand_idx
        outputs[target]["corpus_idx"] = corpus_idx

        explicand_dataloader = DataLoader(
            Subset(dataset, indices=explicand_idx),
            batch_size=args.batch_size,
            shuffle=False,
        )
        corpus_dataloader = DataLoader(
            Subset(dataset, indices=corpus_idx),
            batch_size=args.batch_size,
            shuffle=False,
        )

        explanation_model = CorpusSimilarity(
            encoder, corpus_dataloader, corpus_batch_size=args.batch_size
        )

        if args.attribution_name == "vanilla_grad":
            attribution_model = Saliency(explanation_model)
            attribute = partial(attribution_model.attribute, abs=False)
            use_baseline = False
        elif args.attribution_name == "int_grad":
            attribution_model = IntegratedGradients(explanation_model)
            attribute = partial(attribution_model.attribute)
            use_baseline = True
        elif args.attribution_name == "kernel_shap":
            feature_mask = make_superpixel_map(
                img_h, img_w, args.superpixel_dim, args.superpixel_dim
            )
            feature_mask = feature_mask.to(device)
            attribution_model = KernelShap(explanation_model)
            attribute = partial(
                attribution_model.attribute, n_samples=10000, feature_mask=feature_mask
            )
            use_baseline = True
        elif args.attribution_name == "random_baseline":
            attribution_model = RandomBaseline(explanation_model)
            attribute = partial(attribution_model.attribute)
            use_baseline = False
        else:
            raise NotImplementedError(
                f"{args.attribution_name} attribution is not implemented!"
            )

        attribution_list = []
        for explicand, _ in explicand_dataloader:
            explicand = explicand.to(device)
            baseline = get_baseline(explicand)
            explicand.requires_grad = True
            if use_baseline:
                attribution = attribute(explicand, baselines=baseline)
            else:
                attribution = attribute(explicand)
            attribution_list.append(attribution.detach().cpu())
        outputs[target]["attributions"] = torch.cat(attribution_list)

    print("Saving outputs...")
    result_path = os.path.join(
        constants.RESULT_PATH,
        args.dataset_name,
        args.encoder_name,
        args.attribution_name,
        f"{args.seed}",
    )
    os.makedirs(result_path, exist_ok=True)
    output_filename = "outputs"
    output_filename += f"_corpus_size={args.corpus_size}"
    output_filename += f"_explicand_size={args.explicand_size}"
    if args.attribution_name in constants.SUPERPIXEL_ATTRIBUTION_METHODS:
        output_filename += f"_superpixel_dim={args.superpixel_dim}"
    output_filename += f"_removal={removal}"
    if removal == "blurring":
        output_filename += f"_blur_strength={args.blur_strength}"
    output_filename += ".pkl"
    with open(os.path.join(result_path, output_filename), "wb") as handle:
        pickle.dump(outputs, handle)
    print("Done!")


if __name__ == "__main__":
    main()
