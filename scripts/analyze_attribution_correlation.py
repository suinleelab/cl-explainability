"""Analyze correlation between attributions from an original vs. randomized model."""

import argparse
import os
import pickle
import sys
from itertools import combinations

import constants
import numpy as np
import torch
from experiment_utils import get_device, get_output_filename, make_reproducible
from torchmetrics.functional import spearman_corrcoef
from tqdm import tqdm

RESULT_PATH = "/projects/leelab/cl-explainability/results"
ENCODER_DICT = {
    "imagenet": "simclr_x1",
    "cifar": "simsiam_18",
    "mura": "classifier_18",
}
ATTRIBUTION_NAME_LIST = ["int_grad", "gradient_shap", "rise", "random_baseline"]


def parse_analysis_args():
    """Parse command line arguments for this analysis script."""
    parser = argparse.ArgumentParser()
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
        "dataset_name",
        type=str,
        choices=["imagenet", "cifar", "mura"],
        help="name of dataset to analyze",
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
    args = parser.parse_args()
    print(f"Running {sys.argv[0]} with arguments")
    for arg in vars(args):
        print(f"\t{arg}={getattr(args, arg)}")
    return args


def get_result_path(
    parent_result_path: str,
    dataset_name: str,
    encoder_name: str,
    normalize_similarity: bool,
    explanation_name: str,
    attribution_name: str,
    seed: int,
    randomize_model: bool = False,
) -> str:
    """Generate path for result location."""
    if normalize_similarity:
        method_name = "normalized_"
    else:
        method_name = "unnormalized_"
    if randomize_model:
        method_name = "randomized_model_" + method_name
    method_name += explanation_name + "_" + attribution_name
    return os.path.join(
        parent_result_path,
        dataset_name,
        encoder_name,
        method_name,
        f"{seed}",
    )


def average_spearman(
    pred_attributions: torch.Tensor,
    target_attributions: torch.Tensor,
) -> torch.Tensor:
    """Calculate average Spearman's correlation for a batch of image attributions."""
    spearman_values = []
    for i in range(pred_attributions.size(0)):
        spearman_values.append(
            spearman_corrcoef(
                preds=pred_attributions[i].mean(dim=0).flatten(),
                target=target_attributions[i].mean(dim=0).flatten(),
            )
        )
    return torch.stack(spearman_values).mean()


def main():
    """Main function."""
    make_reproducible(42)
    args = parse_analysis_args()
    encoder_name = ENCODER_DICT[args.dataset_name]
    device = get_device(args.use_gpu, args.gpu_num)

    analysis_outputs = {}
    attribution_combos = combinations(ATTRIBUTION_NAME_LIST, 2)

    print("Computing similarity between each pair of attribution methods...")
    for attribution_combo in tqdm(attribution_combos):
        attribution_combo = tuple(sorted(attribution_combo))
        if attribution_combo[0] == "random_baseline":
            attribution_combo = (attribution_combo[1], attribution_combo[0])
        analysis_outputs[attribution_combo] = {"spearman": []}
        for seed in constants.SEED_LIST:
            result_path_0 = get_result_path(
                RESULT_PATH,
                args.dataset_name,
                encoder_name,
                args.normalize_similarity,
                args.explanation_name,
                attribution_combo[0],
                seed,
                randomize_model=False,
            )
            output_filename_0 = get_output_filename(
                different_classes=args.different_classes,
                corpus_size=100,
                explanation_name=args.explanation_name,
                foil_size=1500,
                explicand_size=25,
                attribution_name=attribution_combo[0],
                superpixel_dim=1,
                removal="blurring",
                blur_strength=5.0,
            )
            with open(
                os.path.join(result_path_0, output_filename_0),
                "rb",
            ) as handle:
                model_outputs_0 = pickle.load(handle)
            targets = model_outputs_0.keys()

            if attribution_combo[1] == "random_baseline":
                model_outputs_1 = {
                    target: {
                        "attributions": torch.randn(
                            model_outputs_0[target]["attributions"].shape
                        )
                    }
                    for target in targets
                }
            else:
                result_path_1 = get_result_path(
                    RESULT_PATH,
                    args.dataset_name,
                    encoder_name,
                    args.normalize_similarity,
                    args.explanation_name,
                    attribution_combo[1],
                    seed,
                    randomize_model=False,
                )
                output_filename_1 = get_output_filename(
                    different_classes=args.different_classes,
                    corpus_size=100,
                    explanation_name=args.explanation_name,
                    foil_size=1500,
                    explicand_size=25,
                    attribution_name=attribution_combo[1],
                    superpixel_dim=1,
                    removal="blurring",
                    blur_strength=5.0,
                )
                with open(
                    os.path.join(result_path_1, output_filename_1),
                    "rb",
                ) as handle:
                    model_outputs_1 = pickle.load(handle)

            spearman_list = []
            for target in targets:
                attributions_0 = (
                    model_outputs_0[target]["attributions"].to(device).double()
                )
                attributions_1 = (
                    model_outputs_1[target]["attributions"].to(device).double()
                )
                spearman_list.append(
                    average_spearman(attributions_0, attributions_1).cpu().numpy()
                )
            analysis_outputs[attribution_combo]["spearman"].append(
                np.mean(spearman_list)
            )

    print("Saving analysis outputs...")
    analysis_output_dir = "normalized" if args.normalize_similarity else "unnormalized"
    analysis_output_dir += f"_{args.explanation_name}"
    analysis_output_dir = os.path.join(
        constants.RESULT_PATH,
        "attribution_correlation",
        f"{args.dataset_name}_{encoder_name}",
        analysis_output_dir,
    )
    os.makedirs(analysis_output_dir, exist_ok=True)
    output_filename = "diff_class" if args.different_classes else "same_class"
    output_filename += "_outputs.pkl"
    with open(os.path.join(analysis_output_dir, output_filename), "wb") as handle:
        pickle.dump(analysis_outputs, handle)
    print("Done!")


if __name__ == "__main__":
    main()
