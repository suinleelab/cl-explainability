"""Analyze similarity between attributions from an original vs. randomized model."""

import argparse
import os
import pickle
import sys

import constants
import numpy as np
import torch
from experiment_utils import get_device, get_output_filename
from scipy.stats import pearsonr
from skimage.feature import hog
from skimage.metrics import structural_similarity
from torchmetrics.functional import spearman_corrcoef
from tqdm import tqdm

RANDOMIZED_RESULT_PATH = "/homes/gws/clin25/cl-explainability/results"
ORIGINAL_RESULT_PATH = "/projects/leelab/cl-explainability/results"
DATASET_ENCODER_NAME_LIST = [
    ("imagenet", "simclr_x1"),
    ("cifar", "simsiam_18"),
    ("mura", "classifier_18"),
]


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


def preprocess_randomized_attributions(
    randomized_attributions: torch.Tensor,
) -> torch.Tensor:
    """Preprocess randomized attributions."""
    for i in range(randomized_attributions.size(0)):
        isnan = torch.isnan(randomized_attributions[i])
        nan_count = isnan.sum()
        if nan_count == isnan.nelement:  # If all nan's.
            randomized_attributions[i] = 0
        elif nan_count > 0:  # If some nan's.
            if randomized_attributions[i].nansum() == 0:
                randomized_attributions[i][isnan] = 0
            else:
                raise NotImplementedError("Non-zero nan's are not handled!")
    return randomized_attributions


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


def average_hog_pearson(
    pred_attributions: np.ndarray,
    target_attributions: np.ndarray,
) -> np.ndarray:
    """Calculate average HOG Pearson's correlation for a batch of image attributions."""
    hog_pearson_values = []
    for i in range(pred_attributions.shape[0]):
        hog_pearson = pearsonr(
            hog(pred_attributions[i], channel_axis=0),
            hog(target_attributions[i], channel_axis=0),
        )[0]
        if np.isnan(hog_pearson):
            hog_pearson = 0.0
        hog_pearson_values.append(hog_pearson)
    return np.mean(hog_pearson_values)


def average_ssim(
    pred_attributions: np.ndarray,
    target_attributions: np.ndarray,
) -> np.ndarray:
    """Calculate SSIM for a batch of image attributions."""
    ssim_values = []
    for i in range(pred_attributions.shape[0]):
        ssim_values.append(
            structural_similarity(
                pred_attributions[i],
                target_attributions[i],
                channel_axis=0,
            )
        )
    return np.mean(ssim_values)


def main():
    """Main function."""
    args = parse_analysis_args()
    device = get_device(args.use_gpu, args.gpu_num)
    output_filename = get_output_filename(
        different_classes=args.different_classes,
        corpus_size=100,
        explanation_name=args.explanation_name,
        foil_size=1500,
        explicand_size=25,
        attribution_name=args.attribution_name,
        superpixel_dim=1,
        removal="blurring",
        blur_strength=5.0,
    )
    analysis_outputs = {}
    print(
        "Computing similarities between randomized and original attributions"
        "for each pair of dataset and model..."
    )
    for dataset_encoder_name in tqdm(DATASET_ENCODER_NAME_LIST):
        dataset_name, encoder_name = dataset_encoder_name
        analysis_outputs[dataset_encoder_name] = {
            "spearman": [],
            "ssim": [],
            "hog_pearson": [],
        }

        for seed in constants.SEED_LIST:
            randomized_model_result_path = get_result_path(
                RANDOMIZED_RESULT_PATH,
                dataset_name,
                encoder_name,
                args.normalize_similarity,
                args.explanation_name,
                args.attribution_name,
                seed,
                randomize_model=True,
            )
            original_model_result_path = get_result_path(
                ORIGINAL_RESULT_PATH,
                dataset_name,
                encoder_name,
                args.normalize_similarity,
                args.explanation_name,
                args.attribution_name,
                seed,
                randomize_model=False,
            )
            with open(
                os.path.join(randomized_model_result_path, output_filename),
                "rb",
            ) as handle:
                randomized_model_outputs = pickle.load(handle)
            with open(
                os.path.join(original_model_result_path, output_filename),
                "rb",
            ) as handle:
                original_model_outputs = pickle.load(handle)

            targets = original_model_outputs.keys()
            spearman_list = []
            ssim_list = []
            hog_pearson_list = []
            for target in targets:
                randomized_attributions = randomized_model_outputs[target][
                    "attributions"
                ].to(device)
                randomized_attributions = preprocess_randomized_attributions(
                    randomized_attributions
                )
                original_attributions = original_model_outputs[target][
                    "attributions"
                ].to(device)

                assert not torch.any(torch.isnan(randomized_attributions))
                assert not torch.any(torch.isnan(original_attributions))
                assert randomized_attributions.size(0) == original_attributions.size(0)

                spearman_list.append(
                    average_spearman(randomized_attributions, original_attributions)
                    .cpu()
                    .numpy()
                )

                randomized_attributions = randomized_attributions.cpu().numpy()
                original_attributions = original_attributions.cpu().numpy()

                ssim_list.append(
                    average_ssim(randomized_attributions, original_attributions)
                )
                hog_pearson_list.append(
                    average_hog_pearson(randomized_attributions, original_attributions)
                )

            analysis_outputs[dataset_encoder_name]["spearman"].append(
                np.mean(spearman_list)
            )
            analysis_outputs[dataset_encoder_name]["ssim"].append(np.mean(ssim_list))
            analysis_outputs[dataset_encoder_name]["hog_pearson"].append(
                np.mean(hog_pearson_list)
            )

    print("Saving analysis outputs...")

    analysis_output_dir = "normalized" if args.normalize_similarity else "unnormalized"
    analysis_output_dir += f"_{args.explanation_name}_{args.attribution_name}"
    analysis_output_dir = os.path.join(
        constants.RESULT_PATH,
        "model_randomization",
        analysis_output_dir,
    )
    os.makedirs(analysis_output_dir, exist_ok=True)
    with open(os.path.join(analysis_output_dir, output_filename), "wb") as handle:
        pickle.dump(analysis_outputs, handle)
    print("Done!")


if __name__ == "__main__":
    main()
