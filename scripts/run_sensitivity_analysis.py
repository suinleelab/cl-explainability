"""Run experiments with varying corpus or foil size."""
import argparse
import os
import sys

explanations = [
    "self_weighted",
    "contrastive_self_weighted",
    "corpus",
    "contrastive_corpus",
]

corpus_size_list = [5, 20, 50, 100, 200]
foil_size_list = [100, 500, 1500, 2500, 5000]


def main():
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "encoder_name",
        type=str,
        choices=["simclr_x1", "simsiam_18", "classifier_18"],
        help="name of pre-trained encoder to explain",
    )
    parser.add_argument(
        "dataset_name",
        type=str,
        choices=["imagenet", "cifar", "mura"],
        help="name of dataset to use",
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
        "parameter",
        type=str,
        choices=["corpus_size", "foil_size"],
        help="parameter to study",
    )
    parser.add_argument(
        "device",
        type=int,
        help="index of GPU to use",
    )
    parser.add_argument(
        "--normalize-similarity",
        action="store_true",
        help="flag to normalize dot product similarity to use cosine similarity",
        dest="normalize_similarity",
    )
    args = parser.parse_args()
    print(f"Running {sys.argv[0]} with arguments")
    for arg in vars(args):
        print(f"\t{arg}={getattr(args, arg)}")

    if args.attribution_name in ["int_grad", "smooth_int_grad", "gradient_shap"]:
        batch_size = 1
    elif args.attribution_name in ["smooth_vanilla_grad"]:
        batch_size = 4
    else:
        batch_size = 32

    if args.attribution_name == "kernel_shap" and args.dataset_name != "cifar":
        superpixel_dim = 8
        eval_superpixel_dim = 8
    else:
        superpixel_dim = 1
        eval_superpixel_dim = 1

    if args.parameter == "corpus_size":
        parameter_val_list = corpus_size_list
    elif args.parameter == "foil_size":
        parameter_val_list = foil_size_list
    else:
        raise ValueError(
            f"parameter={args.parameter} should be one of [corpus_size, foil_size]!"
        )
    parameter_arg = args.parameter.replace("_", "-")
    for parameter_val in parameter_val_list:
        for different_classes in [False, True]:
            for explanation in explanations:
                command_args = args.encoder_name
                command_args += f" {explanation}"
                command_args += f" {args.attribution_name}"
                if args.normalize_similarity:
                    command_args += " --normalize-similarity"
                if different_classes:
                    command_args += " --different-classes"

                command_args += f" --{parameter_arg} {parameter_val}"
                command_args += f" --dataset-name {args.dataset_name}"
                command_args += f" --batch-size {batch_size}"
                command_args += f" --use-gpu --gpu-num {args.device}"
                command_args += f" --superpixel-dim {superpixel_dim}"
                command_args += f" --eval-superpixel-dim {eval_superpixel_dim}"
                os.system("python scripts/run.py " + command_args)


if __name__ == "__main__":
    main()
