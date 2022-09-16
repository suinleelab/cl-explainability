"""Run experiments with varying corpus or foil size."""
import argparse
import os
import sys

attribution_choices = ["int_grad", "rise", "gradient_shap"]

corpus_explanation_list = ["corpus", "contrastive_corpus"]
corpus_size_list = [5, 20, 50, 100, 200]

foil_explanation_list = ["contrastive_self_weighted", "contrastive_corpus"]
foil_size_list = [100, 500, 1500, 2500, 5000]

dataset_map = {
    "simclr_x1": "imagenet",
    "simsiam_18": "cifar",
    "classifier_18": "mura",
}


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
        "parameter",
        type=str,
        choices=["corpus_size", "foil_size"],
        help="parameter to study",
    )
    parser.add_argument(
        "--attribution-name",
        type=str,
        default="all",
        choices=attribution_choices + ["all"],
        help="feature attribution method to run",
    )
    parser.add_argument(
        "--gpu-num",
        type=int,
        help="index of the GPU to use",
        dest="gpu_num",
    )
    parser.add_argument(
        "--normalize-similarity",
        action="store_true",
        help="flag to normalize dot product similarity to use cosine similarity",
        dest="normalize_similarity",
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

    dataset_name = dataset_map[args.encoder_name]

    if args.parameter == "corpus_size":
        explanation_list = corpus_explanation_list
        parameter_val_list = corpus_size_list
    elif args.parameter == "foil_size":
        explanation_list = foil_explanation_list
        parameter_val_list = foil_size_list
    else:
        raise ValueError(
            f"parameter={args.parameter} should be one of [corpus_size, foil_size]!"
        )
    parameter_arg = args.parameter.replace("_", "-")

    if args.attribution_name == "all":
        attribution_list = attribution_choices
    else:
        attribution_list = [args.attribution_name]

    for attribution in attribution_list:
        if attribution in ["int_grad", "gradient_shap"]:
            batch_size = 1
        else:
            batch_size = 32
        for parameter_val in parameter_val_list:
            for explanation in explanation_list:
                command_args = args.encoder_name
                command_args += f" {explanation}"
                command_args += f" {attribution}"
                if args.normalize_similarity:
                    command_args += " --normalize-similarity"
                command_args += f" --{parameter_arg} {parameter_val}"
                command_args += f" --dataset-name {dataset_name}"
                command_args += f" --batch-size {batch_size}"
                command_args += f" --use-gpu --gpu-num {args.gpu_num}"
                command_args += f" --seed {args.seed}"
                os.system(f" python scripts/run.py {command_args} --one-seed")


if __name__ == "__main__":
    main()
