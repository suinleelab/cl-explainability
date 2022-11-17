"""Run feature attribution experiments across multiple explanation targets."""
import argparse
import os
import sys

explanations = [
    "contrastive_corpus",
    "self_weighted",
    "contrastive_self_weighted",
    "corpus",
]

attribution_choices = [
    "gradient_shap",
    "int_grad",
    "rise",
]


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
        choices=attribution_choices + ["all"],
        help="name of feature attribution method to use",
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
    parser.add_argument(
        "--randomize-model",
        action="store_true",
        help="flag to enable randomization of model parameters",
        dest="randomize_model",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="attribute_and_eval",
        choices=["attribute_only", "eval_only", "attribute_and_eval"],
        help="for meta script, whether to run attribute.py and or eval.py",
        dest="mode",
    )
    args = parser.parse_args()
    print(f"Running {sys.argv[0]} with arguments")
    for arg in vars(args):
        print(f"\t{arg}={getattr(args, arg)}")

    if args.attribution_name == "kernel_shap" and args.dataset_name != "cifar":
        superpixel_dim = 8
        eval_superpixel_dim = 8
    else:
        superpixel_dim = 1
        eval_superpixel_dim = 1

    # Run through all attributions or a specific one
    if args.attribution_name == "all":
        attributions = attribution_choices
    else:
        attributions = [args.attribution_name]

    for attribution in attributions:
        if attribution in ["int_grad", "smooth_int_grad", "gradient_shap"]:
            batch_size = 1
        elif attribution in ["smooth_vanilla_grad"]:
            batch_size = 4
        else:
            batch_size = 32

        for explanation in explanations:
            command_args = args.encoder_name
            command_args += f" {explanation}"
            command_args += f" {attribution}"
            if args.normalize_similarity:
                command_args += " --normalize-similarity"
            if args.randomize_model:
                command_args += " --randomize-model"
            command_args += f" --dataset-name {args.dataset_name}"
            command_args += f" --batch-size {batch_size}"
            command_args += f" --use-gpu --gpu-num {args.device}"
            command_args += f" --superpixel-dim {superpixel_dim}"
            command_args += f" --eval-superpixel-dim {eval_superpixel_dim}"
            command_args += f" --mode {args.mode}"
            command_args += " --comprehensive"
            os.system("python scripts/run.py " + command_args)


if __name__ == "__main__":
    main()
