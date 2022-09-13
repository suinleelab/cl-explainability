"""Run feature attribution experiments across multiple explanation targets."""
import argparse
import os
import sys

attributions = [
    "int_grad",
    "gradient_shap",
    "rise",
]

# Experiments defined by tuples (target_name, explicand, foil, corpus)
experiments = [
    ("self_weighted", "astronaut", "none", "none"),
    ("contrastive_corpus", "astronaut", "woman", "cifar100"),
    ("contrastive_corpus", "astronaut", "rocket", "cifar100"),
    ("contrastive_corpus", "astronaut", "astronaut", "cifar100"),
    ("contrastive_corpus", "astronaut", "orange", "cifar100"),
    ("contrastive_corpus", "astronaut", "woman", "astronaut"),
    ("contrastive_corpus", "astronaut", "woman", "rocket"),
    ("contrastive_corpus", "astronaut", "woman", "face"),
    ("contrastive_corpus", "astronaut", "woman", "hair"),
    ("self_weighted", "camera", "none", "none"),
    ("contrastive_corpus", "camera", "man", "cifar100"),
    ("contrastive_corpus", "camera", "camera", "cifar100"),
]


def main():
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "device",
        type=int,
        help="index of GPU to use",
    )
    args = parser.parse_args()
    print(f"Running {sys.argv[0]} with arguments")
    for arg in vars(args):
        print(f"\t{arg}={getattr(args, arg)}")

    for experiment in experiments:
        explanation, explicand, foil, corpus = experiment
        for attribution in attributions:
            command_args = f" {explanation}"
            command_args += f" {attribution}"
            command_args += f" {explicand}"
            command_args += f" {foil}"
            command_args += f" {corpus}"
            command_args += f" --use-gpu --gpu-num {args.device}"
            os.system("python scripts/clip_use_case.py " + command_args)


if __name__ == "__main__":
    main()
