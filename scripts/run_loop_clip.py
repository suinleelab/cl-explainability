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
    ("contrastive_corpus", "camera", "suit", "cifar100"),
    ("self_weighted", "dog_cat", "none", "none"),
    ("contrastive_corpus", "dog_cat", "dog", "cifar100"),
    ("contrastive_corpus", "dog_cat", "cat", "cifar100"),
    ("contrastive_corpus", "dog_cat", "dog", "cat"),
    ("contrastive_corpus", "dog_cat", "cat", "dog"),
    ("self_weighted", "dogs", "none", "none"),
    ("contrastive_corpus", "dogs", "dog", "cifar100"),
    ("contrastive_corpus", "dogs", "truck", "cifar100"),
    ("contrastive_corpus", "dogs", "dog", "truck"),
    ("contrastive_corpus", "dogs", "truck", "dog"),
    ("self_weighted", "zebra", "none", "none"),
    ("contrastive_corpus", "zebra", "zebra", "cifar100"),
    ("contrastive_corpus", "zebra", "stripes", "cifar100"),
    ("contrastive_corpus", "zebra", "horse", "cifar100"),
    ("contrastive_corpus", "zebra", "zebra", "stripes"),
    ("contrastive_corpus", "zebra", "stripes", "zebra"),
    ("contrastive_corpus", "zebra", "zebra", "horse"),
    ("self_weighted", "bee", "none", "none"),
    ("contrastive_corpus", "bee", "bee", "cifar100"),
    ("contrastive_corpus", "bee", "flower", "cifar100"),
    ("contrastive_corpus", "bee", "bee", "flower"),
    ("contrastive_corpus", "bee", "flower", "bee"),
    ("self_weighted", "mountain", "none", "none"),
    ("contrastive_corpus", "mountain", "mountain", "cifar100"),
    ("contrastive_corpus", "mountain", "cloud", "cifar100"),
    ("contrastive_corpus", "mountain", "mountain", "cloud"),
    ("contrastive_corpus", "mountain", "cloud", "mountain"),
    ("self_weighted", "boy_and_girl", "none", "none"),
    ("contrastive_corpus", "boy_and_girl", "boy", "cifar100"),
    ("contrastive_corpus", "boy_and_girl", "girl", "cifar100"),
    ("contrastive_corpus", "boy_and_girl", "boy", "girl"),
    ("contrastive_corpus", "boy_and_girl", "girl", "boy"),
    ("self_weighted", "boy_and_girl1", "none", "none"),
    ("contrastive_corpus", "boy_and_girl1", "boy", "cifar100"),
    ("contrastive_corpus", "boy_and_girl1", "girl", "cifar100"),
    ("contrastive_corpus", "boy_and_girl1", "boy", "girl"),
    ("contrastive_corpus", "boy_and_girl1", "girl", "boy"),
    ("self_weighted", "boy_and_girl2", "none", "none"),
    ("contrastive_corpus", "boy_and_girl2", "boy", "cifar100"),
    ("contrastive_corpus", "boy_and_girl2", "girl", "cifar100"),
    ("contrastive_corpus", "boy_and_girl2", "boy", "girl"),
    ("contrastive_corpus", "boy_and_girl2", "girl", "boy"),
    ("self_weighted", "bicycle_and_car", "none", "none"),
    ("contrastive_corpus", "bicycle_and_car", "bicycle", "cifar100"),
    ("contrastive_corpus", "bicycle_and_car", "car", "cifar100"),
    ("contrastive_corpus", "bicycle_and_car", "bicycle", "car"),
    ("contrastive_corpus", "bicycle_and_car", "car", "bicycle"),
    ("self_weighted", "bicycle_and_car1", "none", "none"),
    ("contrastive_corpus", "bicycle_and_car1", "bicycle", "cifar100"),
    ("contrastive_corpus", "bicycle_and_car1", "car", "cifar100"),
    ("contrastive_corpus", "bicycle_and_car1", "bicycle", "car"),
    ("contrastive_corpus", "bicycle_and_car1", "car", "bicycle"),
    ("self_weighted", "bicycle_and_car2", "none", "none"),
    ("contrastive_corpus", "bicycle_and_car2", "bicycle", "cifar100"),
    ("contrastive_corpus", "bicycle_and_car2", "car", "cifar100"),
    ("contrastive_corpus", "bicycle_and_car2", "bicycle", "car"),
    ("contrastive_corpus", "bicycle_and_car2", "car", "bicycle"),
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
