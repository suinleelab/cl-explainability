"""Evaluate explanation methods for encoder representations."""

import argparse
import os
import pickle
import sys
from functools import partial
from typing import List, Tuple

import constants
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from captum.attr import IntegratedGradients, Saliency
from eval_utils import get_device, make_reproducible
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from cl_explain.encoders.simclr.resnet_wider import resnet50x1, resnet50x2, resnet50x4
from cl_explain.explanations.corpus_similarity import CorpusSimilarity
from cl_explain.metrics.ablation import ImageAblation


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
        choices=["vanilla_grad", "int_grad", "random_baseline"],
        help="name of feature attribution method to use",
    )
    parser.add_argument(
        "--take-attribution-abs",
        action="store_true",
        help="flag to take absolute value for attribution scores",
        dest="take_attribution_abs",
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


def _encoder_not_implemented_error(encoder_name: str) -> None:
    raise NotImplementedError(f"{encoder_name} loading is not implemented!")


def load_encoder(encoder_name: str) -> nn.Module:
    """Load encoder."""
    encoder = None
    if "simclr" in encoder_name:
        state_dict_path = os.path.join(constants.ENCODER_PATH, "simclr")
        if encoder_name == "simclr_x1":
            encoder = resnet50x1()
            state_dict_path = os.path.join(state_dict_path, "resnet50-1x.pth")
        elif encoder_name == "simclr_x2":
            encoder = resnet50x2()
            state_dict_path = os.path.join(state_dict_path, "resnet50-2x.pth")
        elif encoder_name == "simclr_x4":
            encoder = resnet50x4()
            state_dict_path = os.path.join(state_dict_path, "resnet50-4x.pth")
        else:
            _encoder_not_implemented_error(encoder_name)
        state_dict = torch.load(state_dict_path, map_location="cpu")
        encoder.load_state_dict(state_dict["state_dict"])
    else:
        _encoder_not_implemented_error(encoder_name)
    return encoder


def load_data(
    dataset_name: str, batch_size: int
) -> Tuple[Dataset, DataLoader, List[str]]:
    """Load data."""
    if dataset_name == "imagenette2":
        dataset_path = os.path.join(constants.DATA_PATH, dataset_name, "val")
        dataset = torchvision.datasets.ImageFolder(
            dataset_path,
            transform=transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ]
            ),
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        class_map = dataset.find_classes(dataset_path)[0]
    else:
        raise NotImplementedError(f"{dataset_name} loading is not implemented!")
    return dataset, dataloader, class_map


def main():
    """Main function."""
    args = parse_args()
    make_reproducible(args.seed)
    device = get_device(args.use_gpu, args.gpu_num)
    encoder = load_encoder(args.encoder_name)
    encoder.eval()
    encoder.to(device)
    dataset, dataloader, class_map = load_data(args.dataset_name, args.batch_size)
    if args.dataset_name == "imagenette2":
        feature_attr_size = 1 * 224 * 224
        baseline = torch.zeros(1, 3, 224, 224).to(device)
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
        image_ablation = ImageAblation(explanation_model, feature_attr_size)

        if args.attribution_name == "vanilla_grad":
            attribution_model = Saliency(explanation_model)
            attribute = partial(attribution_model.attribute, abs=False)
        elif args.attribution_name == "int_grad":
            attribution_model = IntegratedGradients(explanation_model)
            attribute = partial(attribution_model.attribute)
        else:
            raise NotImplementedError(
                f"{args.attribution_name} attribution is not implemented!"
            )

        attribution_list = []
        insertion_curve_list = []
        deletion_curve_list = []
        insertion_num_features = None
        deletion_num_features = None

        for explicand, _ in explicand_dataloader:
            explicand = explicand.to(device)
            explicand.requires_grad = True
            attribution = attribute(explicand)
            if args.take_attribution_abs:
                attribution = attribution.abs()
            attribution = attribution.sum(dim=1).unsqueeze(1)
            insertion_curve, insertion_num_features = image_ablation.evaluate(
                explicand,
                attribution,
                baseline,
                kind="insertion",
            )
            deletion_curve, deletion_num_features = image_ablation.evaluate(
                explicand,
                attribution,
                baseline,
                kind="deletion",
            )
            attribution_list.append(attribution.detach().cpu())
            insertion_curve_list.append(insertion_curve.detach().cpu())
            deletion_curve_list.append(deletion_curve.detach().cpu())

        outputs[target]["attributions"] = torch.cat(attribution_list)
        outputs[target]["insertion_curves"] = torch.cat(insertion_curve_list)
        outputs[target]["deletion_curves"] = torch.cat(deletion_curve_list)
        outputs[target]["insertion_num_features"] = insertion_num_features
        outputs[target]["deletion_num_features"] = deletion_num_features

    # Save all outputs.
    attribution_name = args.attribution_name
    if args.take_attribution_abs:
        attribution_name += "_abs"
    result_path = os.path.join(
        constants.RESULT_PATH,
        args.dataset_name,
        args.encoder_name,
        attribution_name,
        f"{args.seed}",
    )
    os.makedirs(result_path, exist_ok=True)
    output_filename = "outputs"
    output_filename += f"_corpus_size={args.corpus_size}"
    output_filename += f"_explicand_size={args.explicand_size}"
    output_filename += ".pkl"
    with open(os.path.join(result_path, output_filename), "wb") as handle:
        pickle.dump(outputs, handle)


if __name__ == "__main__":
    main()
