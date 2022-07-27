"""Run explanation methods for encoder representations."""

import os
import pickle
from functools import partial

import torch
import torchvision.transforms as transforms
from captum.attr import IntegratedGradients, KernelShap, Saliency
from experiment_utils import (
    get_device,
    get_image_dataset_meta,
    get_output_filename,
    get_result_path,
    load_data,
    load_encoder,
    make_reproducible,
    parse_args,
)
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from cl_explain.attributions.random_baseline import RandomBaseline
from cl_explain.explanations.corpus_similarity import CorpusSimilarity
from cl_explain.utils import make_superpixel_map


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
    img_h, img_w, removal = get_image_dataset_meta(args.dataset_name)
    if removal == "blurring":
        get_baseline = transforms.GaussianBlur(21, sigma=args.blur_strength).to(device)
    else:
        raise NotImplementedError(f"removal={removal} is not implemented!")

    labels = []
    for _, label in dataloader:
        labels.append(label)
    labels = torch.cat(labels)
    unique_labels = labels.unique().numpy()
    outputs = {target: {"source_label": class_map[target]} for target in unique_labels}

    for target in unique_labels:
        target_idx = (labels == target).nonzero().flatten()
        target_idx = target_idx[torch.randperm(target_idx.size(0))]
        outputs[target]["explicand_idx"] = target_idx[: args.explicand_size]
        outputs[target]["corpus_idx"] = target_idx[
            args.explicand_size : (args.explicand_size + args.corpus_size)
        ]

    print("Computing feature attributions for each class...")
    for target in tqdm(unique_labels):
        explicand_idx = outputs[target]["explicand_idx"]
        corpus_idx = outputs[target]["corpus_idx"]

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
            encoder, corpus_dataloader, batch_size=args.batch_size
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
    result_path = get_result_path(
        args.dataset_name, args.encoder_name, args.attribution_name, args.seed
    )
    os.makedirs(result_path, exist_ok=True)
    output_filename = get_output_filename(
        args.corpus_size,
        args.explicand_size,
        args.attribution_name,
        args.superpixel_dim,
        removal,
        args.blur_strength,
    )
    with open(os.path.join(result_path, output_filename), "wb") as handle:
        pickle.dump(outputs, handle)
    print("Done!")


if __name__ == "__main__":
    main()
