"""Evaluate explanation methods for encoder representations."""

import os
import pickle

import torch
import torch.nn as nn
import torchvision.transforms as transforms
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
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm

from cl_explain.explanations.contrastive_corpus_similarity import (
    ContrastiveCorpusSimilarity,
)
from cl_explain.explanations.corpus_majority_prob import CorpusMajorityProb
from cl_explain.explanations.corpus_similarity import CorpusSimilarity
from cl_explain.metrics.ablation import ImageAblation


def main():
    """Main function."""
    args = parse_args(evaluate=True)
    make_reproducible(args.seed)
    device = get_device(args.use_gpu, args.gpu_num)
    print("Loading encoder...")
    encoder = load_encoder(args.encoder_name)
    encoder.eval()
    encoder.to(device)
    print("Loading dataset...")
    val_dataset, _, _ = load_data(
        dataset_name=args.dataset_name, subset="val", batch_size=args.batch_size
    )
    img_h, img_w, removal = get_image_dataset_meta(args.dataset_name)
    if removal == "blurring":
        get_baseline = transforms.GaussianBlur(21, sigma=args.blur_strength).to(device)
    else:
        raise NotImplementedError(f"removal={removal} is not implemented!")

    result_path = get_result_path(
        dataset_name=args.dataset_name,
        encoder_name=args.encoder_name,
        attribution_name=args.attribution_name,
        seed=args.seed,
        contrast=args.contrast,
    )
    output_filename = get_output_filename(
        corpus_size=args.corpus_size,
        contrast=args.contrast,
        foil_size=args.foil_size,
        explicand_size=args.explicand_size,
        attribution_name=args.attribution_name,
        superpixel_dim=args.superpixel_dim,
        removal=removal,
        blur_strength=args.blur_strength,
    )
    with open(os.path.join(result_path, output_filename), "rb") as handle:
        outputs = pickle.load(handle)

    if args.eval_superpixel_dim > 1:
        pixelate = nn.AvgPool2d(
            kernel_size=(args.eval_superpixel_dim, args.eval_superpixel_dim)
        )
    else:
        pixelate = None

    print("Evaluating feature attributions for each class...")
    results = {target: {} for target in outputs.keys()}
    for target, target_output in tqdm(outputs.items()):
        explicand_dataloader = DataLoader(
            Subset(val_dataset, indices=target_output["explicand_idx"]),
            batch_size=args.batch_size,
            shuffle=False,
        )
        attribution_dataloader = DataLoader(
            TensorDataset(target_output["attributions"]),
            batch_size=args.batch_size,
            shuffle=False,
        )
        corpus_dataloader = DataLoader(
            Subset(val_dataset, indices=target_output["corpus_idx"]),
            batch_size=args.batch_size,
            shuffle=False,
        )
        leftover_idx = target_output["leftover_idx"]
        # Shuffle indices to ensure fair comparison between contrastive vs.
        # non-contrastive explanation methods. Otherwise contrastive methods would use
        # the same foil during attribution and evaluation.
        leftover_idx = leftover_idx[torch.randperm(leftover_idx.size(0))]
        eval_foil_dataloader = DataLoader(
            Subset(val_dataset, indices=leftover_idx[: args.eval_foil_size]),
            batch_size=args.batch_size,
            shuffle=False,
        )

        model_list = [
            CorpusSimilarity(
                encoder=encoder,
                corpus_dataloader=corpus_dataloader,
                batch_size=args.batch_size,
            ),
            ContrastiveCorpusSimilarity(
                encoder=encoder,
                corpus_dataloader=corpus_dataloader,
                foil_dataloader=eval_foil_dataloader,
                batch_size=args.batch_size,
            ),
            CorpusMajorityProb(encoder=encoder, corpus_dataloader=corpus_dataloader),
        ]
        model_name_list = [
            "similarity",
            "contrastive_similarity",
            "majority_pred_prob",
        ]
        image_ablation = ImageAblation(
            model_list,
            img_h,
            img_w,
            superpixel_h=args.eval_superpixel_dim,
            superpixel_w=args.eval_superpixel_dim,
        )

        insertion_curve_list = [[] for _ in range(image_ablation.num_models)]
        deletion_curve_list = [[] for _ in range(image_ablation.num_models)]
        insertion_num_features = None
        deletion_num_features = None

        for ([explicand, _], [attribution]) in zip(
            explicand_dataloader, attribution_dataloader
        ):
            explicand = explicand.to(device)
            attribution = attribution.to(device)
            baseline = get_baseline(explicand)

            if args.take_attribution_abs:
                attribution = attribution.abs()
            attribution = attribution.mean(dim=1).unsqueeze(
                1
            )  # Combine channel attributions.
            if args.eval_superpixel_dim > 1:
                attribution = pixelate(attribution)  # Get superpixel attributions.

            insertion_curves, insertion_num_features = image_ablation.evaluate(
                explicand,
                attribution,
                baseline,
                kind="insertion",
            )
            deletion_curves, deletion_num_features = image_ablation.evaluate(
                explicand,
                attribution,
                baseline,
                kind="deletion",
            )
            for j in range(image_ablation.num_models):
                insertion_curve_list[j].append(insertion_curves[j].detach().cpu())
                deletion_curve_list[j].append(deletion_curves[j].detach().cpu())

        results[target]["insertion_curves"] = [
            torch.cat(curve) for curve in insertion_curve_list
        ]
        results[target]["deletion_curves"] = [
            torch.cat(curve) for curve in deletion_curve_list
        ]
        results[target]["eval_model_names"] = model_name_list
        results[target]["insertion_num_features"] = insertion_num_features
        results[target]["deletion_num_features"] = deletion_num_features

    print("Saving results...")
    result_filename = output_filename.replace("outputs", "eval_results").replace(
        ".pkl", ""
    )
    result_filename += f"_eval_superpixel_dim={args.eval_superpixel_dim}"
    result_filename += f"_eval_foil_size={args.eval_foil_size}"
    if args.take_attribution_abs:
        result_filename += "_abs"
    result_filename += ".pkl"
    with open(os.path.join(result_path, result_filename), "wb") as handle:
        pickle.dump(results, handle)
    print("Done!")


if __name__ == "__main__":
    main()
