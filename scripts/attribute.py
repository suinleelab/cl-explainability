"""Run explanation methods for encoder representations."""

import os
import pickle

import torch
import torchvision.transforms as transforms
from captum.attr import (
    GradientShap,
    IntegratedGradients,
    KernelShap,
    NoiseTunnel,
    Saliency,
)
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
from cl_explain.explanations.contrastive_corpus_similarity import (
    ContrastiveCorpusSimilarity,
)
from cl_explain.explanations.corpus_similarity import CorpusSimilarity
from cl_explain.explanations.weighted_score import WeightedScore
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
    dataset, dataloader, class_map = load_data(
        dataset_name=args.dataset_name, subset="val", batch_size=args.batch_size
    )
    img_h, img_w, removal = get_image_dataset_meta(args.dataset_name)
    if removal == "blurring":
        get_baseline = transforms.GaussianBlur(21, sigma=args.blur_strength).to(device)
    else:
        raise NotImplementedError(f"removal={removal} is not implemented!")
    feature_mask = make_superpixel_map(
        img_h, img_w, args.superpixel_dim, args.superpixel_dim
    )  # Mask for grouping pixels into superpixels.
    feature_mask = feature_mask.to(device)

    labels = []
    for _, label in dataloader:
        labels.append(label)
    labels = torch.cat(labels)
    unique_labels = labels.unique().numpy()
    outputs = {target: {"source_label": class_map[target]} for target in unique_labels}
    all_idx = torch.arange(labels.size(0))

    for target in unique_labels:
        target_idx = (labels == target).nonzero().flatten()
        target_idx = target_idx[torch.randperm(target_idx.size(0))]
        explicand_idx = target_idx[: args.explicand_size]
        corpus_idx = target_idx[
            args.explicand_size : (args.explicand_size + args.corpus_size)
        ]
        outputs[target]["explicand_idx"] = explicand_idx
        outputs[target]["corpus_idx"] = corpus_idx
        leftover_idx = set(all_idx.numpy()) - set(explicand_idx.numpy()).union(
            set(corpus_idx.numpy())
        )
        leftover_idx = torch.LongTensor(list(leftover_idx))
        leftover_idx = leftover_idx[torch.randperm(leftover_idx.size(0))]
        outputs[target]["leftover_idx"] = leftover_idx
        if args.explanation_name == "contrastive":
            outputs[target]["foil_idx"] = leftover_idx[: args.foil_size]

    print("Computing feature attributions for each class...")
    for target in tqdm(unique_labels):
        explicand_dataloader = DataLoader(
            Subset(dataset, indices=outputs[target]["explicand_idx"]),
            batch_size=args.batch_size,
            shuffle=False,
        )
        corpus_dataloader = DataLoader(
            Subset(dataset, indices=outputs[target]["corpus_idx"]),
            batch_size=args.batch_size,
            shuffle=False,
        )
        if args.explanation_name == "self_weighted":
            explanation_model = WeightedScore(encoder=encoder)
        elif args.explanation_name == "corpus":
            explanation_model = CorpusSimilarity(
                encoder=encoder,
                corpus_dataloader=corpus_dataloader,
                batch_size=args.batch_size,
            )
        elif args.explanation_name == "contrastive":
            foil_dataloader = DataLoader(
                Subset(dataset, indices=outputs[target]["foil_idx"]),
                batch_size=args.batch_size,
                shuffle=False,
            )
            explanation_model = ContrastiveCorpusSimilarity(
                encoder=encoder,
                corpus_dataloader=corpus_dataloader,
                foil_dataloader=foil_dataloader,
                batch_size=args.batch_size,
            )
        else:
            raise NotImplementedError(
                f"{args.explanation_name} explanation is not implemented!"
            )

        attribution_list = []
        for explicand, _ in explicand_dataloader:
            explicand = explicand.to(device)
            baseline = get_baseline(explicand)
            explicand.requires_grad = True

            # Update weight for self-weighted explanation model.
            if args.explanation_name == "self_weighted":
                explanation_model.generate_weight(explicand.detach().clone())

            if args.attribution_name == "vanilla_grad":
                attribution_model = Saliency(explanation_model)
                attribution = attribution_model.attribute(explicand, abs=False)
            elif args.attribution_name == "int_grad":
                attribution_model = IntegratedGradients(explanation_model)
                attribution = attribution_model.attribute(explicand, baselines=baseline)
            elif args.attribution_name == "smooth_vanilla_grad":
                attribution_model = NoiseTunnel(Saliency(explanation_model))
                attribution = attribution_model.attribute(
                    explicand,
                    nt_type="smoothgrad",
                    nt_samples=25,
                    nt_samples_batch_size=args.batch_size,
                    stdevs=1.0,
                )
            elif args.attribution_name == "smooth_int_grad":
                attribution_model = NoiseTunnel(IntegratedGradients(explanation_model))
                attribution = attribution_model.attribute(
                    explicand,
                    nt_type="smoothgrad",
                    nt_samples=25,
                    nt_samples_batch_size=args.batch_size,
                    stdevs=1.0,
                    baselines=baseline,
                )
            elif args.attribution_name == "kernel_shap":
                attribution_model = KernelShap(explanation_model)
                attribution = attribution_model.attribute(
                    explicand,
                    baselines=baseline,
                    feature_mask=feature_mask,
                    n_samples=10000,
                )
            elif args.attribution_name == "gradient_shap":
                attribution_model = GradientShap(explanation_model)
                attribution = attribution_model.attribute(
                    explicand,
                    baselines=baseline,
                    n_samples=50,
                    stdevs=1.0,
                )
            elif args.attribution_name == "random_baseline":
                attribution_model = RandomBaseline(explanation_model)
                attribution = attribution_model.attribute(explicand)
            else:
                raise NotImplementedError(
                    f"{args.attribution_name} attribution is not implemented!"
                )
            attribution_list.append(attribution.detach().cpu())
        outputs[target]["attributions"] = torch.cat(attribution_list)

    print("Saving outputs...")
    result_path = get_result_path(
        dataset_name=args.dataset_name,
        encoder_name=args.encoder_name,
        explanation_name=args.explanation_name,
        attribution_name=args.attribution_name,
        seed=args.seed,
    )
    os.makedirs(result_path, exist_ok=True)
    output_filename = get_output_filename(
        corpus_size=args.corpus_size,
        explanation_name=args.explanation_name,
        foil_size=args.foil_size,
        explicand_size=args.explicand_size,
        attribution_name=args.attribution_name,
        superpixel_dim=args.superpixel_dim,
        removal=removal,
        blur_strength=args.blur_strength,
    )
    with open(os.path.join(result_path, output_filename), "wb") as handle:
        pickle.dump(outputs, handle)
    print("Done!")


if __name__ == "__main__":
    main()
