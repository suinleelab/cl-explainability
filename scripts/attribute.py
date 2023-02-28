"""Run explanation methods for encoder representations."""

import os
import pickle
from functools import partial

import constants
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
    get_black_baseline,
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
from cl_explain.attributions.rise import RISE
from cl_explain.explanations.contrastive_corpus_similarity import (
    ContrastiveCorpusSimilarity,
)
from cl_explain.explanations.contrastive_weighted_score import ContrastiveWeightedScore
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
    # Normalize CIFAR-10 and MURA.
    normalize = False
    if args.dataset_name in ["cifar", "mura"]:
        normalize = True
    val_dataset, _, _ = load_data(
        dataset_name=args.dataset_name,
        subset="val",
        batch_size=args.batch_size,
        normalize=normalize,
        augment=False,
    )
    train_dataset, _, _ = load_data(
        dataset_name=args.dataset_name,
        subset="train",
        batch_size=args.batch_size,
        normalize=normalize,
        augment=False,
    )
    if args.dataset_name in ["imagenet", "imagenette2"]:
        val_labels = [sample[0].split("/")[-2] for sample in val_dataset.samples]
        train_labels = [sample[0].split("/")[-2] for sample in train_dataset.samples]
        unique_labels = constants.IMAGENETTE_SYNSETS
    elif args.dataset_name in ["cifar"]:
        val_labels = val_dataset.targets
        train_labels = train_dataset.targets
        unique_labels = list(range(constants.NUM_CLASSES_CIFAR))
    elif args.dataset_name in ["mura"]:
        val_labels = [sample[1] for sample in val_dataset.samples]
        train_labels = [sample[1] for sample in train_dataset.samples]
        unique_labels = list(range(constants.NUM_CLASSES_MURA))
    else:
        raise NotImplementedError(
            f"--dataset-name={args.dataset_name} is not implemented!"
        )

    img_h, img_w, removal = get_image_dataset_meta(args.dataset_name)
    if removal == "blurring":
        get_baseline = transforms.GaussianBlur(21, sigma=args.blur_strength).to(device)
    elif removal == "black":
        get_baseline = partial(
            get_black_baseline, dataset_name=args.dataset_name, normalize=normalize
        )
    else:
        raise NotImplementedError(f"removal={removal} is not implemented!")
    feature_mask = make_superpixel_map(
        img_h, img_w, args.superpixel_dim, args.superpixel_dim
    )  # Mask for grouping pixels into superpixels.
    feature_mask = feature_mask.to(device)

    outputs = {target: {} for target in unique_labels}
    for target in unique_labels:
        if args.different_classes:
            val_explicand_idx = (
                torch.Tensor(
                    [label != target and label in unique_labels for label in val_labels]
                )
                .nonzero()
                .flatten()
            )
        else:
            val_explicand_idx = (
                torch.Tensor([label == target for label in val_labels])
                .nonzero()
                .flatten()
            )
        train_target_idx = (
            torch.Tensor([label == target for label in train_labels])
            .nonzero()
            .flatten()
        )
        train_nontarget_idx = (
            torch.Tensor([label != target for label in train_labels])
            .nonzero()
            .flatten()
        )
        train_nontarget_size = train_nontarget_idx.size(0)

        val_explicand_idx = val_explicand_idx[torch.randperm(val_explicand_idx.size(0))]
        train_target_idx = train_target_idx[torch.randperm(train_target_idx.size(0))]

        val_explicand_idx = val_explicand_idx[: args.explicand_size]
        train_corpus_idx = train_target_idx[: args.corpus_size]
        outputs[target]["val_explicand_idx"] = val_explicand_idx
        outputs[target]["train_corpus_idx"] = train_corpus_idx

        # Get left-over indices from the training set for foil sampling.
        # Make sure that the non-target training samples and target training samples
        # have the same ratio before and after the corpus set has been taken out.
        corpus_prop = args.corpus_size / train_target_idx.size(0)
        leftover_nontarget_size = int(train_nontarget_size * (1 - corpus_prop))
        leftover_nontarget_idx = train_nontarget_idx[
            torch.randperm(train_nontarget_size)
        ][:leftover_nontarget_size]

        leftover_target_idx = set(train_target_idx.numpy()) - set(
            train_corpus_idx.numpy()
        )
        leftover_target_idx = torch.LongTensor(list(leftover_target_idx))

        train_leftover_idx = torch.cat([leftover_nontarget_idx, leftover_target_idx])
        train_leftover_idx = train_leftover_idx[
            torch.randperm(train_leftover_idx.size(0))
        ]
        outputs[target]["train_leftover_idx"] = train_leftover_idx
        if "contrastive" in args.explanation_name:
            outputs[target]["train_foil_idx"] = train_leftover_idx[: args.foil_size]

    if args.randomize_model:
        print("Randomizing encoder parameters...")
        for param in encoder.parameters():
            torch.nn.init.trunc_normal_(
                param,
                mean=torch.mean(param),
                std=torch.std(param),
                a=torch.min(param),
                b=torch.max(param),
            )

    print("Computing feature attributions for each class...")
    explicand_batch_size = args.batch_size
    if args.attribution_name == "kernel_shap":
        explicand_batch_size = 1
    for target in tqdm(unique_labels):
        explicand_dataloader = DataLoader(
            Subset(val_dataset, indices=outputs[target]["val_explicand_idx"]),
            batch_size=explicand_batch_size,
            shuffle=False,
        )
        corpus_dataloader = DataLoader(
            Subset(train_dataset, indices=outputs[target]["train_corpus_idx"]),
            batch_size=args.batch_size,
            shuffle=False,
        )
        if args.explanation_name == "self_weighted":
            explanation_model = WeightedScore(
                encoder=encoder, normalize=args.normalize_similarity
            )
        elif args.explanation_name == "contrastive_self_weighted":
            foil_dataloader = DataLoader(
                Subset(train_dataset, indices=outputs[target]["train_foil_idx"]),
                batch_size=args.batch_size,
                shuffle=False,
            )
            explanation_model = ContrastiveWeightedScore(
                encoder=encoder,
                foil_dataloader=foil_dataloader,
                normalize=args.normalize_similarity,
                batch_size=args.batch_size,
            )
        elif args.explanation_name == "corpus":
            explanation_model = CorpusSimilarity(
                encoder=encoder,
                corpus_dataloader=corpus_dataloader,
                normalize=args.normalize_similarity,
                batch_size=args.batch_size,
            )
        elif args.explanation_name == "contrastive_corpus":
            foil_dataloader = DataLoader(
                Subset(train_dataset, indices=outputs[target]["train_foil_idx"]),
                batch_size=args.batch_size,
                shuffle=False,
            )
            explanation_model = ContrastiveCorpusSimilarity(
                encoder=encoder,
                corpus_dataloader=corpus_dataloader,
                foil_dataloader=foil_dataloader,
                normalize=args.normalize_similarity,
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

            # Update weight for self-weighted explanation models.
            if "self_weighted" in args.explanation_name:
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
                    stdevs=0.2,
                    baselines=baseline,
                )
            elif args.attribution_name == "kernel_shap":
                attribution_model = KernelShap(explanation_model)
                attribution = attribution_model.attribute(
                    explicand,
                    baselines=baseline,
                    feature_mask=feature_mask,
                    n_samples=5000,
                )
            elif args.attribution_name == "gradient_shap":
                attribution_model = GradientShap(explanation_model)
                attribution = attribution_model.attribute(
                    explicand,
                    baselines=baseline,
                    n_samples=50,
                    stdevs=0.2,
                )
            elif args.attribution_name == "rise":
                if args.dataset_name in ["cifar"]:
                    grid_shape = (4, 4)
                else:
                    grid_shape = (7, 7)
                attribution_model = RISE(explanation_model)
                attribution = attribution_model.attribute(
                    explicand,
                    grid_shape=grid_shape,
                    baselines=baseline,
                    mask_prob=0.5,
                    n_samples=5000,
                    normalize_by_mask_prob=True,
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
        normalize_similarity=args.normalize_similarity,
        explanation_name=args.explanation_name,
        attribution_name=args.attribution_name,
        seed=args.seed,
        randomize_model=args.randomize_model,
    )
    os.makedirs(result_path, exist_ok=True)
    output_filename = get_output_filename(
        different_classes=args.different_classes,
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
