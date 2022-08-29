"""Load and evaluate a classifier network."""
import os

import constants
import pytorch_lightning as pl
from classifier_utils import (
    get_classifier_output_path,
    parse_classifier_args,
    split_train_val_dataset,
)
from experiment_utils import get_device, load_data
from torch.utils.data import DataLoader, Subset

from cl_explain.modules.classifier import LitClassifier


def main():
    """Main function."""
    args = parse_classifier_args(evaluate=True)
    pl.seed_everything(args.seed, workers=True)

    train_val_dataset, _, _ = load_data(
        dataset_name=args.dataset_name,
        subset="train",
        batch_size=args.batch_size,
        normalize=True,
        # No data augmentation during evaluation.
    )
    train_idx, val_idx = split_train_val_dataset(
        train_val_dataset, val_size=0.2, seed=constants.TRAIN_VAL_SPLIT_SEED
    )  # Keep training and validation set consistent.
    test_dataset, _, _ = load_data(
        dataset_name=args.dataset_name,
        subset="val",
        batch_size=args.batch_size,
        normalize=True,
    )
    test_idx = [i for i in range(len(test_dataset.samples))]
    max_epochs = 100
    if args.debug:
        train_idx = train_idx[:100]
        val_idx = val_idx[:50]
        test_idx = test_idx[:50]
        max_epochs = 3

    train_dataloader = DataLoader(
        Subset(train_val_dataset, indices=train_idx),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
    )
    val_dataloader = DataLoader(
        Subset(train_val_dataset, indices=val_idx),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
    )
    test_dataloader = DataLoader(
        Subset(test_dataset, indices=test_idx),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
    )
    save_dir = get_classifier_output_path(
        dataset_name=args.dataset_name,
        classifier_name=args.classifier_name,
        seed=args.seed,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        batch_size=args.batch_size,
        max_epochs=max_epochs,
    )
    model = LitClassifier.load_from_checkpoint(
        os.path.join(save_dir, args.ckpt_filename)
    )

    device = get_device(args.use_gpu, args.gpu_num)
    if args.use_gpu:
        accelerator = "gpu"
    else:
        accelerator = "cpu"

    # Trainer should not use distributed data parallelization during evaluation. This
    # is to ensure consistent metric values are obtained.
    if type(device) is list:
        assert (
            len(device) == 1
        ), "Only one device should be used for evaluation to ensure stable results!"
    trainer = pl.Trainer(accelerator=accelerator, devices=device)
    train_results = trainer.test(model=model, dataloaders=train_dataloader)[0]
    val_results = trainer.test(model=model, dataloaders=val_dataloader)[0]
    test_results = trainer.test(model=model, dataloaders=test_dataloader)[0]
    all_results = {"train": train_results, "val": val_results, "test": test_results}

    with open(os.path.join(save_dir, "eval_results.txt"), "w") as file:
        for subset, results in all_results.items():
            for metric, value in results.items():
                metric = metric.replace("test", subset)
                file.write(f"{metric}: {value:.4f}\n")


if __name__ == "__main__":
    main()
