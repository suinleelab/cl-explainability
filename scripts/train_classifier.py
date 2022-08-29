"""Train a classifier network."""

import os

import pytorch_lightning as pl
from classifier_utils import (
    get_classifier_output_path,
    parse_classifier_args,
    split_train_val_dataset,
)
from experiment_utils import get_device, load_data
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Subset

from cl_explain.modules.classifier import LitClassifier
from cl_explain.networks.resnet import PretrainedResNet18, PretrainedResNet50


def main():
    """Main function."""
    args = parse_classifier_args()
    pl.seed_everything(args.seed, workers=True)

    train_val_dataset, _, _ = load_data(
        dataset_name=args.dataset_name,
        subset="train",
        batch_size=args.batch_size,
        normalize=True,
    )
    train_val_augmented_dataset, _, _ = load_data(
        dataset_name=args.dataset_name,
        subset="train",
        batch_size=args.batch_size,
        normalize=True,
        augment=True,
    )
    train_idx, val_idx = split_train_val_dataset(
        train_val_dataset, val_size=0.2, seed=42
    )  # Keep training and validation set consistent.
    if args.debug:
        train_idx = train_idx[:100]
        val_idx = val_idx[:50]

    train_dataloader = DataLoader(
        Subset(
            train_val_augmented_dataset, indices=train_idx
        ),  # Training with data augmentation.
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    val_dataloader = DataLoader(
        Subset(train_val_dataset, indices=val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
    )

    device = get_device(args.use_gpu, args.gpu_num)
    if args.use_gpu:
        accelerator = "gpu"
    else:
        accelerator = "cpu"

    if args.dataset_name == "mura":
        num_classes = 2
    else:
        raise NotImplementedError(
            f"--dataset-name={args.dataset_name} is not implemented!"
        )

    if args.classifier_name == "resnet18":
        network = PretrainedResNet18(use_pretrained_head=False, num_classes=num_classes)
    elif args.classifier_name == "resnet50":
        network = PretrainedResNet50(use_pretrained_head=False, num_classes=num_classes)
    else:
        raise NotImplementedError(
            f"classifier_name={args.classifier_name} is not implemented!"
        )
    model = LitClassifier(
        network,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
    )

    if args.debug:
        max_epochs = 3
    else:
        max_epochs = 100
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
    os.makedirs(save_dir, exist_ok=True)
    logger = TensorBoardLogger(
        save_dir=save_dir,
        name="lightning_logs",
    )
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=device,
        strategy="ddp",
        max_epochs=max_epochs,
        check_val_every_n_epoch=1,
        callbacks=[
            EarlyStopping(monitor="val/loss", mode="min", patience=25),
            ModelCheckpoint(dirpath=save_dir, save_last=True),
        ],
        logger=logger,
    )
    trainer.fit(
        model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )


if __name__ == "__main__":
    main()
