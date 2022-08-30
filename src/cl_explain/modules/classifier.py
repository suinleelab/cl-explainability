"""PyTorch Lightning Module for classifiers."""

from typing import Dict, List, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import cohen_kappa


class LitClassifier(pl.LightningModule):
    """
    PyTorch Lightning Module for a classifier.

    Args:
    ----
        network: A pre-trained or randomly initialized network.
        lr: Learning rate.
        weight_decay: Weight decay parameter.
        lr_step_size: Decay the learning rate every `lr_step_size` epochs.
        lr_gamma: Multiple the learning rate by `lr_gamma` every `lr_step_size` epochs.
    """

    def __init__(
        self,
        network: nn.Module,
        lr: float = 0.1,
        weight_decay: float = 0.0001,
        lr_step_size: int = 10,
        lr_gamma: float = 0.1,
    ) -> None:
        super().__init__()
        self.network = network
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor, apply_eval_head: bool = False) -> torch.Tensor:
        return self.network(x, apply_eval_head=apply_eval_head)

    @staticmethod
    def evaluate(out: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute evaluation metrics."""
        loss = F.cross_entropy(out, target)
        pred = out.argmax(dim=-1)
        return {
            "loss": loss,
            "out": out.detach().cpu(),
            "pred": pred.detach().cpu(),
            "target": target.detach().cpu(),
        }

    def step(self, batch) -> Dict[str, Union[torch.Tensor, int]]:
        """Generate model output values for a batch of data."""
        x, target = batch
        out = self.evaluate(self(x, apply_eval_head=True), target)
        return out

    def epoch_end(
        self, outputs: List[Dict[str, Union[torch.Tensor, int]]], subset: str
    ) -> None:
        """Aggregate evaluation metrics at the end of an epoch."""
        outs = torch.cat([output["out"] for output in outputs])
        preds = torch.cat([output["pred"] for output in outputs])
        targets = torch.cat([output["target"] for output in outputs])
        loss = F.cross_entropy(outs, targets)
        acc = (preds == targets).float().mean()
        kappa = cohen_kappa(preds, targets, num_classes=preds.size(-1))

        self.log(f"{subset}/loss", loss, sync_dist=True)
        self.log(f"{subset}/acc", acc, sync_dist=True)
        self.log(f"{subset}/kappa", kappa, sync_dist=True)

    def training_step(self, batch, batch_idx) -> Dict[str, Union[torch.Tensor, int]]:
        return self.step(batch=batch)

    def validation_step(self, batch, batch_idx) -> Dict[str, Union[torch.Tensor, int]]:
        return self.step(batch=batch)

    def test_step(self, batch, batch_idx) -> Dict[str, Union[torch.Tensor, int]]:
        return self.step(batch=batch)

    def training_epoch_end(
        self, outputs: List[Dict[str, Union[torch.Tensor, int]]]
    ) -> None:
        self.epoch_end(outputs=outputs, subset="train")

    def validation_epoch_end(
        self, outputs: List[Dict[str, Union[torch.Tensor, int]]]
    ) -> None:
        self.epoch_end(outputs=outputs, subset="val")

    def test_epoch_end(
        self, outputs: List[Dict[str, Union[torch.Tensor, int]]]
    ) -> None:
        self.epoch_end(outputs=outputs, subset="test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma
        )
        return [optimizer], [scheduler]
