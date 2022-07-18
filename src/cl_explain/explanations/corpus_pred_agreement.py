"""Explanation behaviors using prediction agreement between an explicand and corpus."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class CorpusPredAgreement(nn.Module):
    """
    Module class for computing an explicand's total prediction agreement with a corpus.

    Args:
    ----
        encoder: Encoder module to explain. It should return predicted output with
            `apply_eval_head=True` in `encoder.forward()`.
        corpus_dataloader: Data loader of corpus examples, with shape (corpus_size, *),
            where * indicates the input dimensions of `encoder`.
    """

    def __init__(
        self,
        encoder: nn.Module,
        corpus_dataloader: DataLoader,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.corpus_dataloader = corpus_dataloader

    def forward(self, explicand: torch.Tensor) -> torch.Tensor:
        explicand_pred = self.encoder(explicand, apply_eval_head=True).argmax(dim=-1)
        agreement = 0
        for x, _ in self.corpus_dataloader:
            x = x.to(explicand.device)
            x = self.encoder(x, apply_eval_head=True).argmax(dim=-1)
            x = explicand_pred.unsqueeze(1) - x.unsqueeze(0)
            x = (
                x == 0
            ) * 1.0  # Indicator of whether predictions are pairwise identical.
            x = x.mean(dim=1).sum()  # Sum of agreement proportions.
            agreement += x
        return agreement
