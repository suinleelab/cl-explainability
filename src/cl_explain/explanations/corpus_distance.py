"""Explanation behaviors based on an explicand's representation distance to a corpus."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cl_explain.explanations.corpus_based_explanation import CorpusBasedExplanation


class CorpusDistance(CorpusBasedExplanation):
    """
    Module class for computing an explicand's average distance to a corpus.

    Args:
    ----
        encoder: Encoder module to explain.
        corpus_dataloader: Data loader of corpus examples to be encoded by `encoder`.
        batch_size: Mini-batch size for loading the corpus representations. This is
            useful when the entire corpus set fails to fit in compute memory.
    """

    def __init__(
        self,
        encoder: nn.Module,
        corpus_dataloader: DataLoader,
        batch_size: int = 64,
    ) -> None:
        super().__init__(
            encoder=encoder, corpus_dataloader=corpus_dataloader, batch_size=batch_size
        )

    def forward(self, explicand: torch.Tensor) -> torch.Tensor:
        return self._compute_distance(
            explicand, self.corpus_rep_dataloader, self.corpus_size
        )

    def _compute_distance(
        self, explicand: torch.Tensor, rep_dataloader: DataLoader, rep_data_size: int
    ) -> torch.Tensor:
        explicand_rep = self.encoder(explicand)
        distance = 0
        for (x,) in rep_dataloader:
            x = x.to(explicand_rep.device)
            x = self._compute_norm(explicand_rep, x) ** 2
            x = x.sum(dim=1)
            distance += x
        return distance / rep_data_size
