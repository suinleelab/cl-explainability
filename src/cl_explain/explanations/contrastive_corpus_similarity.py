"""Contrastive explanation based on an explicand's similarity to a corpus vs. foil."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cl_explain.explanations.corpus_similarity import (
    CorpusCosineSimilarity,
    CorpusGaussianSimilarity,
)


class ContrastiveCorpusGaussianSimilarity(CorpusGaussianSimilarity):
    """
    Module for an explicand's Gaussian similarity score with a corpus vs. a foil.

    Args:
    ----
       encoder: Encoder module to explain.
       corpus_dataloader: Data loader of corpus examples to be encoded by `encoder`.
       foil_dataloader: Data loader of foil examples to be encoded by `encoder`.
       batch_size: Mini-batch size for loading the corpus and foil representations.
       sigma2: The variance parameter for the Gaussian similarity kernel.
    """

    def __init__(
        self,
        encoder: nn.Module,
        corpus_dataloader: DataLoader,
        foil_dataloader: DataLoader,
        batch_size: int = 64,
        sigma2: float = 1.0,
    ) -> None:
        super().__init__(
            encoder=encoder,
            corpus_dataloader=corpus_dataloader,
            batch_size=batch_size,
            sigma2=sigma2,
        )
        self.foil_dataloader = foil_dataloader
        self.foil_rep = self._encode(self.foil_dataloader)
        self.foil_size = self.foil_rep.size(0)
        self.foil_rep_dataloader = DataLoader(
            TensorDataset(self.foil_rep),
            batch_size=batch_size,
            shuffle=False,
        )

    def forward(self, explicand: torch.Tensor) -> torch.Tensor:
        corpus_similarity = self._compute_similarity(
            explicand, self.corpus_rep_dataloader, self.corpus_size
        )
        foil_similarity = self._compute_similarity(
            explicand, self.foil_rep_dataloader, self.foil_size
        )
        return corpus_similarity / (corpus_similarity + foil_similarity)


class ContrastiveCorpusCosineSimilarity(CorpusCosineSimilarity):
    """
    Module for an explicand's cosine similarity score with a corpus vs. a foil.

    Args:
    ----
       encoder: Encoder module to explain.
       corpus_dataloader: Data loader of corpus examples to be encoded by `encoder`.
       foil_dataloader: Data loader of foil examples to be encoded by `encoder`.
       batch_size: Mini-batch size for loading the corpus and foil representations.
    """

    def __init__(
        self,
        encoder: nn.Module,
        corpus_dataloader: DataLoader,
        foil_dataloader: DataLoader,
        batch_size: int = 64,
    ) -> None:
        super().__init__(
            encoder=encoder,
            corpus_dataloader=corpus_dataloader,
            batch_size=batch_size,
        )
        self.foil_dataloader = foil_dataloader
        self.foil_rep = self._encode(self.foil_dataloader)
        self.foil_size = self.foil_rep.size(0)
        self.foil_rep_dataloader = DataLoader(
            TensorDataset(self.foil_rep),
            batch_size=batch_size,
            shuffle=False,
        )

    def forward(self, explicand: torch.Tensor) -> torch.Tensor:
        corpus_similarity = self._compute_similarity(
            explicand, self.corpus_rep_dataloader, self.corpus_size
        )
        foil_similarity = self._compute_similarity(
            explicand, self.foil_rep_dataloader, self.foil_size
        )
        return corpus_similarity - foil_similarity
