"""Contrastive explanation based on an explicand's similarity to a corpus vs. foil."""
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cl_explain.explanations.corpus_similarity import (
    CorpusGaussianSimilarity,
    CorpusSimilarity,
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
        explicand_encoder: Optional alternative encoder for explicand.  Same as
             encoder by default.
        device: Optional device to perform encoding on.
    """

    def __init__(
        self,
        encoder: nn.Module,
        corpus_dataloader: DataLoader,
        foil_dataloader: DataLoader,
        batch_size: int = 64,
        sigma2: float = 1.0,
        explicand_encoder: Optional[nn.Module] = None,
        device: Optional[int] = None,
    ) -> None:
        super().__init__(
            encoder=encoder,
            corpus_dataloader=corpus_dataloader,
            batch_size=batch_size,
            sigma2=sigma2,
            explicand_encoder=explicand_encoder,
            device=device,
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


class ContrastiveCorpusSimilarity(CorpusSimilarity):
    """
    An explicand's cosine or dot product similarity score with a corpus vs. a foil.

    Args:
    ----
        encoder: Encoder module to explain.
        corpus_dataloader: Data loader of corpus examples to be encoded by `encoder`.
        foil_dataloader: Data loader of foil examples to be encoded by `encoder`.
        normalize: Whether to normalize dot product similarity by product of vector
            norms (that is, whether to use cosine similarity).
        batch_size: Mini-batch size for loading the corpus and foil representations.
        explicand_encoder: Optional alternative encoder for explicand.  Same as
            encoder by default.
        device: Optional device to perform encoding on.
    """

    def __init__(
        self,
        encoder: nn.Module,
        corpus_dataloader: DataLoader,
        foil_dataloader: DataLoader,
        normalize: bool,
        batch_size: int = 64,
        explicand_encoder: Optional[nn.Module] = None,
        device: Optional[int] = None,
    ) -> None:
        super().__init__(
            encoder=encoder,
            corpus_dataloader=corpus_dataloader,
            normalize=normalize,
            batch_size=batch_size,
            explicand_encoder=explicand_encoder,
            device=device,
        )
        self.foil_dataloader = foil_dataloader
        self.foil_rep = self._encode(self.foil_dataloader)
        self.foil_size = self.foil_rep.size(0)
        self.foil_rep_dataloader = DataLoader(
            TensorDataset(self.foil_rep),
            batch_size=batch_size,
            shuffle=False,
        )
        self.foil_rep_mean = self._encode_mean(
            self.foil_dataloader, normalize=self.normalize
        )

    def _rep_mean_forward(self, explicand: torch.Tensor) -> torch.Tensor:
        explicand_rep = self.encoder(explicand)
        if self.normalize:
            explicand_rep /= explicand_rep.norm(dim=-1).unsqueeze(-1)
        corpus_similarity = (
            explicand_rep * self.corpus_rep_mean.to(explicand_rep.device)
        ).sum(dim=-1)
        foil_similarity = (
            explicand_rep * self.foil_rep_mean.to(explicand_rep.device)
        ).sum(dim=-1)
        return corpus_similarity - foil_similarity

    def _rep_pairwise_forward(self, explicand: torch.Tensor) -> torch.Tensor:
        corpus_similarity = self._compute_similarity(
            explicand, self.corpus_rep_dataloader, self.corpus_size
        )
        foil_similarity = self._compute_similarity(
            explicand, self.foil_rep_dataloader, self.foil_size
        )
        return corpus_similarity - foil_similarity

    def forward(
        self, explicand: torch.Tensor, implementation: str = "mean"
    ) -> torch.Tensor:
        """
        Forward  pass.

        Args:
        ----
            explicand: Input explicands to explain, with shape `(batch_size, *)`, where
                * denotes the encoder input size of one sample.
            implementation: "mean" for using the foil representation mean to compute
                similarity. "pairwise" for computing the similarity between each
                explicand and all foil samples then averaging. The two implementations
                return the same results.
        """
        available_implementations = ["mean", "pairwise"]
        if implementation == "mean":
            return self._rep_mean_forward(explicand)
        elif implementation == "pairwise":
            return self._rep_pairwise_forward(explicand)
        else:
            raise NotImplementedError(
                f"implementation={implementation} is"
                f" not one of {available_implementations}!"
            )
