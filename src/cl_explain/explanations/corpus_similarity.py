"""Explanation behaviors based on an explicand representation similarity to a corpus."""

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cl_explain.explanations.corpus_based_explanation import CorpusBasedExplanation


class CorpusGaussianSimilarity(CorpusBasedExplanation):
    """
    Module class for an explicand's average Gaussian similarity score with a corpus.

    Args:
    ----
        encoder: Encoder module to explain.
        corpus_dataloader: Data loader of corpus examples to be encoded by `encoder`.
        batch_size: Mini-batch size for loading the corpus representations. This is
            useful when the entire corpus set fails to fit in compute memory.
        sigma2: The variance parameter for the Gaussian similarity kernel.
        explicand_encoder: Optional alternative encoder for explicand.  Same as
            encoder by default.
        device: Optional device to perform encoding on.
    """

    def __init__(
        self,
        encoder: nn.Module,
        corpus_dataloader: DataLoader,
        batch_size: int = 64,
        sigma2: float = 1.0,
        explicand_encoder: Optional[nn.Module] = None,
        device: Optional[int] = None,
    ) -> None:
        super().__init__(
            encoder=encoder,
            corpus_dataloader=corpus_dataloader,
            batch_size=batch_size,
            device=device,
        )
        self.sigma2 = sigma2

        # By default use same encoder for explicand as for the corpus
        if not explicand_encoder:
            self.explicand_encoder = encoder
        else:
            self.explicand_encoder = explicand_encoder

    def forward(self, explicand: torch.Tensor) -> torch.Tensor:
        return self._compute_similarity(
            explicand, self.corpus_rep_dataloader, self.corpus_size
        )

    def _compute_similarity(
        self, explicand: torch.Tensor, rep_dataloader: DataLoader, rep_data_size: int
    ) -> torch.Tensor:
        explicand_rep = self.explicand_encoder(explicand)
        similarity = 0
        for (x,) in rep_dataloader:
            x = x.to(explicand_rep.device)
            x = -self._compute_difference_norm(explicand_rep, x) ** 2
            x /= self.rep_dim  # Normalize by embedding dimension to avoid large values.
            x = torch.exp(x / (2 * self.sigma2))
            x = x.sum(dim=1)
            similarity += x
        return similarity / rep_data_size  # Average over number of comparisons.


class CorpusSimilarity(CorpusBasedExplanation):
    """
    An explicand's average cosine or dot product similarity score with a corpus.

    Args:
    ----
        encoder: Encoder module to explain.
        corpus_dataloader: Data loader of corpus examples to be encoded by `encoder`.
        normalize: Whether to normalize dot product similarity by product of vector
            norms (that is, whether to use cosine similarity).
        batch_size: Mini-batch size for loading the corpus representations. This is
            useful when the entire corpus set fails to fit in compute memory.
        explicand_encoder: Optional alternative encoder for explicand.  Same as
            encoder by default.
        device: Optional device to perform encoding on.
    """

    def __init__(
        self,
        encoder: nn.Module,
        corpus_dataloader: DataLoader,
        normalize: bool,
        batch_size: int = 64,
        explicand_encoder: Optional[nn.Module] = None,
        device: Optional[int] = None,
    ) -> None:
        super().__init__(
            encoder=encoder,
            corpus_dataloader=corpus_dataloader,
            batch_size=batch_size,
            device=device,
        )
        self.normalize = normalize
        self.corpus_rep_mean = self._encode_mean(
            self.corpus_dataloader, normalize=self.normalize
        )

        # By default use same encoder for explicand as for the corpus
        if not explicand_encoder:
            self.explicand_encoder = encoder
        else:
            self.explicand_encoder = explicand_encoder

    def _rep_mean_forward(self, explicand: torch.Tensor) -> torch.Tensor:
        explicand_rep = self.encoder(explicand)
        if self.normalize:
            explicand_rep /= explicand_rep.norm(dim=-1).unsqueeze(-1)
        return (explicand_rep * self.corpus_rep_mean.to(explicand_rep.device)).sum(
            dim=-1
        )

    def _rep_pairwise_forward(self, explicand: torch.Tensor) -> torch.Tensor:
        return self._compute_similarity(
            explicand, self.corpus_rep_dataloader, self.corpus_size
        )

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

    def _compute_similarity(
        self, explicand: torch.Tensor, rep_dataloader: DataLoader, rep_data_size: int
    ) -> torch.Tensor:
        explicand_rep = self.explicand_encoder(explicand)
        similarity = 0
        for (x,) in rep_dataloader:
            x = x.to(explicand_rep.device)
            if self.normalize:
                x = self._compute_cosine_similarity(explicand_rep, x)
            else:
                x = self._compute_dot_product(explicand_rep, x)
            x = x.sum(dim=1)
            similarity += x
        return similarity / rep_data_size  # Average over number of comparisons.
