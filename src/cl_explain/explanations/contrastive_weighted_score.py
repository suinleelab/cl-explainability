"""Contrastive explanation for an explicand's weighted similarity vs. foil."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cl_explain.explanations.weighted_score import WeightedScore


class ContrastiveWeightedScore(WeightedScore):
    """
    Module for an explicand's cosine similarity to a weight vs. to a foil.

    Args:
    ----
       encoder: Encoder module to explain.
       foil_dataloader: Data loader of foil examples to be encoded by `encoder`.
       normalize: Whether to normalize dot product similarity by product of vector
           norms (that is, whether to use cosine similarity).
       batch_size: Mini-batch size for loading the foil representations.
    """

    def __init__(
        self,
        encoder: nn.Module,
        foil_dataloader: DataLoader,
        normalize: bool,
        batch_size: int = 64,
    ) -> None:
        super().__init__(encoder=encoder, normalize=normalize)
        self.foil_dataloader = foil_dataloader
        self.foil_rep = self._encode(self.foil_dataloader)
        self.foil_size = self.foil_rep.size(0)
        self.foil_rep_dataloader = DataLoader(
            TensorDataset(self.foil_rep),
            batch_size=batch_size,
            shuffle=False,
        )

    def forward(self, explicand: torch.Tensor) -> torch.Tensor:
        weight_similarity = self._compute_pointwise_similarity(explicand)
        foil_similarity = self._compute_pairwise_similarity(
            explicand, self.foil_rep_dataloader, self.foil_size
        )
        return weight_similarity - foil_similarity
