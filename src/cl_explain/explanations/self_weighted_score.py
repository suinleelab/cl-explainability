"""
Explanation behavior using an explicand representation's self-weighted importance, as
proposed by Crabbe et al. Label-Free Explainability for Unsupervised Models.
https://arxiv.org/abs/2203.01928
"""

import torch
import torch.nn as nn


class SelfWeightedScore(nn.Module):
    """
    Module class for label-free feature importance proposed by Crabbe et al. 2022.

    Args:
    ----
        encoder: Encoder module to explain.
    """

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, explicand: torch.Tensor) -> torch.Tensor:
        explicand_rep = self.encoder(explicand)
        return (explicand_rep * explicand_rep).sum(dim=-1)
