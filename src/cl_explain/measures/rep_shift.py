"""Representation shift as a measure for evaluating feature importance."""

import torch
import torch.nn as nn


class RepShift(nn.Module):
    """
    Module class for measuring representation shift between original and modified input.

    Args:
    ----
        encoder: Encoder module.
    """

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, original_explicand, modified_explicand) -> torch.Tensor:
        """
        Forward pass.

        Args:
        ----
            original_explicand: Original explicand without any modifications, with
                shape `(batch_size, *)`, where * indicates the input dimensions of
                `RepShift.encoder`.
            modified_explicand: Explicand after feature modifications, with shape
                `(batch_size, *)`, where * indicates the input dimensions of
                `RepShift.encoder`.

        Return:
        ------
            The squared L2 distance between the original representation and modified
            representation for each sample.
        """
        original_rep = self.encoder(original_explicand)
        modified_rep = self.encoder(modified_explicand)
        return ((original_rep - modified_rep) ** 2).sum(dim=-1)
