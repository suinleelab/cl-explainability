"""Base classes for all representation explanation target functions."""
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ExplanationBase(nn.Module):
    """
    Base class for all representation-based explanation target functions.

    Args:
    ----
        encoder: Encoder module to explain.
        device: Optional device to perform encoding on.
    """

    def __init__(
        self,
        encoder: nn.Module,
        device: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.device = device

    def _encode(self, dataloader: DataLoader) -> torch.Tensor:
        """Encode all data in a data loader into representations."""
        encoder_device = self.device
        if not encoder_device:
            encoder_device = [param.device for param in self.encoder.parameters()][0]
        rep = []
        for x, _ in dataloader:
            x = x.to(encoder_device)
            x = self.encoder(x).detach().cpu()
            rep.append(x)
        return torch.cat(rep)

    @staticmethod
    def _compute_difference_norm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute pairwise difference norms."""
        return torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=-1)

    def _compute_cosine_similarity(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Compute pairwise cosine similarities."""
        return self._compute_dot_product(x, y) / (
            x.norm(dim=-1).unsqueeze(1) * y.norm(dim=-1).unsqueeze(0)
        )

    @staticmethod
    def _compute_dot_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute pairwise dot products."""
        return (x.unsqueeze(1) * y.unsqueeze(0)).sum(dim=-1)
