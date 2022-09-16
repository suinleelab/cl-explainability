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
        print(self.device)

    def _encode(self, dataloader: DataLoader) -> torch.Tensor:
        """Encode all data in a data loader into representations."""
        encoder_device = self.device
        if encoder_device is None:
            encoder_device = [param.device for param in self.encoder.parameters()][0]
        rep = []
        for x, _ in dataloader:
            x = x.to(encoder_device)
            x = self.encoder(x).detach().cpu()
            rep.append(x)
        return torch.cat(rep)

    def _encode_mean(self, dataloader: DataLoader, normalize: bool) -> torch.Tensor:
        """
        Compute representation mean of all data.

        Args:
        ----
            dataloader: Dataloader of all data. Each iteration loads a tuple of encoder
                input and the input's label. The label is not used.
            normalize: Whether to compute the mean of l2-normalized representations
                or just the mean of representations.

        Returns
        -------
            A representation mean tensor with size `representation_size`.
        """
        encoder_device = self.device
        if encoder_device is None:
            encoder_device = [param.device for param in self.encoder.parameters()][0]
        rep = []
        for x, _ in dataloader:
            x = x.to(encoder_device)
            x = self.encoder(x).detach()  # Detach right away to minimize memory usage.
            if normalize:
                x /= x.norm(dim=-1).unsqueeze(-1)
            x = x.cpu()
            rep.append(x)
        return torch.cat(rep).mean(dim=0)

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
