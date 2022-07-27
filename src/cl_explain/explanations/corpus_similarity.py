"""Explanation behaviors based on an explicand representation similarity to a corpus."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class CorpusSimilarity(nn.Module):
    """
    Module class for computing an explicand's average similarity score with a corpus.

    Args:
    ----
        encoder: Encoder module to explain.
        corpus_dataloader: Data loader of corpus examples, with shape (corpus_size, *),
            where * indicates the input dimensions of `encoder`.
        batch_size: Mini-batch size for loading the corpus representations. This is
            useful when the entire corpus set fails to fit in compute memory.
        sigma2: The variance parameter for the Gaussian similarity kernel.
    """

    def __init__(
        self,
        encoder: nn.Module,
        corpus_dataloader: DataLoader,
        batch_size: int = 64,
        sigma2: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.corpus_dataloader = corpus_dataloader
        self.sigma2 = sigma2

        self.corpus_rep = self._encode(self.corpus_dataloader)
        self.corpus_size = self.corpus_rep.size(0)
        self.rep_dim = self.corpus_rep.size(-1)
        self.corpus_rep_dataloader = DataLoader(
            TensorDataset(self.corpus_rep),
            batch_size=batch_size,
            shuffle=False,
        )

    def forward(self, explicand: torch.Tensor) -> torch.Tensor:
        return self._compute_similarity(
            explicand, self.corpus_rep_dataloader, self.corpus_size
        )

    def _compute_similarity(
        self, explicand: torch.Tensor, rep_dataloader: DataLoader, rep_data_size: int
    ) -> torch.Tensor:
        explicand_rep = self.encoder(explicand)
        similarity = 0
        for (x,) in rep_dataloader:
            x = x.to(explicand_rep.device)
            x = -self._compute_norm(explicand_rep, x) ** 2
            x /= self.rep_dim  # Normalize by embedding dimension to avoid large values.
            x = torch.exp(x / (2 * self.sigma2))
            x = x.sum(dim=1)
            similarity += x
        return similarity / rep_data_size  # Average over number of comparisons.

    def _encode(self, dataloader: DataLoader) -> torch.Tensor:
        encoder_device = [param.device for param in self.encoder.parameters()][0]
        rep = []
        for x, _ in dataloader:
            x = x.to(encoder_device)
            x = self.encoder(x).detach().cpu()
            rep.append(x)
        return torch.cat(rep)

    @staticmethod
    def _compute_norm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=-1)
