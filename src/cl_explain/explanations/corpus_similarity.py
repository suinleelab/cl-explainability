"""Explanation behaviors based on an explicand representation similarity to a corpus."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class CorpusSimilarity(nn.Module):
    """
    Module class for computing an explicand's total similarity score with a corpus.

    Args:
    ----
        encoder: Encoder module to explain.
        corpus_dataloader: Data loader of corpus examples, with shape (corpus_size, *),
            where * indicates the input dimensions of `encoder`.
        corpus_batch_size: Mini-batch size for loading the corpus representations. This
            is useful when the entire corpus set fails to fit in compute memory.
        sigma2: The variance parameter for the Gaussian similarity kernel.
    """

    def __init__(
        self,
        encoder: nn.Module,
        corpus_dataloader: DataLoader,
        corpus_batch_size: int = 64,
        sigma2: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.corpus_dataloader = corpus_dataloader
        self.sigma2 = sigma2

        self.corpus_rep = self._encode_corpus()
        self.rep_dim = self.corpus_rep.size(-1)
        self.corpus_rep_dataloader = DataLoader(
            TensorDataset(self.corpus_rep),
            batch_size=corpus_batch_size,
            shuffle=False,
        )

    def forward(self, explicand: torch.Tensor) -> torch.Tensor:
        explicand_rep = self.encoder(explicand)
        distance = 0
        for (x,) in self.corpus_rep_dataloader:
            x = x.to(explicand_rep.device)
            x = -self._compute_norm(explicand_rep, x) ** 2
            x /= self.rep_dim  # Normalize by embedding dimension to avoid large values.
            x = torch.exp(x / (2 * self.sigma2))
            x = x.sum(dim=1)
            distance += x
        return distance

    def _encode_corpus(self) -> torch.Tensor:
        encoder_device = [param.device for param in self.encoder.parameters()][0]
        corpus_rep = []
        for x, _ in self.corpus_dataloader:
            x = x.to(encoder_device)
            x = self.encoder(x).detach().cpu()
            corpus_rep.append(x)
        return torch.cat(corpus_rep)

    def _compute_norm(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=-1)
