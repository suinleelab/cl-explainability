"""Representation learning explanation behaviors based on a corpus."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class CorpusBasedExplanation(nn.Module):
    """
    Base class for all representation-based explanation behaviors with a corpus.

    Args:
    ----
        encoder: Encoder module to explain.
        corpus_dataloader: Data loader of corpus examples to be encoded by `encoder`.
        batch_size: Mini-batch size for loading the corpus representations. This is
            useful when the entire corpus set fails to fit in compute memory.
    """

    def __init__(
        self,
        encoder: nn.Module,
        corpus_dataloader: DataLoader,
        batch_size: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.corpus_dataloader = corpus_dataloader

        self.corpus_rep = self._encode(self.corpus_dataloader)
        self.corpus_size = self.corpus_rep.size(0)
        self.rep_dim = self.corpus_rep.size(-1)
        self.corpus_rep_dataloader = DataLoader(
            TensorDataset(self.corpus_rep),
            batch_size=batch_size,
            shuffle=False,
        )

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
