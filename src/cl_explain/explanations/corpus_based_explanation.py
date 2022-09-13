"""Representation learning explanation behaviors based on a corpus."""
from typing import Optional

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cl_explain.explanations.explanation_base import ExplanationBase


class CorpusBasedExplanation(ExplanationBase):
    """
    Base class for behaviors related to a corpus.

    Args:
    ----
        encoder: Encoder module to explain.
        corpus_dataloader: Data loader of corpus examples to be encoded by `encoder`.
        batch_size: Mini-batch size for loading the corpus representations. This is
            useful when the entire corpus set fails to fit in compute memory.
        device: Optional device to perform encoding on.
    """

    def __init__(
        self,
        encoder: nn.Module,
        corpus_dataloader: DataLoader,
        batch_size: int = 64,
        device: Optional[int] = None,
    ) -> None:
        super().__init__(encoder=encoder, device=device)
        self.corpus_dataloader = corpus_dataloader

        self.corpus_rep = self._encode(self.corpus_dataloader)
        self.corpus_size = self.corpus_rep.size(0)
        self.rep_dim = self.corpus_rep.size(-1)
        self.corpus_rep_dataloader = DataLoader(
            TensorDataset(self.corpus_rep),
            batch_size=batch_size,
            shuffle=False,
        )
