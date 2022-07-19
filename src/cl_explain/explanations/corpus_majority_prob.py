"""Explanation behavior using predicted probability of corpus prediction majority."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class CorpusMajorityProb(nn.Module):
    """
    Class to compute an explicand's predicted probability of corpus prediction majority.

    Args:
    ----
        encoder: Encoder module to explain. It should return predicted output with
            `apply_eval_head=True` in `encoder.forward()`.
        corpus_dataloader: Data loader of corpus examples, with shape (corpus_size, *),
            where * indicates the input dimensions of `encoder`.
    """

    def __init__(
        self,
        encoder: nn.Module,
        corpus_dataloader: DataLoader,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.corpus_dataloader = corpus_dataloader
        self.corpus_majority_pred = self._find_corpus_majority_pred()

    def forward(self, explicand: torch.Tensor) -> torch.Tensor:
        pred_prob = self.encoder(explicand, apply_eval_head=True).softmax(dim=-1)
        return pred_prob.select(dim=-1, index=self.corpus_majority_pred)

    def _find_corpus_majority_pred(self) -> torch.Tensor:
        encoder_device = [param.device for param in self.encoder.parameters()][0]
        pred = []
        for x, _ in self.corpus_dataloader:
            x = x.to(encoder_device)
            x = self.encoder(x, apply_eval_head=True).argmax(dim=-1).detach().cpu()
            pred.append(x)
        pred = torch.cat(pred)
        return pred.mode()[0].to(encoder_device)
