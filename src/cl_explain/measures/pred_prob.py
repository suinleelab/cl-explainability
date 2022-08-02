"""Predicted probability as a measure for evaluating feature importance."""

import torch
import torch.nn as nn


class PredProb(nn.Module):
    """
    Module class for measuring predicted probability after explicand modifications.

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
            For each sample, the predicted probability of the modified explicand for
            the predicted class of the original explicand.
        """
        original_pred = self.encoder(original_explicand, apply_eval_head=True).argmax(
            dim=-1
        )
        modified_pred_prob = self.encoder(
            modified_explicand, apply_eval_head=True
        ).softmax(dim=-1)
        modified_pred_prob = modified_pred_prob.gather(
            dim=-1, index=original_pred.unsqueeze(-1)
        )
        return modified_pred_prob.squeeze(-1)
