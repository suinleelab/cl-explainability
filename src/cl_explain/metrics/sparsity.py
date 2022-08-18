"""Sparsity metrics for feature attribution scores."""

import torch


def compute_gini_index(attribution: torch.Tensor) -> torch.Tensor:
    """
    Compute the Gini Index for a batch of attribution scores.

    The formula for the Gini Index can be found at
    https://en.wikipedia.org/wiki/Gini_coefficient#Alternative_expressions.

    Args:
    ----
        attribution: A batch of attribution scores with shape `(batch_size, *)`, where
            * indicates the feature dimension sizes.
    Returns
    -------
        A tensor of Gini Index values for the attribution scores, with size
            `batch_size`.
    """
    attribution, _ = attribution.flatten(start_dim=1).sort()
    feature_size = attribution.size(-1)
    gini = 2 * (
        (torch.arange(feature_size).to(attribution.device) + 1) * attribution
    ).sum(dim=1)
    gini /= feature_size * attribution.sum(dim=1)
    gini -= (feature_size + 1) / feature_size
    return gini
