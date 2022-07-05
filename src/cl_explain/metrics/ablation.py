"""Ablation metrics for evaluating feature attribution methods."""

from typing import Callable, Tuple, Union

import torch
import torch.nn as nn


class ImageAblation:
    """
    Class for evaluating image feature attribution methods with ablation curve.

    Args:
    ----
        model: Model that returns output for plotting the ablation curve.
        feature_attr_size: Total number of feature attribution scores per image.
        num_steps: Number of ablation steps. Roughly `feature_attr_size / num_steps`
        features are additionally ablated in each iteration.
    """

    def __init__(
        self,
        model: Union[Callable, nn.Module],
        feature_attr_size: int,
        num_steps: int = 10,
    ) -> None:
        self.model = model
        self.num_steps = num_steps
        self.feature_attr_size = feature_attr_size
        self.step_sizes = self._get_step_sizes()

    def evaluate(
        self,
        explicand: torch.Tensor,
        attribution: torch.Tensor,
        baseline: torch.Tensor,
        kind: str = "insertion",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates feature attribution scores with insertion/deletion ablation metric.

        Args:
        ----
            explicand: A batch of image explicands, with shape
                `(batch_size, channel_size, width, height)`.
            attribution: A batch of corresponding feature attributions, with shape
                `(batch_size, attribution_size_per_pixel, width, height)`.
            baseline: Baseline image(s) for values of removed pixels, with shape that
                can be broadcasted for `explicand`.
            kind: Ablation kind, either `"insertion"` or `"deletion"`.

        Returns
        -------
            A tuple of two tensors. The first one contains model outputs for all
            explicands after each ablation step, with shape
            `(batch_size, *, num_steps + 1)`, where * denotes the model output
            dimension. The second one contains the total number of ablated features for
            each step, with size `num_steps + 1`.
        """
        # TODO: Include option for upsampling the mask when each attribution score
        # corresponds to a superpixel instead of a single pixel.
        available_kinds = ["insertion", "deletion"]

        assert attribution.size(0) == explicand.size(
            0
        ), "explicand and attribution should have the same batch size!"
        batch_size = attribution.size(0)
        flat_attribution = attribution.view(batch_size, -1)
        sorted_idx = flat_attribution.argsort(descending=True, dim=-1)

        if kind == "insertion":
            flat_mask = torch.zeros(flat_attribution.shape).to(explicand.device)
            fill_val = 1
        elif kind == "deletion":
            flat_mask = torch.ones(flat_attribution.shape).to(explicand.device)
            fill_val = 0
        else:
            raise ValueError(f"kind={kind} should be one of {available_kinds}!")

        curve = []
        total_num_features = []
        num_features = 0
        for step_size in self.step_sizes:
            num_features += step_size
            for i in range(batch_size):
                flat_mask[i, sorted_idx[i, :num_features]] = fill_val

            mask = flat_mask.view(attribution.shape)
            masked_explicand = explicand * mask + baseline * (1 - mask)
            output = self.model(masked_explicand)
            curve.append(output.detach().cpu())
            total_num_features.append(num_features)

        return torch.stack(curve, dim=-1), torch.Tensor(total_num_features)

    def _get_step_sizes(self):
        step_size, remainder = divmod(self.feature_attr_size, self.num_steps)
        step_size_list = [step_size] * self.num_steps
        for i in range(remainder):
            step_size_list[i] += 1
        return [0] + step_size_list  # Add a step size of zero to get initial output.
