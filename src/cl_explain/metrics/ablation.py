"""Ablation metrics for evaluating feature attribution methods."""

from typing import Callable, List, Tuple, Union

import torch
import torch.nn as nn


class ImageAblation:
    """
    Class for evaluating image feature attribution methods with ablation curve.

    Args:
    ----
        model_list: List of models that return outputs for plotting the ablation curves.
            A model requires only an ablated image as an input.
        measure_list: List of measures for the difference between original explicand
            output and ablated explicand output, for plotting the ablation curves.
            Different from a model, a measure requires both the original and ablated
            image as inputs.
        img_h: Model input image height.
        img_w: Model input image width.
        superpixel_h: Superpixels can be the ablated features instead of individual
            pixels. This determines the superpixel height.
        superpixel_w: Superpixel width.
        num_steps: Number of ablation steps. Roughly `feature_attr_size / num_steps`
        features are additionally ablated in each iteration.
    """

    def __init__(
        self,
        model_list: List[Union[Callable, nn.Module]],
        measure_list: List[Union[Callable, nn.Module]],
        img_h: int,
        img_w: int,
        superpixel_h: int = 1,
        superpixel_w: int = 1,
        num_steps: int = 50,
    ) -> None:
        self.model_list = model_list
        self.measure_list = measure_list
        self.img_h = img_h
        self.img_w = img_w
        self.superpixel_h = superpixel_h
        self.superpixel_w = superpixel_w
        self.num_steps = num_steps
        self.num_models = len(model_list)
        self.num_measures = len(measure_list)

        attr_h, remainder_h = divmod(img_h, superpixel_h)
        attr_w, remainder_w = divmod(img_w, superpixel_w)
        assert (
            remainder_h == 0
        ), f"img_h={img_h} is not divisible by superpixel_h={superpixel_h}!"
        assert (
            remainder_w == 0
        ), f"img_w={img_w} is not divisible by superpixel_w={superpixel_w}!"

        self.attr_h = attr_h
        self.attr_w = attr_w
        self.feature_attr_size = attr_h * attr_w
        self.step_sizes = self._get_step_sizes()

        if superpixel_h == 1 and superpixel_w == 1:
            self.mask_upsampler = None
        else:
            mask_upsampler = nn.Upsample(
                scale_factor=(superpixel_h, superpixel_w), mode="nearest"
            )
            self.mask_upsampler = mask_upsampler

    def evaluate(
        self,
        explicand: torch.Tensor,
        attribution: torch.Tensor,
        baseline: torch.Tensor,
        kind: str = "insertion",
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
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
            model_curves: A list containing model output tensors for all explicands
                after the ablation steps, each tensor with shape
                `(batch_size, *, num_steps + 1)`, where * denotes the model output
                dimension.
            measure_curves: A list containing measure output tensors for all explicands
                after the ablation steps.
            total_num_features: A tensor containing the total number of ablated
                features for each step, with size `num_steps + 1`.
        """
        available_kinds = ["insertion", "deletion"]

        batch_size = attribution.size(0)
        assert batch_size == explicand.size(
            0
        ), "explicand and attribution should have the same batch size!"

        attr_h, attr_w = attribution.size(-2), attribution.size(-1)
        assert attr_h == self.attr_h, (
            f"attribution height = {attr_h} is not the same as "
            f"image height / superpixel height = {self.attr_h}!"
        )
        assert attr_w == self.attr_w, (
            f"attribution width = {attr_w} is not the same as "
            f"image width / superpixel width = {self.attr_w}!"
        )

        explicand_h, explicand_w = explicand.size(-2), explicand.size(-1)
        assert explicand_h == self.img_h, (
            f"explicand height = {explicand_h} is not the same as "
            f"expected image height = {self.img_h}!"
        )
        assert explicand_w == self.img_w, (
            f"explicand width = {explicand_w} is not the same as "
            f"expected image width = {self.img_w}!"
        )

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

        model_curves = [[] for _ in range(self.num_models)]
        measure_curves = [[] for _ in range(self.num_measures)]
        total_num_features = []
        num_features = 0
        for step_size in self.step_sizes:
            num_features += step_size
            for i in range(batch_size):
                flat_mask[i, sorted_idx[i, :num_features]] = fill_val
            mask = flat_mask.view(attribution.shape)
            if self.mask_upsampler is not None:
                mask = self.mask_upsampler(mask)
            masked_explicand = explicand * mask + baseline * (1 - mask)
            for j in range(self.num_models):
                model_output = self.model_list[j](masked_explicand)
                model_curves[j].append(model_output.detach().cpu())
            for k in range(self.num_measures):
                measure_output = self.measure_list[k](
                    original_explicand=explicand, modified_explicand=masked_explicand
                )
                measure_curves[k].append(measure_output.detach().cpu())
            total_num_features.append(num_features)
        model_curves = [torch.stack(curve, dim=-1) for curve in model_curves]
        measure_curves = [torch.stack(curve, dim=-1) for curve in measure_curves]
        return model_curves, measure_curves, torch.Tensor(total_num_features)

    def _get_step_sizes(self):
        step_size, remainder = divmod(self.feature_attr_size, self.num_steps)
        step_size_list = [step_size] * self.num_steps
        for i in range(remainder):
            step_size_list[i] += 1
        return [0] + step_size_list  # Add a step size of zero to get initial output.


def compute_auc(curve: torch.Tensor, num_features: torch.Tensor) -> torch.Tensor:
    """
    Compute AUCs for insertion or deletion curves.

    Args:
    ----
        curve: Insertion or deletion curves with shape `(batch_size, num_steps)`.
        num_features: Number of inserted or deleted features with size `num_steps`.

    Returns
    -------
        A tensor containing the AUC for each curve, with size `batch_size`.
    """
    num_features_dim = len(num_features.shape)
    if num_features_dim == 1:
        prop_features = num_features / num_features.max()
    elif num_features_dim == 2:
        prop_features = num_features / num_features.max(dim=1)[0].unsqueeze(1)
    else:
        raise ValueError(f"num_features has {num_features_dim} > 2 dimensions!")
    return torch.trapezoid(y=curve, x=prop_features, dim=-1)
