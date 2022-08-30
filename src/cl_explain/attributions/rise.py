"""
RISE: Randomized Input Sampling for Explanation of Black-box Models proposed by Petsiuk
et al. 2018.
"""
from typing import Callable, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import Attribution


class RISE(Attribution):
    """
    RISE for images proposed by Petsiuk et al. 2018.

    Args:
    ----
        forward_func: A function that returns a one-dimensional output for each input
            sample.
    """

    def __init__(self, forward_func: Union[Callable, nn.Module]) -> None:
        super().__init__(forward_func=forward_func)

    def attribute(
        self,
        inputs: torch.Tensor,
        grid_shape: Tuple[int, int],
        baselines: Union[float, torch.Tensor] = 0,
        mask_prob: float = 0.5,
        n_samples: int = 50,
        normalize_by_mask_prob: bool = True,
    ) -> torch.Tensor:
        """
        Attribute feature importance to each pixel.

        Args:
        ----
            inputs: A batch of images with shape
                `(batch_size, channel_size, img_h, img_w)`.
            grid_shape: A tuple of integers indicating the smaller grid shape
                `(grid_h, grid_w)` that's upsampled as a mask.
            baselines: Values for replacing the original inputs when masked.
            mask_prob: Probability of masking a pixel in the smaller binary mask.
            n_samples: Number of masks per input image.
            normalize_by_mask_prob: Whether to normalize each RISE attribution score
                by `mask_prob` as in the original paper.

        Returns
        -------
            A tensor with shape `(batch_size, channel_size, img_h, img_w)` for the
            importance score of each pixel. The values across `channel_size` are all
            the same because importance scores are computed at the pixel level, not the
            pixel-channel level. The dimension `channel_size` is retained for keeping
            the standard that attribution output should have the same shape as the
            input.
        """
        batch_size, channel_size, img_h, img_w = inputs.shape
        device = inputs.device
        scores = torch.zeros(batch_size, channel_size, img_h, img_w).to(device)
        for i in range(n_samples):
            masks = self.generate_mask(
                img_h=img_h,
                img_w=img_w,
                grid_h=grid_shape[0],
                grid_w=grid_shape[1],
                mask_prob=mask_prob,
                n_masks=batch_size,
            ).to(device)
            score = self.forward_func(inputs * masks + baselines * (1 - masks)).detach()
            if i == 0:  # Check output dimension on the first iteration.
                assert (
                    len(score.shape) == 1
                ), "RISE.forward_func needs to return a one-dimensional output!"
            score = score.view(batch_size, 1, 1, 1) * masks
            scores += score
        scores /= n_samples
        if normalize_by_mask_prob:
            scores /= mask_prob
        return scores

    @staticmethod
    def generate_mask(
        img_h: int,
        img_w: int,
        grid_h: int,
        grid_w: int,
        mask_prob: float = 0.5,
        n_masks: int = 50,
    ) -> torch.Tensor:
        """
        Generate masks for dimming input images.

        The mask generation process is summarized as the following.
        1. Sample a binary mask of size `(grid_h, grid_w)`.
        2. Upsample the binary mask through bilinear interpolation to size
            `((grid_h + 1) * (img_h // grid_h), (grid_w + 1) * (img_w // grid_w))`.
        3. Randomly crop a contiguous area of size `(img_h, img_w)` to form the final
            mask.

        Args:
        ----
            img_h: Original image height.
            img_w: Original image width.
            grid_h: Smaller binary mask height.
            grid_w: Smaller binary mask width.
            mask_prob: Probability of masking a pixel in the smaller binary mask.
            n_masks: Number of independent masks to generate.

        Returns
        -------
            A tensor of size `(n_masks, 1, img_h, img_w)` of independently sampled
            masks.
        """
        grid_h_pixel_size = img_h // grid_h
        grid_w_pixel_size = img_w // grid_w
        mask = (torch.rand(n_masks, 1, grid_h, grid_w) < mask_prob) * 1.0
        mask = F.interpolate(
            mask,
            size=((grid_h + 1) * grid_h_pixel_size, (grid_w + 1) * grid_w_pixel_size),
            mode="bilinear",
            align_corners=False,
        )
        shift_h = np.random.randint(grid_h_pixel_size)
        shift_w = np.random.randint(grid_w_pixel_size)
        mask = mask[:, :, shift_h : (shift_h + img_h), shift_w : (shift_w + img_w)]
        return mask
