"""Utility functions"""

import torch
import torch.nn as nn


def make_superpixel_map(
    img_h: int, img_w: int, superpixel_h: int, superpixel_w: int
) -> torch.Tensor:
    """
    Divides an image into superpixels and maps each pixel to a superpixel.

    For example, a 4 x 4 image can be divided into 4 superpixels, each with a superpixel
    height of 2 and a superpixel width of 2. The corresponding superpixel map is
    +---+---+---+---+
    | 0 | 0 | 1 | 1 |
    +---+---+---+---+
    | 0 | 0 | 1 | 1 |
    +---+---+---+---+
    | 2 | 2 | 3 | 3 |
    +---+---+---+---+
    | 2 | 2 | 3 | 3 |
    +---+---+---+---+
    where the integers are superpixel ids.

    Args:
    ----
        img_h: Original image height.
        img_w: Original image width.
        superpixel_h: Height of each superpixel.
        superpixel_w: Width of each superpixel.

    Returns
    -------
        A tensor with shape `(1, 1, img_h, img_w)`. Each pixel contains the superpixel
        id that the pixel is part of.
    """
    num_superpixels_h, remainder_h = divmod(img_h, superpixel_h)
    num_superpixels_w, remainder_w = divmod(img_w, superpixel_w)
    assert (
        remainder_h == 0
    ), f"img_h={img_h} is not divisible by superpixel_h={superpixel_h}!"
    assert (
        remainder_w == 0
    ), f"img_w={img_w} is not divisible by superpixel_w={superpixel_w}!"

    num_superpixels = num_superpixels_h * num_superpixels_w
    superpixel_map = (
        torch.arange(num_superpixels)
        .view(1, 1, num_superpixels_h, num_superpixels_w)
        .float()
    )
    upsample = nn.Upsample(scale_factor=(superpixel_h, superpixel_w), mode="nearest")
    superpixel_map = upsample(superpixel_map).long()
    return superpixel_map
