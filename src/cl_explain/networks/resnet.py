"""ResNets"""
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50


class PretrainedResNet(nn.Module):
    """
    Pytorch Module for ResNet pre-trained on Imagenet.

    The forward pass can be used with or without applying the final linear head.

    Args:
    ----
        num_layers: Number of layers in the ResNet (18 or 50).
        use_pretrained_head: Whether to use the pre-trained linear head or initialize
            a new linear head.
        num_classes: Number of classes for the new classification task.
    """

    def __init__(
        self, num_layers: int, use_pretrained_head: bool = True, num_classes: int = 1000
    ) -> None:
        super().__init__()
        available_num_layers = [18, 50]
        if num_layers == 18:
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif num_layers == 50:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            raise ValueError(
                f"num_layers = {num_layers} is not one of {available_num_layers}!"
            )

        if use_pretrained_head:
            assert (
                num_classes == 1000
            ), "num_classes should be 1000 when use_pretrained_head = True!"
            head = backbone.fc
        else:
            head = nn.Linear(backbone.fc.in_features, num_classes)
        backbone.fc = nn.Identity()

        self.num_layers = num_layers
        self.use_pretrained_head = use_pretrained_head
        self.num_classes = num_classes
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor, apply_eval_head=False) -> torch.Tensor:
        """
        Forward pass.

        Args:
        ----
            x: A batch of input.
            apply_eval_head: Whether to apply the final linear head.

        Returns
        -------
            A batch of output. If `apply_eval_head = True`, the pre-softmax values are
            returned. If `apply_eval_head = False`, the representations before the
            final output layer are returned.
        """
        x = self.backbone(x)
        if apply_eval_head:
            x = self.head(x)
        return x


class PretrainedResNet18(PretrainedResNet):
    """Pytorch Module for ResNet18 pre-trained on Imagenet."""

    def __init__(
        self, use_pretrained_head: bool = True, num_classes: int = 1000
    ) -> None:
        super().__init__(
            num_layers=18,
            use_pretrained_head=use_pretrained_head,
            num_classes=num_classes,
        )


class PretrainedResNet50(PretrainedResNet):
    """Pytorch Module for ResNet50 pre-trained on Imagenet."""

    def __init__(
        self, use_pretrained_head: bool = True, num_classes: int = 1000
    ) -> None:
        super().__init__(
            num_layers=50,
            use_pretrained_head=use_pretrained_head,
            num_classes=num_classes,
        )
