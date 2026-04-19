# backbone.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: CNN backbone factory using torchvision models

from __future__ import annotations

import torch.nn as nn
import torchvision.models as tv_models


def build_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Return ``(encoder, output_channels)``.

    The returned encoder produces feature maps at 1/32 of the input spatial
    resolution (after strided conv1 + maxpool + layer1-4 of ResNet).

    Args:
        name: Backbone variant — one of "resnet18", "resnet34", "resnet50",
              "resnet101".
        pretrained: When *True*, load ImageNet-1K pretrained weights.

    Returns:
        A tuple of (encoder nn.Sequential, number of output channels).

    Raises:
        ValueError: When ``name`` is not a supported backbone.
    """
    weights_id = "IMAGENET1K_V1" if pretrained else None

    match name:
        case "resnet18":
            model = tv_models.resnet18(weights=weights_id)
            out_channels = 512
        case "resnet34":
            model = tv_models.resnet34(weights=weights_id)
            out_channels = 512
        case "resnet50":
            model = tv_models.resnet50(weights=weights_id)
            out_channels = 2048
        case "resnet101":
            model = tv_models.resnet101(weights=weights_id)
            out_channels = 2048
        case _:
            raise ValueError(
                f"Unsupported backbone: {name!r}. "
                "Supported: resnet18, resnet34, resnet50, resnet101"
            )

    # Strip the classification head (avgpool + fc) — keep feature extractor only
    encoder = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
    )
    return encoder, out_channels
