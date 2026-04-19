# regression.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: Coordinate regression model for landmark detection

from __future__ import annotations

import torch
import torch.nn as nn

from cephnet.config.schema import HeatmapModelConfig, RegressionModelConfig
from cephnet.models.backbone import build_backbone


class RegressionModel(nn.Module):
    """ResNet backbone + FC head for direct landmark coordinate regression.

    Input:  ``(B, 3, H, W)`` normalised image  [0, 1]
    Output: ``(B, N, 2)`` predicted coordinates normalised to ``[0, 1]``
    """

    def __init__(self, config: RegressionModelConfig) -> None:
        super().__init__()
        self.config = config
        encoder, enc_out_ch = build_backbone(config.backbone, pretrained=config.pretrained)
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enc_out_ch, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.n_landmarks * 2),
            nn.Sigmoid(),  # constrain outputs to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(B, 3, H, W)`` normalised input image.

        Returns:
            ``(B, N, 2)`` predicted coordinates in ``[0, 1]``.
        """
        features = self.encoder(x)
        pooled = self.pool(features)
        out = self.head(pooled)
        return out.view(-1, self.config.n_landmarks, 2)  # (B, N, 2)


def build_model(config: HeatmapModelConfig | RegressionModelConfig) -> nn.Module:
    """Factory: instantiate the correct model from a config object.

    Args:
        config: Either a :class:`~cephnet.config.schema.HeatmapModelConfig`
                or a :class:`~cephnet.config.schema.RegressionModelConfig`.

    Returns:
        Instantiated :class:`torch.nn.Module`.
    """
    from cephnet.models.heatmap import HeatmapModel

    if isinstance(config, HeatmapModelConfig):
        return HeatmapModel(config)
    return RegressionModel(config)
