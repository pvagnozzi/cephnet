# heatmap.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: Heatmap-based landmark detection model with encoder-decoder architecture

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from cephnet.config.schema import HeatmapModelConfig
from cephnet.models.backbone import build_backbone


class ConvBnRelu(nn.Sequential):
    """Conv2d → BatchNorm2d → ReLU block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class HeatmapModel(nn.Module):
    """Encoder-decoder heatmap model for cephalometric landmark detection.

    Architecture::

        ResNet encoder  →  3× (ConvTranspose2d + ConvBnRelu)  →  1×1 head

    Input:  ``(B, 3, H, W)`` normalised image  [0, 1]
    Output: ``(B, N, Hh, Hw)`` raw heatmaps where *N* = ``n_landmarks``
    """

    def __init__(self, config: HeatmapModelConfig) -> None:
        super().__init__()
        self.config = config
        encoder, enc_out_ch = build_backbone(config.backbone, pretrained=config.pretrained)
        self.encoder = encoder

        dec_channels = [256, 128, 64]
        decoder_layers: list[nn.Module] = []
        in_ch = enc_out_ch
        for out_ch in dec_channels:
            decoder_layers.extend(
                [
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                    ConvBnRelu(out_ch, out_ch),
                ]
            )
            in_ch = out_ch

        self.decoder = nn.Sequential(*decoder_layers)
        self.head = nn.Conv2d(in_ch, config.n_landmarks, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(B, 3, H, W)`` normalised input image.

        Returns:
            ``(B, N, Hh, Hw)`` heatmap logits, upsampled to ``config.heatmap_size``.
        """
        features = self.encoder(x)
        decoded = self.decoder(features)
        heatmaps = self.head(decoded)

        target_h, target_w = self.config.heatmap_size
        if (heatmaps.shape[2], heatmaps.shape[3]) != (target_h, target_w):
            heatmaps = F.interpolate(
                heatmaps,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
        return cast(torch.Tensor, heatmaps)  # (B, N, Hh, Hw)

    def predict_landmarks(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """Convert heatmaps to landmark coordinates via soft-argmax.

        Args:
            heatmaps: ``(B, N, H, W)`` raw or sigmoid-normalised heatmaps.

        Returns:
            ``(B, N, 2)`` landmark coordinates ``(x, y)`` in heatmap pixel space.
        """
        B, N, H, W = heatmaps.shape
        flat = heatmaps.view(B, N, -1)
        argmax = flat.argmax(dim=-1)
        y = (argmax // W).float()
        x = (argmax % W).float()
        return torch.stack([x, y], dim=-1)  # (B, N, 2)
