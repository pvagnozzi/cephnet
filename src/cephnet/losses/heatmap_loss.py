# heatmap_loss.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: MSE and Adaptive Wing Loss for heatmap-based landmark detection

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_gaussian_heatmaps(
    landmarks: torch.Tensor,
    heatmap_size: tuple[int, int],
    sigma: float = 3.0,
    image_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Generate Gaussian heatmaps from landmark pixel coordinates.

    Args:
        landmarks: ``(B, N, 2)`` coordinates ``(x, y)`` in image pixel space.
        heatmap_size: ``(H, W)`` output heatmap dimensions.
        sigma: Gaussian standard deviation in heatmap pixels.
        image_size: ``(H, W)`` original image dimensions used to scale
                    landmark coordinates into heatmap space. When *None*,
                    coordinates are assumed to already be in heatmap space.

    Returns:
        ``(B, N, H, W)`` Gaussian heatmaps with values in ``[0, 1]``.
    """
    B, N, _ = landmarks.shape
    H, W = heatmap_size
    device = landmarks.device

    if image_size is not None:
        scale_x = W / image_size[1]
        scale_y = H / image_size[0]
        scaled = landmarks.clone()
        scaled[..., 0] = landmarks[..., 0] * scale_x
        scaled[..., 1] = landmarks[..., 1] * scale_y
    else:
        scaled = landmarks

    y_range = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
    x_range = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W)

    cx = scaled[..., 0].view(B, N, 1, 1)
    cy = scaled[..., 1].view(B, N, 1, 1)

    heatmaps = torch.exp(
        -((x_range - cx) ** 2 + (y_range - cy) ** 2) / (2 * sigma**2)
    )
    return heatmaps  # (B, N, H, W)


class HeatmapLoss(nn.Module):
    """MSE loss between predicted and target Gaussian heatmaps."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute element-wise MSE.

        Args:
            pred: ``(B, N, H, W)`` predicted heatmaps.
            target: ``(B, N, H, W)`` ground-truth Gaussian heatmaps.

        Returns:
            Scalar loss.
        """
        return F.mse_loss(pred, target)


class AdaptiveWingLoss(nn.Module):
    """Adaptive Wing Loss for robust heatmap regression.

    Reference:
        Wang et al., *Adaptive Wing Loss for Robust Face Alignment via
        Heatmap Regression*, ICCV 2019.
    """

    def __init__(
        self,
        omega: float = 14.0,
        theta: float = 0.5,
        epsilon: float = 1.0,
        alpha: float = 2.1,
    ) -> None:
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Adaptive Wing loss.

        Args:
            pred: ``(B, N, H, W)`` predicted heatmaps.
            target: ``(B, N, H, W)`` ground-truth Gaussian heatmaps.

        Returns:
            Scalar mean loss.
        """
        diff = (target - pred).abs()
        A = (
            self.omega
            * (1.0 / (1.0 + (self.theta / self.epsilon) ** (self.alpha - target)))
            * (self.alpha - target)
            * ((self.theta / self.epsilon) ** (self.alpha - target - 1.0))
            * (1.0 / self.epsilon)
        )
        C = self.theta * A - self.omega * torch.log(
            1.0 + (self.theta / self.epsilon) ** (self.alpha - target)
        )

        loss = torch.where(
            diff < self.theta,
            self.omega * torch.log(1.0 + (diff / self.epsilon) ** (self.alpha - target)),
            A * diff - C,
        )
        return loss.mean()
