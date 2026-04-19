# regression_loss.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: Smooth L1 / Wing losses for coordinate regression models

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionLoss(nn.Module):
    """Smooth L1 (Huber) loss for landmark coordinate regression.

    Args:
        beta: Transition point between quadratic and linear behaviour.
              Equivalent to the ``beta`` parameter in
              :func:`torch.nn.functional.smooth_l1_loss`.
    """

    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Smooth L1 loss.

        Args:
            pred: ``(B, N, 2)`` predicted normalised coordinates ``[0, 1]``.
            target: ``(B, N, 2)`` ground-truth normalised coordinates.

        Returns:
            Scalar loss.
        """
        return F.smooth_l1_loss(pred, target, beta=self.beta)


class WingLoss(nn.Module):
    """Wing Loss for face / landmark alignment.

    Reference:
        Feng et al., *Wing Loss for Robust Facial Landmark Localisation with
        Convolutional Neural Networks*, CVPR 2018.

    Args:
        w: Width of the non-linear region.
        epsilon: Curvature of the non-linear region.
    """

    def __init__(self, w: float = 10.0, epsilon: float = 2.0) -> None:
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        # Pre-compute the constant C so that the loss is continuous at |x| = w
        self.C: float = w - w * torch.log(  # type: ignore[assignment]
            torch.tensor(1.0 + w / epsilon)
        ).item()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Wing loss.

        Args:
            pred: ``(B, N, 2)`` predicted coordinates.
            target: ``(B, N, 2)`` ground-truth coordinates.

        Returns:
            Scalar mean loss.
        """
        diff = (pred - target).abs()
        loss = torch.where(
            diff < self.w,
            self.w * torch.log(1.0 + diff / self.epsilon),
            diff - self.C,
        )
        return loss.mean()
