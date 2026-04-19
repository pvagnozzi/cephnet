# test_losses.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: Tests for HeatmapLoss, generate_gaussian_heatmaps and RegressionLoss

from __future__ import annotations

import pytest
import torch

from cephnet.losses.heatmap_loss import HeatmapLoss, generate_gaussian_heatmaps
from cephnet.losses.regression_loss import RegressionLoss

# ---------------------------------------------------------------------------
# HeatmapLoss
# ---------------------------------------------------------------------------


def test_heatmap_loss_zero() -> None:
    """HeatmapLoss is ~0 when pred and target are identical tensors."""
    criterion = HeatmapLoss()
    x = torch.rand(2, 19, 16, 16)
    loss = criterion(x, x)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_heatmap_loss_positive() -> None:
    """HeatmapLoss is strictly positive when pred != target."""
    criterion = HeatmapLoss()
    pred = torch.zeros(2, 19, 16, 16)
    target = torch.ones(2, 19, 16, 16)
    loss = criterion(pred, target)
    assert loss.item() > 0.0


# ---------------------------------------------------------------------------
# generate_gaussian_heatmaps
# ---------------------------------------------------------------------------


def test_generate_gaussian_heatmaps_shape() -> None:
    """Output tensor has shape (B, N, H, W)."""
    B, N = 2, 19
    H, W = 16, 16
    landmarks = torch.rand(B, N, 2) * 16.0  # coords already in heatmap space
    heatmaps = generate_gaussian_heatmaps(landmarks, heatmap_size=(H, W), sigma=2.0)
    assert heatmaps.shape == (B, N, H, W)


def test_generate_gaussian_heatmaps_peak() -> None:
    """Gaussian peak is at the pixel nearest to the landmark coordinate."""
    lx, ly = 8.0, 6.0
    landmarks = torch.tensor([[[lx, ly]]])  # (1, 1, 2)
    heatmaps = generate_gaussian_heatmaps(landmarks, heatmap_size=(16, 16), sigma=1.5)

    flat_idx = int(heatmaps[0, 0].argmax().item())
    peak_y = flat_idx // 16
    peak_x = flat_idx % 16

    assert peak_x == round(lx)
    assert peak_y == round(ly)


def test_generate_gaussian_heatmaps_with_image_size() -> None:
    """image_size argument scales landmark coords from image space to heatmap space."""
    # Landmark at (256, 256) in a 512x512 image → should map to (8, 8) in 16x16 heatmap
    landmarks = torch.tensor([[[256.0, 256.0]]])  # (1, 1, 2)
    heatmaps = generate_gaussian_heatmaps(
        landmarks,
        heatmap_size=(16, 16),
        sigma=1.0,
        image_size=(512, 512),
    )
    flat_idx = int(heatmaps[0, 0].argmax().item())
    peak_y = flat_idx // 16
    peak_x = flat_idx % 16
    assert peak_x == 8
    assert peak_y == 8


# ---------------------------------------------------------------------------
# RegressionLoss
# ---------------------------------------------------------------------------


def test_regression_loss_zero() -> None:
    """RegressionLoss is ~0 when pred and target are identical tensors."""
    criterion = RegressionLoss()
    x = torch.rand(2, 19, 2)
    loss = criterion(x, x)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_regression_loss_positive() -> None:
    """RegressionLoss is strictly positive when pred != target."""
    criterion = RegressionLoss()
    pred = torch.zeros(2, 19, 2)
    target = torch.ones(2, 19, 2)
    loss = criterion(pred, target)
    assert loss.item() > 0.0
