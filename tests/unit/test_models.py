# test_models.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: Tests for HeatmapModel, RegressionModel and build_model factory

from __future__ import annotations

import pytest
import torch

from cephnet.config.schema import HeatmapModelConfig, RegressionModelConfig
from cephnet.models.heatmap import HeatmapModel
from cephnet.models.regression import RegressionModel, build_model

# Shared small config for fast tests (resnet18, no pretrained weights download)
_HM_CFG = HeatmapModelConfig(
    n_landmarks=19,
    backbone="resnet18",
    pretrained=False,
    heatmap_size=(16, 16),
)
_REG_CFG = RegressionModelConfig(
    n_landmarks=19,
    backbone="resnet18",
    pretrained=False,
)

# Small synthetic batch: (B=2, 3, 64, 64)
_BATCH = torch.rand(2, 3, 64, 64)


def test_heatmap_model_output_shape() -> None:
    """HeatmapModel produces (B, N, Hh, Hw) heatmaps at the configured size."""
    model = HeatmapModel(_HM_CFG)
    model.eval()
    with torch.no_grad():
        out = model(_BATCH)
    assert out.shape == (2, 19, 16, 16)


def test_regression_model_output_shape() -> None:
    """RegressionModel produces (B, N, 2) coordinates all in [0, 1]."""
    model = RegressionModel(_REG_CFG)
    model.eval()
    with torch.no_grad():
        out = model(_BATCH)
    assert out.shape == (2, 19, 2)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_build_model_heatmap() -> None:
    """build_model returns a HeatmapModel when given HeatmapModelConfig."""
    model = build_model(_HM_CFG)
    assert isinstance(model, HeatmapModel)


def test_build_model_regression() -> None:
    """build_model returns a RegressionModel when given RegressionModelConfig."""
    model = build_model(_REG_CFG)
    assert isinstance(model, RegressionModel)


def test_heatmap_model_forward_no_crash(synthetic_batch: dict) -> None:
    """HeatmapModel forward pass with synthetic batch does not raise."""
    model = HeatmapModel(_HM_CFG)
    model.eval()
    with torch.no_grad():
        out = model(synthetic_batch["image"])
    assert out.shape[0] == synthetic_batch["image"].shape[0]
    assert out.ndim == 4


def test_regression_model_forward_no_crash(synthetic_batch: dict) -> None:
    """RegressionModel forward pass with synthetic batch does not raise."""
    model = RegressionModel(_REG_CFG)
    model.eval()
    with torch.no_grad():
        out = model(synthetic_batch["image"])
    assert out.shape[0] == synthetic_batch["image"].shape[0]
    assert out.ndim == 3
