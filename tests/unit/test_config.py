# test_config.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: Tests for Pydantic config schema and YAML loader

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from cephnet.config.loader import load_config
from cephnet.config.schema import (
    CephnetConfig,
    DatasetConfig,
    HeatmapModelConfig,
    RegressionModelConfig,
)


def test_dataset_config_valid() -> None:
    """DatasetConfig accepts valid parameters and stores them correctly."""
    cfg = DatasetConfig(
        name="isbi2015",
        root=Path("/data"),
        n_landmarks=19,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
    )
    assert cfg.name == "isbi2015"
    assert cfg.n_landmarks == 19
    assert cfg.pixel_spacing_mm == pytest.approx(0.1)


def test_dataset_config_split_ratios_invalid() -> None:
    """Split ratios that do not sum to 1.0 raise a ValidationError."""
    with pytest.raises(ValidationError, match="sum to 1.0"):
        DatasetConfig(
            name="isbi2015",
            root=Path("/data"),
            n_landmarks=19,
            train_ratio=0.5,
            val_ratio=0.3,
            test_ratio=0.3,  # total = 1.1
        )


def test_heatmap_model_config_valid() -> None:
    """HeatmapModelConfig is created with expected defaults."""
    cfg = HeatmapModelConfig(
        n_landmarks=19,
        backbone="resnet18",
        pretrained=False,
    )
    assert cfg.type == "heatmap"
    assert cfg.n_landmarks == 19
    assert cfg.backbone == "resnet18"
    assert cfg.heatmap_size == (128, 128)


def test_regression_model_config_valid() -> None:
    """RegressionModelConfig is created with expected defaults."""
    cfg = RegressionModelConfig(
        n_landmarks=19,
        backbone="resnet18",
        pretrained=False,
    )
    assert cfg.type == "regression"
    assert cfg.n_landmarks == 19
    assert cfg.hidden_dim == 512


def test_cephnet_config_landmark_mismatch() -> None:
    """CephnetConfig raises ValidationError when model and dataset n_landmarks differ."""
    with pytest.raises(ValidationError, match="n_landmarks"):
        CephnetConfig(
            dataset=DatasetConfig(name="isbi2015", root=Path("/data"), n_landmarks=19),
            model=HeatmapModelConfig(n_landmarks=10, pretrained=False),
        )


def test_load_config_valid(tmp_config: Path) -> None:
    """load_config returns a CephnetConfig from a well-formed YAML file."""
    config = load_config(tmp_config)
    assert isinstance(config, CephnetConfig)
    assert config.dataset.n_landmarks == 19
    assert config.dataset.name == "isbi2015"
    assert config.training.epochs == 2


def test_load_config_file_not_found(tmp_path: Path) -> None:
    """load_config raises FileNotFoundError when the path does not exist."""
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent_config.yaml")


def test_load_config_invalid_yaml(tmp_path: Path) -> None:
    """load_config raises an exception for syntactically invalid YAML."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("dataset: {name: isbi2015, unclosed: [bracket")
    with pytest.raises(Exception):
        load_config(bad)
