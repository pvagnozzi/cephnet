# conftest.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: Shared pytest fixtures for cephnet test suite

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    """Create a minimal valid config YAML for testing."""
    config = {
        "dataset": {
            "name": "isbi2015",
            "root": str(tmp_path / "data"),
            "image_size": [64, 64],
            "n_landmarks": 19,
            "pixel_spacing_mm": 0.1,
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
        },
        "model": {
            "type": "heatmap",
            "backbone": "resnet18",
            "pretrained": False,
            "n_landmarks": 19,
            "heatmap_sigma": 2.0,
            "heatmap_size": [16, 16],
        },
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "scheduler": {"type": "cosine", "params": {"T_max": 2}},
            "mixed_precision": False,
            "gradient_clip_norm": 1.0,
            "early_stopping": {"enabled": False, "patience": 5, "min_delta": 0.0001},
            "seed": 42,
            "experiment_name": "test_experiment",
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "export_dir": str(tmp_path / "exports"),
            "num_workers": 0,
        },
        "logging": {
            "level": "DEBUG",
            "console": True,
            "file": False,
            "log_dir": str(tmp_path / "logs"),
            "log_interval_steps": 1,
            "save_metrics_every_n_epochs": 1,
        },
        "augmentation": {
            "enabled": False,
            "p_horizontal_flip": 0.0,
            "p_rotation": 0.0,
            "rotation_limit": 5.0,
            "p_brightness_contrast": 0.0,
            "p_gaussian_noise": 0.0,
            "p_elastic_transform": 0.0,
        },
    }
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml.dump(config))
    return config_path


@pytest.fixture
def n_landmarks() -> int:
    return 19


@pytest.fixture
def image_size() -> tuple[int, int]:
    return (64, 64)


@pytest.fixture
def batch_size() -> int:
    return 2


@pytest.fixture
def synthetic_batch(
    n_landmarks: int,
    image_size: tuple[int, int],
    batch_size: int,
) -> dict:
    """Create a synthetic batch of images and landmarks."""
    H, W = image_size
    return {
        "image": torch.rand(batch_size, 3, H, W),
        "landmarks": torch.rand(batch_size, n_landmarks, 2) * torch.tensor([W, H]).float(),
        "image_path": [f"synthetic/test_{i}.png" for i in range(batch_size)],
    }
