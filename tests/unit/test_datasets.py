# test_datasets.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-19
# Description: Tests for ISBI2015Dataset, AarizDataset, and dataset factory

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cephnet.datasets.aariz import AarizDataset
from cephnet.datasets.factory import get_dataset
from cephnet.datasets.isbi2015 import ISBI2015Dataset

# All tests use a non-existent root so the dataset falls back to synthetic data.
_N_LM = 19
_IMG_SIZE = (64, 64)


def test_isbi2015_synthetic_train(tmp_path: Path) -> None:
    """Train split has a positive number of synthetic samples."""
    ds = ISBI2015Dataset(root=tmp_path / "nonexistent", split="train")
    assert len(ds) > 0


def test_isbi2015_synthetic_val(tmp_path: Path) -> None:
    """Val split has a positive number of synthetic samples."""
    ds = ISBI2015Dataset(root=tmp_path / "nonexistent", split="val")
    assert len(ds) > 0


def test_isbi2015_synthetic_test(tmp_path: Path) -> None:
    """Test split has a positive number of synthetic samples."""
    ds = ISBI2015Dataset(root=tmp_path / "nonexistent", split="test")
    assert len(ds) > 0


def test_isbi2015_getitem_shape(tmp_path: Path) -> None:
    """__getitem__ returns image tensor (3, H, W) and landmarks (N_LM, 2)."""
    ds = ISBI2015Dataset(
        root=tmp_path / "nonexistent",
        split="train",
        image_size=_IMG_SIZE,
    )
    item = ds[0]

    assert "image" in item
    assert "landmarks" in item
    assert item["image"].shape == (3, _IMG_SIZE[1], _IMG_SIZE[0])
    assert item["landmarks"].shape == (_N_LM, 2)
    assert item["image"].dtype == torch.float32
    assert item["landmarks"].dtype == torch.float32


def test_isbi2015_split_sizes(tmp_path: Path) -> None:
    """Sum of all split sizes equals total expected synthetic samples (32)."""
    root = tmp_path / "nonexistent"
    train_ds = ISBI2015Dataset(root=root, split="train")
    val_ds = ISBI2015Dataset(root=root, split="val")
    test_ds = ISBI2015Dataset(root=root, split="test")

    total = len(train_ds) + len(val_ds) + len(test_ds)
    # Synthetic data: 20 train + 6 val + 6 test = 32
    assert total == 32


def test_factory_unknown_dataset(tmp_path: Path) -> None:
    """get_dataset raises ValueError for an unregistered dataset name."""
    with pytest.raises(ValueError, match="Unknown dataset"):
        get_dataset("totally_unknown_ds", root=tmp_path, split="train")


def test_factory_isbi2015(tmp_path: Path) -> None:
    """get_dataset('isbi2015', ...) returns an ISBI2015Dataset instance."""
    ds = get_dataset(
        "isbi2015",
        root=tmp_path / "nonexistent",
        split="train",
        image_size=_IMG_SIZE,
    )
    assert isinstance(ds, ISBI2015Dataset)
    assert len(ds) > 0


# ---------------------------------------------------------------------------
# AarizDataset — verification-only dataset
# ---------------------------------------------------------------------------

def test_aariz_synthetic_test_has_samples(tmp_path: Path) -> None:
    """Aariz test split has synthetic samples when data is absent."""
    ds = AarizDataset(root=tmp_path / "nonexistent", split="test")
    assert len(ds) == 50


def test_aariz_synthetic_train_is_empty(tmp_path: Path) -> None:
    """Aariz train split is empty — Aariz is verification-only (train_ratio=0)."""
    ds = AarizDataset(root=tmp_path / "nonexistent", split="train")
    assert len(ds) == 0


def test_aariz_synthetic_val_is_empty(tmp_path: Path) -> None:
    """Aariz val split is empty — Aariz is verification-only (val_ratio=0)."""
    ds = AarizDataset(root=tmp_path / "nonexistent", split="val")
    assert len(ds) == 0


def test_aariz_getitem_shape(tmp_path: Path) -> None:
    """AarizDataset __getitem__ returns correct tensor shapes."""
    ds = AarizDataset(root=tmp_path / "nonexistent", split="test", image_size=_IMG_SIZE)
    item = ds[0]
    assert item["image"].shape == (3, _IMG_SIZE[1], _IMG_SIZE[0])
    assert item["landmarks"].shape == (_N_LM, 2)
    assert item["image"].dtype == torch.float32
    assert item["landmarks"].dtype == torch.float32


def test_aariz_and_isbi2015_use_different_roots(tmp_path: Path) -> None:
    """ISBI2015 (training) and Aariz (verification) use distinct dataset roots."""
    isbi = ISBI2015Dataset(root=tmp_path / "isbi2015", split="train")
    aariz = AarizDataset(root=tmp_path / "aariz", split="test")
    assert isbi.root != aariz.root, "Training and verification datasets must have separate roots"


def test_factory_aariz(tmp_path: Path) -> None:
    """get_dataset('aariz', ...) returns an AarizDataset instance with test split."""
    ds = get_dataset("aariz", root=tmp_path / "nonexistent", split="test")
    assert isinstance(ds, AarizDataset)
    assert len(ds) == 50
