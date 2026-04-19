# test_augmentations.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: Tests for augmentation and validation pipeline builders

from __future__ import annotations

import numpy as np
import pytest

from cephnet.augmentations.pipeline import build_augmentation_pipeline, build_val_pipeline
from cephnet.config.schema import AugmentationConfig

_IMG_SIZE = (64, 64)  # (W, H)


def _make_image(h: int = 128, w: int = 128) -> np.ndarray:
    """Return a deterministic uint8 image."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_build_augmentation_pipeline() -> None:
    """build_augmentation_pipeline returns a non-None Compose object."""
    config = AugmentationConfig(enabled=True, p_rotation=0.5)
    pipeline = build_augmentation_pipeline(config, image_size=_IMG_SIZE)
    assert pipeline is not None


def test_augmentation_output_shape() -> None:
    """Pipeline resizes the image to the requested (W, H) dimensions."""
    config = AugmentationConfig(enabled=False)
    pipeline = build_augmentation_pipeline(config, image_size=_IMG_SIZE)

    image = _make_image(128, 128)
    kps = [(50.0, 50.0), (30.0, 30.0)]
    result = pipeline(image=image, keypoints=kps)

    # albumentations resizes to (H, W) → shape[0]=H, shape[1]=W
    assert result["image"].shape == (_IMG_SIZE[1], _IMG_SIZE[0], 3)


def test_val_pipeline_no_augmentation() -> None:
    """Validation pipeline produces identical output on repeated calls (deterministic)."""
    pipeline = build_val_pipeline(image_size=_IMG_SIZE)
    image = _make_image(128, 128)
    kps = [(50.0, 50.0)]

    out1 = pipeline(image=image, keypoints=kps)
    out2 = pipeline(image=image, keypoints=kps)

    assert np.array_equal(out1["image"], out2["image"])


def test_augmentation_keypoints_preserved() -> None:
    """Pipeline returns the same number of keypoints as provided."""
    config = AugmentationConfig(
        enabled=True,
        p_horizontal_flip=0.0,
        p_rotation=0.0,
        p_brightness_contrast=0.0,
    )
    pipeline = build_augmentation_pipeline(config, image_size=_IMG_SIZE)
    image = _make_image(64, 64)
    kps = [(10.0, 20.0), (30.0, 40.0), (50.0, 55.0)]

    result = pipeline(image=image, keypoints=kps)

    assert len(result["keypoints"]) == len(kps)


def test_val_pipeline_resize() -> None:
    """Validation pipeline resizes a larger image to the target size."""
    pipeline = build_val_pipeline(image_size=(32, 32))
    image = _make_image(256, 256)
    result = pipeline(image=image, keypoints=[(100.0, 100.0)])

    assert result["image"].shape == (32, 32, 3)


def test_build_augmentation_pipeline_disabled() -> None:
    """Pipeline with enabled=False applies only resize (no stochastic ops)."""
    config = AugmentationConfig(
        enabled=False,
        p_horizontal_flip=1.0,  # would be applied if enabled
        p_rotation=1.0,
    )
    pipeline = build_augmentation_pipeline(config, image_size=_IMG_SIZE)
    image = _make_image(64, 64)
    kps = [(10.0, 10.0)]

    out1 = pipeline(image=image, keypoints=kps)
    out2 = pipeline(image=image, keypoints=kps)

    # Without augmentation both runs must produce identical images
    assert np.array_equal(out1["image"], out2["image"])
