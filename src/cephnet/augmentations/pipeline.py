# pipeline.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: Albumentations-based augmentation pipeline for cephalometric images

from __future__ import annotations

import albumentations as A

from cephnet.config.schema import AugmentationConfig


def build_augmentation_pipeline(
    config: AugmentationConfig,
    image_size: tuple[int, int],
) -> A.Compose:
    """Build a training augmentation pipeline with landmark-aware transforms.

    Args:
        config: Augmentation hyper-parameters.
        image_size: (W, H) target image dimensions.

    Returns:
        :class:`albumentations.Compose` with ``KeypointParams`` so that
        landmark coordinates are transformed consistently with the image.
    """
    transforms: list[A.BasicTransform] = [
        A.Resize(height=image_size[1], width=image_size[0]),
    ]

    if config.enabled:
        if config.p_horizontal_flip > 0:
            transforms.append(A.HorizontalFlip(p=config.p_horizontal_flip))
        if config.p_rotation > 0:
            transforms.append(
                A.Rotate(
                    limit=config.rotation_limit,
                    p=config.p_rotation,
                    border_mode=0,
                )
            )
        if config.p_brightness_contrast > 0:
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=config.p_brightness_contrast,
                )
            )
        if config.p_gaussian_noise > 0:
            transforms.append(A.GaussNoise(p=config.p_gaussian_noise))
        if config.p_elastic_transform > 0:
            transforms.append(
                A.ElasticTransform(
                    alpha=1.0,
                    sigma=50.0,
                    p=config.p_elastic_transform,
                )
            )

    return A.Compose(
        transforms,
        keypoint_params=A.KeypointParams(
            format="xy",
            remove_invisible=False,
            label_fields=[],
        ),
    )


def build_val_pipeline(image_size: tuple[int, int]) -> A.Compose:
    """Build a minimal validation pipeline (resize only, no augmentation).

    Args:
        image_size: (W, H) target image dimensions.

    Returns:
        :class:`albumentations.Compose` with resize and keypoint support.
    """
    return A.Compose(
        [A.Resize(height=image_size[1], width=image_size[0])],
        keypoint_params=A.KeypointParams(
            format="xy",
            remove_invisible=False,
        ),
    )
