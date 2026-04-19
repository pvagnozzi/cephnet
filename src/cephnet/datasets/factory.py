# factory.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: Dataset factory — returns the correct dataset adapter by name

from __future__ import annotations

from pathlib import Path
from typing import Any

from cephnet.datasets.aariz import AarizDataset
from cephnet.datasets.base import BaseDataset
from cephnet.datasets.isbi2015 import ISBI2015Dataset

_REGISTRY: dict[str, type[BaseDataset]] = {
    "isbi2015": ISBI2015Dataset,
    "aariz": AarizDataset,
}


def get_dataset(
    name: str,
    root: Path,
    split: str,
    image_size: tuple[int, int] = (512, 512),
    transform: object | None = None,
    **kwargs: Any,
) -> BaseDataset:
    """Factory: instantiate a dataset adapter by name.

    Args:
        name: Dataset key — one of "isbi2015", "aariz".
        root: Root directory of the dataset on the host mount.
        split: "train", "val", or "test".
        image_size: (W, H) resize target applied to all images.
        transform: Optional albumentations Compose pipeline.
        **kwargs: Additional keyword arguments forwarded to the dataset class.

    Returns:
        Instantiated :class:`BaseDataset` subclass.

    Raises:
        ValueError: When ``name`` is not registered.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name!r}. Available: {sorted(_REGISTRY.keys())}"
        )
    cls = _REGISTRY[name]
    return cls(root=root, split=split, image_size=image_size, transform=transform, **kwargs)
