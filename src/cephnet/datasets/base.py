# base.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: Abstract base class for cephalometric landmark datasets

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset[dict[str, Any]], abc.ABC):
    """Abstract base dataset for cephalometric landmark detection."""

    def __init__(
        self,
        root: Path,
        split: str,  # "train", "val", "test"
        image_size: tuple[int, int] = (512, 512),
        transform: Any | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.transform = transform
        # {"image_path": Path, "landmarks": np.ndarray (N,2)}
        self._samples: list[dict[str, Any]] = []
        self._load_samples()

    @abc.abstractmethod
    def _load_samples(self) -> None:
        """Populate self._samples list."""

    @property
    @abc.abstractmethod
    def n_landmarks(self) -> int:
        """Number of landmarks per image."""

    @property
    @abc.abstractmethod
    def pixel_spacing_mm(self) -> float:
        """Pixel spacing in mm for metric conversion."""

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self._samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        landmarks = np.array(sample["landmarks"], dtype=np.float32)  # (N, 2) pixel coords

        # Resize image and scale landmarks proportionally
        orig_w, orig_h = image.size
        image = image.resize(self.image_size, Image.Resampling.BILINEAR)
        scale_x = self.image_size[0] / orig_w
        scale_y = self.image_size[1] / orig_h
        landmarks[:, 0] *= scale_x
        landmarks[:, 1] *= scale_y

        # Apply albumentations transform (keypoint-aware)
        if self.transform is not None:
            result = self.transform(
                image=np.array(image),
                keypoints=[(float(x), float(y)) for x, y in landmarks],
            )
            image_np: np.ndarray = result["image"]
            kps: list[tuple[float, float]] = result["keypoints"]
            landmarks = np.array([[x, y] for x, y in kps], dtype=np.float32)
        else:
            image_np = np.array(image)

        # Normalise to [0, 1] and convert to tensors
        image_t = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        landmarks_t = torch.from_numpy(landmarks)

        return {
            "image": image_t,          # (3, H, W)
            "landmarks": landmarks_t,  # (N, 2) pixel coords in resized image
            "image_path": str(sample["image_path"]),
        }
