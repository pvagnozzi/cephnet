# aariz.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-19
# Description: Aariz cephalometric dataset adapter (cross-dataset verification)

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from cephnet.datasets.base import BaseDataset

logger = logging.getLogger(__name__)


class AarizDataset(BaseDataset):
    """Aariz cephalometric dataset.

    Used primarily for cross-dataset verification / drift analysis.

    Expected on-disk layout::

        <root>/
          images/          # *.png files named by image_id
          annotations/
            landmarks.csv  # columns: image_id, lm_0_x, lm_0_y, …, lm_18_x, lm_18_y
    """

    N_LANDMARKS: int = 19
    PIXEL_SPACING_MM: float = 0.1

    def __init__(
        self,
        root: Path,
        split: str = "test",
        image_size: tuple[int, int] = (512, 512),
        train_ratio: float = 0.0,
        val_ratio: float = 0.0,
        seed: int = 42,
        transform: object | None = None,
    ) -> None:
        self._train_ratio = train_ratio
        self._val_ratio = val_ratio
        self._seed = seed
        super().__init__(root=root, split=split, image_size=image_size, transform=transform)

    @property
    def n_landmarks(self) -> int:
        return self.N_LANDMARKS

    @property
    def pixel_spacing_mm(self) -> float:
        return self.PIXEL_SPACING_MM

    def _load_samples(self) -> None:
        annotations_path = self.root / "annotations" / "landmarks.csv"
        images_dir = self.root / "images"

        if not annotations_path.exists():
            logger.warning(
                "⚠️  Aariz dataset not found at %s — using synthetic data for testing",
                self.root,
            )
            self._samples = self._create_synthetic_samples()
            return

        df = pd.read_csv(annotations_path)
        all_samples: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            img_path = images_dir / f"{row['image_id']}.png"
            landmarks = [
                [float(row[f"lm_{i}_x"]), float(row[f"lm_{i}_y"])]
                for i in range(self.N_LANDMARKS)
            ]
            all_samples.append(
                {
                    "image_path": img_path,
                    "landmarks": np.array(landmarks, dtype=np.float32),
                }
            )

        # Deterministic split — mirrors ISBI2015 strategy
        rng = np.random.default_rng(self._seed)
        indices = rng.permutation(len(all_samples)).tolist()
        n_total = len(indices)
        n_train = int(n_total * self._train_ratio)
        n_val = int(n_total * self._val_ratio)

        if self.split == "train":
            selected = indices[:n_train]
        elif self.split == "val":
            selected = indices[n_train : n_train + n_val]
        else:  # test
            selected = indices[n_train + n_val :]

        self._samples = [all_samples[i] for i in selected]
        logger.info("✅ Aariz %s split: %d samples", self.split, len(self._samples))

    def _create_synthetic_samples(self) -> list[dict[str, Any]]:
        """Create in-memory synthetic samples when real data is absent."""
        rng = np.random.default_rng(99)
        # Aariz is a verification-only dataset: all samples belong to the test split
        n_samples = {"train": 0, "val": 0, "test": 50}.get(self.split, 50)
        samples: list[dict[str, Any]] = []
        for i in range(n_samples):
            samples.append(
                {
                    "image_path": Path(f"synthetic/aariz/{self.split}/{i:04d}.png"),
                    "landmarks": rng.uniform(50, 462, size=(self.N_LANDMARKS, 2)).astype(
                        np.float32
                    ),
                    "_synthetic": True,
                }
            )
        return samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self._samples[idx]
        if sample.get("_synthetic"):
            rng = np.random.default_rng(idx + 1000)  # offset to differ from ISBI2015
            image_t = torch.from_numpy(
                rng.random((3, self.image_size[1], self.image_size[0])).astype(np.float32)
            )
            landmarks_t = torch.from_numpy(sample["landmarks"])
            return {
                "image": image_t,
                "landmarks": landmarks_t,
                "image_path": str(sample["image_path"]),
            }
        return super().__getitem__(idx)
