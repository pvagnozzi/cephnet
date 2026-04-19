# validator.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: Validation loop for landmark detection models

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cephnet.metrics.landmark_metrics import LandmarkMetrics

logger = logging.getLogger(__name__)


class Validator:
    """Runs evaluation on a dataset split and computes landmark metrics.

    Supports both heatmap-based and regression-based models:

    * **Heatmap** — peak coordinates extracted from predicted heatmaps via
      argmax and scaled back to image space.
    * **Regression** — predicted normalised ``[0, 1]`` coordinates are
      multiplied by the image dimensions.

    Args:
        model: Trained model (set to eval mode automatically).
        device: Target compute device.
        pixel_spacing_mm: Physical pixel size used for MRE / SDR.
        image_size: ``(H, W)`` of the input images.
        heatmap_size: ``(H, W)`` of the model's output heatmaps.
                      Pass *None* for regression models.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        pixel_spacing_mm: float = 0.1,
        image_size: tuple[int, int] = (512, 512),
        heatmap_size: tuple[int, int] | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.pixel_spacing_mm = pixel_spacing_mm
        self.image_size = image_size
        self.heatmap_size = heatmap_size

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> LandmarkMetrics:  # type: ignore[type-arg]
        """Run a full evaluation pass and return aggregated metrics.

        Args:
            loader: DataLoader whose batches contain ``"image"`` and
                    ``"landmarks"`` keys.

        Returns:
            Populated :class:`~cephnet.metrics.landmark_metrics.LandmarkMetrics`.
        """
        self.model.eval()
        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []

        for batch in loader:
            images: torch.Tensor = batch["image"].to(self.device)
            targets: np.ndarray = batch["landmarks"].numpy()  # (B, N, 2)

            output: torch.Tensor = self.model(images)

            if self.heatmap_size is not None:
                preds = self._heatmap_to_coords(output)  # (B, N, 2) in heatmap space
                scale_x = self.image_size[0] / self.heatmap_size[0]
                scale_y = self.image_size[1] / self.heatmap_size[1]
                preds[:, :, 0] *= scale_x
                preds[:, :, 1] *= scale_y
            else:
                preds = output.cpu().numpy()  # (B, N, 2) normalised [0, 1]
                preds[:, :, 0] *= self.image_size[0]
                preds[:, :, 1] *= self.image_size[1]

            all_preds.append(preds)
            all_targets.append(targets)

        predictions = np.concatenate(all_preds, axis=0)
        gt = np.concatenate(all_targets, axis=0)
        return LandmarkMetrics.compute(predictions, gt, self.pixel_spacing_mm)

    def _heatmap_to_coords(self, heatmaps: torch.Tensor) -> np.ndarray:
        """Extract peak coordinates from predicted heatmaps via argmax.

        Args:
            heatmaps: ``(B, N, H, W)`` raw heatmap logits.

        Returns:
            ``(B, N, 2)`` coordinates ``(x, y)`` in heatmap pixel space.
        """
        B, N, H, W = heatmaps.shape
        flat = heatmaps.view(B, N, -1)
        argmax = flat.argmax(dim=-1).cpu().numpy()  # (B, N)
        y = (argmax // W).astype(np.float32)
        x = (argmax % W).astype(np.float32)
        return np.stack([x, y], axis=-1)  # (B, N, 2)
