# cross_dataset.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: Cross-dataset verification — train on dataset A, evaluate on dataset B

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cephnet.engines.validator import Validator
from cephnet.metrics.landmark_metrics import LandmarkMetrics

logger = logging.getLogger(__name__)


class CrossDatasetEvaluator:
    """Evaluates a trained model on a held-out verification dataset.

    Typical usage:
        train on ISBI2015  →  evaluate on Aariz  →  generate verification report
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        report_dir: Path,
        experiment_name: str,
    ) -> None:
        self.model = model
        self.device = device
        self.report_dir = report_dir
        self.experiment_name = experiment_name
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        loader: DataLoader,  # type: ignore[type-arg]
        pixel_spacing_mm: float,
        image_size: tuple[int, int],
        heatmap_size: tuple[int, int] | None,
        landmark_names: list[str] | None = None,
        dataset_name: str = "unknown",
    ) -> LandmarkMetrics:
        """Run cross-dataset evaluation and write verification reports."""
        logger.info("🧪 Cross-dataset verification on: [bold]%s[/bold]", dataset_name)

        validator = Validator(
            model=self.model,
            device=self.device,
            pixel_spacing_mm=pixel_spacing_mm,
            image_size=image_size,
            heatmap_size=heatmap_size,
        )
        metrics = validator.evaluate(loader)
        metrics.log_summary(logger)

        # Write summary JSON
        summary: dict[str, object] = {
            "experiment": self.experiment_name,
            "dataset": dataset_name,
            **metrics.to_dict(),
        }
        summary_path = self.report_dir / f"{self.experiment_name}_{dataset_name}_verification.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("📝 Verification summary → %s", summary_path)

        # Write per-landmark CSV if names provided
        if landmark_names:
            table_path = self.report_dir / f"{self.experiment_name}_{dataset_name}_per_landmark.csv"
            df = pd.DataFrame(
                {
                    "landmark": landmark_names,
                    "mre_mm": metrics.per_landmark_mre.tolist(),
                }
            ).sort_values("mre_mm")
            df.to_csv(table_path, index=False)
            logger.info("📋 Per-landmark table → %s", table_path)

        logger.info("✅ Cross-dataset verification complete")
        return metrics
