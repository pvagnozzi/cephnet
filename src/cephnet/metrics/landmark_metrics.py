# landmark_metrics.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: MRE, SDR, and per-landmark metrics for cephalometric landmark detection

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import numpy as np


def compute_mre(
    predictions: np.ndarray,
    targets: np.ndarray,
    pixel_spacing_mm: float = 0.1,
) -> float:
    """Compute Mean Radial Error (MRE) in millimetres.

    Args:
        predictions: ``(N_samples, N_landmarks, 2)`` predicted pixel coordinates.
        targets:     ``(N_samples, N_landmarks, 2)`` ground-truth pixel coordinates.
        pixel_spacing_mm: Millimetres per pixel used for unit conversion.

    Returns:
        MRE in mm as a Python float.
    """
    distances = np.linalg.norm(predictions - targets, axis=-1)  # (N, L)
    return float(np.mean(distances) * pixel_spacing_mm)


def compute_sdr(
    predictions: np.ndarray,
    targets: np.ndarray,
    pixel_spacing_mm: float = 0.1,
    thresholds_mm: tuple[float, ...] = (2.0, 2.5, 3.0, 4.0),
) -> dict[str, float]:
    """Compute Successful Detection Rate (SDR) at multiple mm thresholds.

    Args:
        predictions: ``(N_samples, N_landmarks, 2)`` predicted pixel coordinates.
        targets:     ``(N_samples, N_landmarks, 2)`` ground-truth pixel coordinates.
        pixel_spacing_mm: Millimetres per pixel.
        thresholds_mm: Detection radius thresholds in mm.

    Returns:
        Mapping ``"sdr@Xmm"`` → fraction of detections within *X* mm.
    """
    distances_mm = np.linalg.norm(predictions - targets, axis=-1) * pixel_spacing_mm
    return {f"sdr@{thr}mm": float(np.mean(distances_mm <= thr)) for thr in thresholds_mm}


def compute_per_landmark_mre(
    predictions: np.ndarray,
    targets: np.ndarray,
    pixel_spacing_mm: float = 0.1,
) -> np.ndarray:
    """Compute per-landmark MRE in mm.

    Args:
        predictions: ``(N_samples, N_landmarks, 2)`` predicted pixel coordinates.
        targets:     ``(N_samples, N_landmarks, 2)`` ground-truth pixel coordinates.
        pixel_spacing_mm: Millimetres per pixel.

    Returns:
        ``(N_landmarks,)`` array of per-landmark mean errors in mm.
    """
    distances = np.linalg.norm(predictions - targets, axis=-1)  # (N_samples, N_landmarks)
    return cast(np.ndarray, np.mean(distances, axis=0) * pixel_spacing_mm)


@dataclass
class LandmarkMetrics:
    """Container for all cephalometric landmark detection metrics."""

    mre_mm: float
    sdr_2mm: float
    sdr_2_5mm: float
    sdr_3mm: float
    sdr_4mm: float
    per_landmark_mre: np.ndarray  # (N_landmarks,)

    @classmethod
    def compute(
        cls,
        predictions: np.ndarray,
        targets: np.ndarray,
        pixel_spacing_mm: float = 0.1,
    ) -> LandmarkMetrics:
        """Compute all metrics at once.

        Args:
            predictions: ``(N_samples, N_landmarks, 2)`` predicted coordinates.
            targets:     ``(N_samples, N_landmarks, 2)`` ground-truth coordinates.
            pixel_spacing_mm: Millimetres per pixel.

        Returns:
            Populated :class:`LandmarkMetrics` instance.
        """
        mre = compute_mre(predictions, targets, pixel_spacing_mm)
        sdr = compute_sdr(predictions, targets, pixel_spacing_mm)
        per_lm = compute_per_landmark_mre(predictions, targets, pixel_spacing_mm)
        return cls(
            mre_mm=mre,
            sdr_2mm=sdr["sdr@2.0mm"],
            sdr_2_5mm=sdr["sdr@2.5mm"],
            sdr_3mm=sdr["sdr@3.0mm"],
            sdr_4mm=sdr["sdr@4.0mm"],
            per_landmark_mre=per_lm,
        )

    def to_dict(self) -> dict[str, float]:
        """Serialise scalar metrics to a plain dict."""
        return {
            "mre_mm": self.mre_mm,
            "sdr@2mm": self.sdr_2mm,
            "sdr@2.5mm": self.sdr_2_5mm,
            "sdr@3mm": self.sdr_3mm,
            "sdr@4mm": self.sdr_4mm,
        }

    def log_summary(self, logger: object) -> None:
        """Emit a structured summary to *logger*.

        Args:
            logger: A :class:`logging.Logger` instance or any object that
                    exposes an ``info`` method with printf-style formatting.
        """
        log = logger if isinstance(logger, logging.Logger) else logging.getLogger(__name__)
        log.info("📊 MRE:       %.2f mm", self.mre_mm)
        log.info("📊 SDR@2mm:   %.1f%%", self.sdr_2mm * 100)
        log.info("📊 SDR@2.5mm: %.1f%%", self.sdr_2_5mm * 100)
        log.info("📊 SDR@3mm:   %.1f%%", self.sdr_3mm * 100)
        log.info("📊 SDR@4mm:   %.1f%%", self.sdr_4mm * 100)
