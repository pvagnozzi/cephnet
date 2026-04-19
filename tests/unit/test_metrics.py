# test_metrics.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: Tests for MRE, SDR and LandmarkMetrics computation

from __future__ import annotations

import numpy as np
import pytest

from cephnet.metrics.landmark_metrics import (
    LandmarkMetrics,
    compute_mre,
    compute_sdr,
)

# Helpers ------------------------------------------------------------------

_N_SAMPLES = 5
_N_LM = 19


def _zeros() -> np.ndarray:
    return np.zeros((_N_SAMPLES, _N_LM, 2), dtype=np.float32)


# Tests --------------------------------------------------------------------


def test_compute_mre_zero_error() -> None:
    """Identical predictions and targets produce MRE of exactly 0.0 mm."""
    pts = _zeros()
    assert compute_mre(pts, pts, pixel_spacing_mm=0.1) == pytest.approx(0.0)


def test_compute_mre_known_error() -> None:
    """10 px horizontal offset with 0.1 mm/px spacing gives MRE = 1.0 mm."""
    preds = _zeros()
    targets = _zeros()
    targets[:, :, 0] = 10.0  # all landmarks shifted 10 px in x
    mre = compute_mre(preds, targets, pixel_spacing_mm=0.1)
    assert mre == pytest.approx(1.0, abs=1e-4)


def test_compute_sdr_all_within_threshold() -> None:
    """1 px errors (= 0.1 mm) are all within the 2 mm threshold → SDR = 1.0."""
    preds = np.ones((_N_SAMPLES, _N_LM, 2), dtype=np.float32)
    targets = _zeros()
    sdr = compute_sdr(preds, targets, pixel_spacing_mm=0.1)
    assert sdr["sdr@2.0mm"] == pytest.approx(1.0)


def test_compute_sdr_none_within_threshold() -> None:
    """100 px errors (= 10 mm) are all outside the 2 mm threshold → SDR = 0.0."""
    preds = _zeros()
    targets = np.full((_N_SAMPLES, _N_LM, 2), 100.0, dtype=np.float32)
    sdr = compute_sdr(preds, targets, pixel_spacing_mm=0.1)
    assert sdr["sdr@2.0mm"] == pytest.approx(0.0)


def test_landmark_metrics_compute() -> None:
    """LandmarkMetrics.compute returns correct values for zero-error predictions."""
    preds = _zeros()
    targets = _zeros()
    m = LandmarkMetrics.compute(preds, targets, pixel_spacing_mm=0.1)

    assert m.mre_mm == pytest.approx(0.0)
    assert m.sdr_2mm == pytest.approx(1.0)
    assert m.sdr_2_5mm == pytest.approx(1.0)
    assert m.sdr_3mm == pytest.approx(1.0)
    assert m.sdr_4mm == pytest.approx(1.0)
    assert m.per_landmark_mre.shape == (_N_LM,)
    assert np.allclose(m.per_landmark_mre, 0.0)


def test_landmark_metrics_to_dict() -> None:
    """LandmarkMetrics.to_dict contains all expected metric keys."""
    preds = _zeros()
    targets = _zeros()
    d = LandmarkMetrics.compute(preds, targets, pixel_spacing_mm=0.1).to_dict()

    for key in ("mre_mm", "sdr@2mm", "sdr@2.5mm", "sdr@3mm", "sdr@4mm"):
        assert key in d, f"Missing key: {key}"
    assert isinstance(d["mre_mm"], float)
