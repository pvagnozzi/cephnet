# test_io.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: Tests for MetricsWriter and ReportWriter artifact writers

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cephnet.io.writers import MetricsWriter, ReportWriter

_EXP = "unit_test"


def test_metrics_writer_log_epoch(tmp_path: Path) -> None:
    """log_epoch creates both a JSON and a CSV metrics file."""
    writer = MetricsWriter(output_dir=tmp_path / "metrics", experiment_name=_EXP)
    writer.log_epoch(1, {"train_loss": 0.5, "val_loss": 0.6})

    assert (tmp_path / "metrics" / f"{_EXP}_metrics.json").exists()
    assert (tmp_path / "metrics" / f"{_EXP}_metrics.csv").exists()


def test_metrics_writer_history(tmp_path: Path) -> None:
    """get_history returns all logged epoch entries in insertion order."""
    writer = MetricsWriter(output_dir=tmp_path / "metrics", experiment_name=_EXP)
    writer.log_epoch(1, {"loss": 1.0})
    writer.log_epoch(2, {"loss": 0.5})

    history = writer.get_history()

    assert len(history) == 2
    assert history[0]["epoch"] == 1
    assert history[0]["loss"] == pytest.approx(1.0)
    assert history[1]["epoch"] == 2
    assert history[1]["loss"] == pytest.approx(0.5)


def test_metrics_writer_json_content(tmp_path: Path) -> None:
    """Flushed JSON contains the same data returned by get_history."""
    writer = MetricsWriter(output_dir=tmp_path / "metrics", experiment_name=_EXP)
    writer.log_epoch(1, {"mre_mm": 2.3})

    json_path = tmp_path / "metrics" / f"{_EXP}_metrics.json"
    loaded = json.loads(json_path.read_text())

    assert loaded[0]["mre_mm"] == pytest.approx(2.3)


def test_report_writer_landmark_table(tmp_path: Path) -> None:
    """write_landmark_table creates a CSV file and returns its path."""
    writer = ReportWriter(reports_dir=tmp_path / "reports")
    names = [f"lm_{i}" for i in range(5)]
    errors = [0.5 + i * 0.1 for i in range(5)]

    path = writer.write_landmark_table(names, errors, experiment_name=_EXP)

    assert path.exists()
    assert path.suffix == ".csv"


def test_report_writer_landmark_table_content(tmp_path: Path) -> None:
    """Landmark table CSV contains landmark and mre_mm columns."""
    import pandas as pd

    writer = ReportWriter(reports_dir=tmp_path / "reports")
    names = [f"lm_{i}" for i in range(3)]
    errors = [1.0, 2.0, 3.0]

    path = writer.write_landmark_table(names, errors, experiment_name=_EXP)
    df = pd.read_csv(path)

    assert "landmark" in df.columns
    assert "mre_mm" in df.columns
    assert len(df) == 3


def test_report_writer_summary(tmp_path: Path) -> None:
    """write_summary creates a JSON file and returns its path."""
    writer = ReportWriter(reports_dir=tmp_path / "reports")
    summary = {"mre_mm": 1.5, "sdr@2mm": 0.8, "experiment": _EXP}

    path = writer.write_summary(summary, experiment_name=_EXP)

    assert path.exists()
    assert path.suffix == ".json"
    content = json.loads(path.read_text())
    assert content["mre_mm"] == pytest.approx(1.5)
    assert content["sdr@2mm"] == pytest.approx(0.8)
