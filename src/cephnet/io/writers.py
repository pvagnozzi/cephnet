# writers.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: Artifact and report writers for cephnet training outputs

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; color: #333; }}
  h1 {{ color: #1a73e8; }} h2 {{ color: #555; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: right; }}
  th {{ background: #f0f4ff; text-align: center; }}
  td:first-child, th:first-child {{ text-align: left; }}
  .metric {{ display: inline-block; background: #e8f0fe; border-radius: 6px;
             padding: 10px 20px; margin: 8px; text-align: center; }}
  .metric .val {{ font-size: 1.6em; font-weight: bold; color: #1a73e8; }}
  .metric .lbl {{ font-size: 0.85em; color: #555; }}
</style>
</head>
<body>
<h1>&#x1F9E0; {title}</h1>
<p><b>Experiment:</b> {experiment}</p>

<h2>&#x1F4CA; Summary Metrics</h2>
<div>
{metric_cards}
</div>

<h2>&#x1F4C8; Learning Curve</h2>
{learning_curve_table}

<h2>&#x1F3AF; Per-Landmark Error (MRE mm)</h2>
{landmark_table}

</body></html>
"""


class MetricsWriter:
    """Writes per-epoch metrics to JSON and CSV files."""

    def __init__(self, output_dir: Path, experiment_name: str) -> None:
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._history: list[dict[str, Any]] = []

    def log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        """Append epoch metrics and flush to disk."""
        entry = {"epoch": epoch, **metrics}
        self._history.append(entry)
        self._flush()

    def _flush(self) -> None:
        json_path = self.output_dir / f"{self.experiment_name}_metrics.json"
        csv_path = self.output_dir / f"{self.experiment_name}_metrics.csv"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(self._history, f, indent=2)
        df = pd.DataFrame(self._history)
        df.to_csv(csv_path, index=False)
        logger.debug("📊 Metrics written to %s", csv_path)

    def get_history(self) -> list[dict[str, Any]]:
        """Return a copy of the metrics history."""
        return list(self._history)


class ReportWriter:
    """Writes final evaluation reports to reports/metrics/."""

    def __init__(self, reports_dir: Path) -> None:
        self.reports_dir = reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def write_landmark_table(
        self,
        landmark_names: list[str],
        per_landmark_mre: list[float],
        experiment_name: str,
    ) -> Path:
        """Write per-landmark MRE table sorted by error, return output path."""
        df = pd.DataFrame(
            {
                "landmark": landmark_names,
                "mre_mm": per_landmark_mre,
            }
        )
        df = df.sort_values("mre_mm")
        out = self.reports_dir / f"{experiment_name}_landmark_table.csv"
        df.to_csv(out, index=False)
        logger.info("📋 Landmark error table saved to %s", out)
        return out

    def write_summary(self, summary: dict[str, Any], experiment_name: str) -> Path:
        """Serialize a summary dict as JSON and return output path."""
        out = self.reports_dir / f"{experiment_name}_summary.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("📝 Summary report saved to %s", out)
        return out

    def write_html_report(
        self,
        experiment_name: str,
        summary: dict[str, Any],
        metrics_history: list[dict[str, Any]] | None = None,
        landmark_names: list[str] | None = None,
        per_landmark_mre: list[float] | None = None,
        title: str | None = None,
    ) -> Path:
        """Write a self-contained HTML report and return the output path.

        Args:
            experiment_name: Used as the filename stem.
            summary: Scalar metrics dict (mre_mm, sdr@*mm, …).
            metrics_history: Optional list of per-epoch dicts from MetricsWriter.
            landmark_names: Optional ordered landmark label list.
            per_landmark_mre: Optional per-landmark MRE values matching the names.
            title: Page title; defaults to the experiment name.
        """
        _title = title or experiment_name

        # --- metric cards ---
        card_keys = ["mre_mm", "sdr@2mm", "sdr@2.5mm", "sdr@3mm", "sdr@4mm"]
        cards_html = ""
        for k in card_keys:
            if k in summary:
                val = summary[k]
                label = k.replace("_", " ").replace("@", " @ ")
                fmt = f"{val:.2f}" if isinstance(val, float) else str(val)
                cards_html += (
                    f'<div class="metric"><div class="val">{fmt}</div>'
                    f'<div class="lbl">{label}</div></div>\n'
                )

        # --- learning curve ---
        if metrics_history:
            lc_df = pd.DataFrame(metrics_history)
            lc_html = lc_df.to_html(index=False, border=0, float_format="%.4f")
        else:
            lc_html = "<p><em>No learning curve data available.</em></p>"

        # --- per-landmark table ---
        if per_landmark_mre is not None:
            n = len(per_landmark_mre)
            names = landmark_names if landmark_names and len(landmark_names) == n else [
                f"L{i + 1:02d}" for i in range(n)
            ]
            lm_df = pd.DataFrame({"Landmark": names, "MRE (mm)": per_landmark_mre})
            lm_df = lm_df.sort_values("MRE (mm)")
            lm_html = lm_df.to_html(index=False, border=0, float_format="%.3f")
        else:
            lm_html = "<p><em>No per-landmark data available.</em></p>"

        html = _HTML_TEMPLATE.format(
            title=_title,
            experiment=experiment_name,
            metric_cards=cards_html,
            learning_curve_table=lc_html,
            landmark_table=lm_html,
        )

        out = self.reports_dir / f"{experiment_name}_report.html"
        out.write_text(html, encoding="utf-8")
        logger.info("📄 HTML report saved to %s", out)
        return out
