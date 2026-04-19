# checkpoint.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: Checkpoint save/load utilities for training state persistence

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


class CheckpointIO:
    """Saves and loads training checkpoints to/from a directory."""

    def __init__(self, checkpoint_dir: Path, experiment_name: str) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        epoch: int,
        model_state: dict[str, Any],
        optimizer_state: dict[str, Any],
        metrics: dict[str, float],
        is_best: bool = False,
    ) -> Path:
        """Persist a checkpoint for the given epoch."""
        payload: dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "metrics": metrics,
        }
        ckpt_path = self.checkpoint_dir / f"{self.experiment_name}_epoch_{epoch:04d}.pt"
        torch.save(payload, ckpt_path)
        logger.info("💾 Checkpoint saved → %s", ckpt_path)

        if is_best:
            best_path = self.checkpoint_dir / f"{self.experiment_name}_best.pt"
            torch.save(payload, best_path)
            logger.info("🏆 Best checkpoint updated → %s", best_path)

        return ckpt_path

    def load(self, checkpoint_path: str | Path) -> dict[str, Any]:
        """Load a checkpoint from disk and return its payload dict."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
        logger.info("📂 Checkpoint loaded ← %s (epoch %d)", path, payload.get("epoch", -1))
        return payload

    def load_best(self) -> dict[str, Any]:
        """Load the best checkpoint for this experiment."""
        best_path = self.checkpoint_dir / f"{self.experiment_name}_best.pt"
        return self.load(best_path)

    def latest_checkpoint(self) -> Path | None:
        """Return the path to the most recent epoch checkpoint, or None."""
        pattern = f"{self.experiment_name}_epoch_*.pt"
        candidates = sorted(self.checkpoint_dir.glob(pattern))
        return candidates[-1] if candidates else None
