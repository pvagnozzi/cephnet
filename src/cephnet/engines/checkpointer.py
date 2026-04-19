# checkpointer.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: Checkpoint save/load utilities for training state

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Checkpointer:
    """Saves and loads model checkpoints during training.

    Maintains a ``_last.pt`` file on every save and a ``_best.pt`` file
    whenever a new best validation metric is achieved.

    Args:
        checkpoint_dir: Directory where checkpoint files are written.
        experiment_name: Prefix used for checkpoint filenames.
    """

    def __init__(self, checkpoint_dir: Path | str, experiment_name: str) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric: float = float("inf")

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict[str, float],
        is_best: bool = False,
    ) -> Path:
        """Persist a full training checkpoint.

        Args:
            model: Model whose ``state_dict`` is saved.
            optimizer: Optimizer whose ``state_dict`` is saved.
            epoch: Current epoch number (1-indexed).
            metrics: Scalar metrics dict attached to the checkpoint payload.
            is_best: When *True*, also write a ``_best.pt`` copy.

        Returns:
            Path of the written checkpoint file.
        """
        state: dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }
        last_path = self.checkpoint_dir / f"{self.experiment_name}_last.pt"
        torch.save(state, last_path)

        if is_best:
            best_path = self.checkpoint_dir / f"{self.experiment_name}_best.pt"
            torch.save(state, best_path)
            logger.info(
                "💾 Best checkpoint saved → epoch=%d  mre=%.4f mm",
                epoch,
                metrics.get("val_mre_mm", metrics.get("mre_mm", 0.0)),
            )
            return best_path

        logger.debug("💾 Last checkpoint saved → epoch=%d", epoch)
        return last_path

    def load(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        checkpoint_path: Path | None = None,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Load a checkpoint and restore model (and optionally optimizer) state.

        Args:
            model: Model into which weights are loaded in-place.
            optimizer: Optional optimizer to restore.
            checkpoint_path: Explicit ``.pt`` path.  Defaults to ``_last.pt``.
            strict: Forwarded to :meth:`~torch.nn.Module.load_state_dict`.

        Returns:
            Full checkpoint payload dict.

        Raises:
            FileNotFoundError: When the resolved path does not exist.
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / f"{self.experiment_name}_last.pt"
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state: dict[str, Any] = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        model.load_state_dict(state["model_state_dict"], strict=strict)
        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])

        logger.info(
            "📂 Checkpoint loaded ← %s  (epoch %d)",
            checkpoint_path,
            state.get("epoch", -1),
        )
        return state

    def load_latest(self) -> dict[str, Any] | None:
        """Return the raw payload of the last checkpoint without touching a model.

        Returns:
            Checkpoint payload dict, or *None* if no ``_last.pt`` exists.
        """
        last_path = self.checkpoint_dir / f"{self.experiment_name}_last.pt"
        if not last_path.exists():
            return None
        state: dict[str, Any] = torch.load(
            last_path, map_location="cpu", weights_only=False
        )
        logger.debug("📂 Latest checkpoint loaded ← %s", last_path)
        return state
