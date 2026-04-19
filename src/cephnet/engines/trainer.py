# trainer.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: Main training loop with mixed precision, early stopping, and checkpointing

from __future__ import annotations

import logging
from typing import cast

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from cephnet.config.schema import CephnetConfig, HeatmapModelConfig
from cephnet.engines.checkpointer import Checkpointer
from cephnet.engines.validator import Validator
from cephnet.io.writers import MetricsWriter
from cephnet.losses.heatmap_loss import HeatmapLoss, generate_gaussian_heatmaps
from cephnet.losses.regression_loss import RegressionLoss

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Monitors a validation metric and signals when training should stop.

    Args:
        patience: Epochs without improvement before stopping.
        min_delta: Minimum improvement that resets the patience counter.
    """

    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter: int = 0
        self.best: float = float("inf")

    def step(self, metric: float) -> bool:
        """Update and return *True* when training should halt.

        Args:
            metric: Current validation metric (lower is better).
        """
        if metric < self.best - self.min_delta:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class Trainer:
    """Orchestrates the full training loop for cephnet models.

    Supports both heatmap and regression objectives, with:

    * Mixed-precision training (CUDA only).
    * AdamW optimiser with configurable LR scheduler.
    * Gradient clipping.
    * Per-epoch validation and best-checkpoint tracking.
    * Early stopping.
    * Metrics persistence via :class:`~cephnet.io.writers.MetricsWriter`.

    Args:
        config: Fully validated experiment configuration.
        model: Instantiated model (``HeatmapModel`` or ``RegressionModel``).
        train_loader: DataLoader for the training split.
        val_loader: DataLoader for the validation split.
        device: Target compute device.
        metrics_writer: Sink for per-epoch scalar metrics.
        checkpointer: Checkpoint manager that tracks ``best_metric``.
    """

    def __init__(
        self,
        config: CephnetConfig,
        model: nn.Module,
        train_loader: DataLoader,  # type: ignore[type-arg]
        val_loader: DataLoader,  # type: ignore[type-arg]
        device: torch.device | str = "cpu",
        metrics_writer: MetricsWriter | None = None,
        checkpointer: Checkpointer | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.metrics_writer = metrics_writer
        self.checkpointer = checkpointer

        self.is_heatmap = isinstance(config.model, HeatmapModelConfig)
        self.criterion: nn.Module = HeatmapLoss() if self.is_heatmap else RegressionLoss()

        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        self.scheduler = self._build_scheduler()

        self.scaler: GradScaler | None = None
        if config.training.mixed_precision and self.device.type == "cuda":
            self.scaler = GradScaler()

        es_cfg = config.training.early_stopping
        self.early_stopping: EarlyStopping | None = (
            EarlyStopping(patience=es_cfg.patience, min_delta=es_cfg.min_delta)
            if es_cfg.enabled
            else None
        )

        heatmap_size = (
            config.model.heatmap_size  # type: ignore[union-attr]
            if isinstance(config.model, HeatmapModelConfig)
            else None
        )
        self.validator = Validator(
            model=self.model,
            device=self.device,
            pixel_spacing_mm=config.dataset.pixel_spacing_mm,
            image_size=config.dataset.image_size,
            heatmap_size=heatmap_size,
        )

    # ------------------------------------------------------------------
    # Scheduler factory
    # ------------------------------------------------------------------

    def _build_scheduler(
        self,
    ) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
        sched_cfg = self.config.training.scheduler
        match sched_cfg.type:
            case "cosine":
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=int(sched_cfg.params.get("T_max", self.config.training.epochs)),
                    eta_min=float(sched_cfg.params.get("eta_min", 1e-6)),
                )
            case "reduce_on_plateau":
                return torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    patience=int(sched_cfg.params.get("patience", 10)),
                    factor=float(sched_cfg.params.get("factor", 0.5)),
                )
            case "step":
                return torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=int(sched_cfg.params.get("step_size", 30)),
                    gamma=float(sched_cfg.params.get("gamma", 0.1)),
                )
            case _:
                return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop until completion or early stopping."""
        cfg_tr = self.config.training
        logger.info("🚀 Training: %s", cfg_tr.experiment_name)
        logger.info(
            "   Epochs=%d  Batch=%d  LR=%.2e  Device=%s  Mixed-precision=%s",
            cfg_tr.epochs,
            cfg_tr.batch_size,
            cfg_tr.learning_rate,
            self.device,
            cfg_tr.mixed_precision,
        )

        for epoch in range(1, cfg_tr.epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_metrics = self.validator.evaluate(self.val_loader)

            current_lr: float = self.optimizer.param_groups[0]["lr"]
            metrics: dict[str, float] = {
                "train_loss": train_loss,
                "val_mre_mm": val_metrics.mre_mm,
                "val_sdr_2mm": val_metrics.sdr_2mm,
                "val_sdr_2_5mm": val_metrics.sdr_2_5mm,
                "val_sdr_3mm": val_metrics.sdr_3mm,
                "val_sdr_4mm": val_metrics.sdr_4mm,
                "lr": current_lr,
            }

            logger.info(
                "Epoch [%3d/%3d] | loss=%.4f | MRE=%.2fmm | SDR@2mm=%.1f%% | LR=%.2e",
                epoch,
                cfg_tr.epochs,
                train_loss,
                val_metrics.mre_mm,
                val_metrics.sdr_2mm * 100,
                current_lr,
            )

            if self.checkpointer is not None:
                is_best = val_metrics.mre_mm < self.checkpointer.best_metric
                if is_best:
                    self.checkpointer.best_metric = val_metrics.mre_mm
                self.checkpointer.save(self.model, self.optimizer, epoch, metrics, is_best=is_best)

            if self.metrics_writer is not None:
                self.metrics_writer.log_epoch(epoch, metrics)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.mre_mm)
                else:
                    self.scheduler.step()

            if self.early_stopping is not None and self.early_stopping.step(val_metrics.mre_mm):
                logger.warning("⚠️  Early stopping triggered at epoch %d", epoch)
                break

        best = self.checkpointer.best_metric if self.checkpointer else float("inf")
        logger.info("✅ Training complete. Best MRE: %.2f mm", best)

    def fit(self) -> None:
        """Alias for :meth:`train` for API compatibility."""
        self.train()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> float:
        """Run one training epoch and return the mean batch loss."""
        self.model.train()
        total_loss = 0.0
        cfg_tr = self.config.training

        for step, batch in enumerate(self.train_loader, start=1):
            images: torch.Tensor = batch["image"].to(self.device)
            landmarks: torch.Tensor = batch["landmarks"].to(self.device)

            with torch.autocast(
                device_type=self.device.type,
                enabled=self.scaler is not None,
            ):
                if self.is_heatmap:
                    hm_cfg = cast(HeatmapModelConfig, self.config.model)
                    target_heatmaps = generate_gaussian_heatmaps(
                        landmarks.float(),
                        heatmap_size=hm_cfg.heatmap_size,
                        sigma=hm_cfg.heatmap_sigma,
                        image_size=self.config.dataset.image_size,
                    )
                    pred: torch.Tensor = self.model(images)
                    loss: torch.Tensor = self.criterion(pred, target_heatmaps)
                else:
                    H, W = self.config.dataset.image_size
                    norm_landmarks = landmarks.clone().float()
                    norm_landmarks[..., 0] = (norm_landmarks[..., 0] / float(W)).clamp(0.0, 1.0)
                    norm_landmarks[..., 1] = (norm_landmarks[..., 1] / float(H)).clamp(0.0, 1.0)
                    pred = self.model(images)
                    loss = self.criterion(pred, norm_landmarks)

            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg_tr.gradient_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg_tr.gradient_clip_norm
                )
                self.optimizer.step()

            total_loss += loss.item()

            if step % self.config.logging.log_interval_steps == 0:
                logger.debug(
                    "  Epoch %d  step %d/%d | loss=%.4f",
                    epoch,
                    step,
                    len(self.train_loader),
                    loss.item(),
                )

        return total_loss / max(len(self.train_loader), 1)
