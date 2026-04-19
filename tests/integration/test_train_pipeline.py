# test_train_pipeline.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: Integration test: mini training loop with synthetic data

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from cephnet.augmentations.pipeline import build_val_pipeline
from cephnet.config.loader import load_config
from cephnet.datasets.isbi2015 import ISBI2015Dataset
from cephnet.engines.checkpointer import Checkpointer
from cephnet.engines.trainer import Trainer
from cephnet.io.writers import MetricsWriter
from cephnet.models.regression import build_model
from cephnet.utils.seeding import seed_everything


@pytest.mark.slow
def test_full_training_loop(tmp_config: Path, tmp_path: Path) -> None:
    """2-epoch training loop with synthetic data creates checkpoints and logs metrics.

    This test verifies:
    - Config loads correctly.
    - ISBI2015Dataset falls back to synthetic data.
    - Trainer runs for the configured number of epochs without crashing.
    - At least one checkpoint ``.pt`` file is written.
    - MetricsWriter records one entry per epoch.
    """
    seed_everything(42)

    config = load_config(tmp_config)

    # Build validation transform (resize only)
    val_transform = build_val_pipeline(image_size=tuple(config.dataset.image_size))  # type: ignore[arg-type]

    # Create synthetic datasets (root does not exist → synthetic fallback)
    data_root = tmp_path / "nonexistent_dataset"
    train_ds = ISBI2015Dataset(
        root=data_root,
        split="train",
        image_size=tuple(config.dataset.image_size),  # type: ignore[arg-type]
        seed=config.training.seed,
    )
    val_ds = ISBI2015Dataset(
        root=data_root,
        split="val",
        image_size=tuple(config.dataset.image_size),  # type: ignore[arg-type]
        transform=val_transform,
        seed=config.training.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Instantiate model, checkpointer, and writer
    model = build_model(config.model)

    checkpoint_dir = Path(str(config.training.checkpoint_dir))
    checkpointer = Checkpointer(
        checkpoint_dir=checkpoint_dir,
        experiment_name=config.training.experiment_name,
    )
    writer = MetricsWriter(
        output_dir=checkpoint_dir,
        experiment_name=config.training.experiment_name,
    )

    # Run training loop — Trainer expects (config, model, loaders, device, writer, checkpointer)
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        metrics_writer=writer,
        checkpointer=checkpointer,
    )
    trainer.train()

    # --- Assertions ---

    # 1. Checkpoint files exist
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    assert len(checkpoints) > 0, "No checkpoint files were written"

    # 2. One entry per epoch in metrics history
    history = writer.get_history()
    assert len(history) == config.training.epochs

    # 3. train_loss key present in every epoch entry
    for entry in history:
        assert "train_loss" in entry
        assert isinstance(entry["train_loss"], float)

    # 4. Checkpoint is loadable and contains expected keys
    latest = checkpointer.load_latest()
    assert latest is not None
    assert "epoch" in latest
    assert "model_state_dict" in latest
    assert latest["epoch"] == config.training.epochs
