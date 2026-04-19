# commands.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: Click command implementations for prepare, train, validate, verify

from __future__ import annotations

import logging
from pathlib import Path

import click
import torch
from torch.utils.data import DataLoader

from cephnet.config.loader import load_config
from cephnet.config.schema import HeatmapModelConfig
from cephnet.logging.logger import setup_logging
from cephnet.utils.seeding import seed_everything


@click.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Training config YAML")
@click.option("--dataset", "-d", default=None, help="Override dataset name")
@click.option("--force", is_flag=True, default=False, help="Overwrite existing dataset")
@click.option("--synthetic", is_flag=True, default=False, help="Always generate synthetic data (skip download)")
@click.option("--n-train", default=80, show_default=True, help="Synthetic training samples")
@click.option("--n-val", default=20, show_default=True, help="Synthetic validation samples")
@click.option("--n-test", default=20, show_default=True, help="Synthetic test samples")
def prepare(
    config: str,
    dataset: str | None,
    force: bool,
    synthetic: bool,
    n_train: int,
    n_val: int,
    n_test: int,
) -> None:
    """📦 Download and prepare datasets for training.

    Attempts to download the real dataset from known public sources.
    Falls back to generating realistic synthetic data if download fails.
    Use --synthetic to skip the download attempt entirely.
    """
    from cephnet.datasets.preparer import DatasetPreparer, generate_synthetic_dataset

    cfg = load_config(config)
    log_dir = cfg.logging.log_dir if cfg.logging.file else None
    setup_logging(cfg.logging.level, log_dir)
    log = logging.getLogger(__name__)

    ds_name = dataset or cfg.dataset.name
    root = Path(str(cfg.dataset.root))
    raw_root = Path(str(cfg.dataset.raw_root)) if cfg.dataset.raw_root else None
    img_size: tuple[int, int] = (cfg.dataset.image_size[0], cfg.dataset.image_size[1])

    log.info("🚀 Preparing dataset: %s → %s", ds_name, root)

    if synthetic:
        log.info("🧪 --synthetic flag: skipping download, generating synthetic data")
        generate_synthetic_dataset(
            root,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            image_size=img_size,
            force=force,
        )
    else:
        preparer = DatasetPreparer(ds_name, root, raw_root)
        preparer.prepare(
            image_size=img_size,
            force=force,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
        )

    # Summary
    ann = root / "annotations" / "landmarks.csv"
    if ann.exists():
        import pandas as pd
        df = pd.read_csv(ann)
        log.info("✅ Dataset ready — %d total samples in %s", len(df), root)
        splits = df["split"].value_counts().to_dict() if "split" in df.columns else {}
        for sp, cnt in splits.items():
            log.info("   %s: %d samples", sp, cnt)
    else:
        log.error("❌ Dataset preparation failed — %s not found", ann)


@click.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Training config YAML")
@click.option("--resume", is_flag=True, default=False, help="Resume from last checkpoint")
def train(config: str, resume: bool) -> None:
    """🚀 Train a landmark detection model."""
    cfg = load_config(config)
    log_dir = cfg.logging.log_dir if cfg.logging.file else None
    setup_logging(cfg.logging.level, log_dir)
    seed_everything(cfg.training.seed)

    log = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("🚀 Training [bold]%s[/bold] on device: %s", cfg.training.experiment_name, device)

    from cephnet.augmentations.pipeline import build_augmentation_pipeline, build_val_pipeline
    from cephnet.datasets.factory import get_dataset
    from cephnet.engines.checkpointer import Checkpointer
    from cephnet.engines.trainer import Trainer
    from cephnet.io.writers import MetricsWriter
    from cephnet.models.heatmap import HeatmapModel
    from cephnet.models.regression import RegressionModel, build_model

    img_size: tuple[int, int] = (cfg.dataset.image_size[0], cfg.dataset.image_size[1])
    train_tfm = build_augmentation_pipeline(cfg.augmentation, img_size)
    val_tfm = build_val_pipeline(img_size)

    train_ds = get_dataset(cfg.dataset.name, cfg.dataset.root, "train", transform=train_tfm)
    val_ds = get_dataset(cfg.dataset.name, cfg.dataset.root, "val", transform=val_tfm)
    log.info("📊 Train: %d samples | Val: %d samples", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    model = build_model(cfg.model)
    checkpointer = Checkpointer(cfg.training.checkpoint_dir, cfg.training.experiment_name)

    if resume:
        try:
            checkpointer.load(model)
            log.info("📂 Resumed from checkpoint")
        except FileNotFoundError:
            log.warning("⚠️  No checkpoint found — starting fresh")

    metrics_dir = Path(str(cfg.logging.log_dir)) / "metrics"
    metrics_writer = MetricsWriter(metrics_dir, cfg.training.experiment_name)

    trainer = Trainer(cfg, model, train_loader, val_loader, device, metrics_writer, checkpointer)
    trainer.train()

    # --- Post-training: ONNX export ---
    from cephnet.io.exporter import export_to_onnx
    img_size_tuple: tuple[int, int] = (cfg.dataset.image_size[0], cfg.dataset.image_size[1])
    try:
        if checkpointer is not None:
            checkpointer.load(model)
        export_to_onnx(
            model,
            export_dir=Path(str(cfg.training.export_dir)),
            experiment_name=cfg.training.experiment_name,
            image_size=img_size_tuple,
            device=device,
        )
    except Exception as exc:
        log.warning("⚠️  ONNX export failed: %s", exc)

    # --- Post-training: HTML report with learning curve ---
    from cephnet.io.writers import ReportWriter
    report_dir = Path("/reports/metrics")
    writer = ReportWriter(report_dir)
    history = metrics_writer.get_history()
    summary = {
        "experiment": cfg.training.experiment_name,
        "best_mre_mm": checkpointer.best_metric if checkpointer else None,
    }
    if history:
        last = history[-1]
        summary.update({k: v for k, v in last.items() if k != "epoch"})
    writer.write_summary(summary, cfg.training.experiment_name)
    writer.write_html_report(
        experiment_name=cfg.training.experiment_name,
        summary=summary,
        metrics_history=history,
        title=f"Training Report — {cfg.training.experiment_name}",
    )
    log.info("✅ Training artifacts exported")


@click.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Training config YAML")
@click.option("--checkpoint", default=None, help="Path to specific checkpoint file")
def validate(config: str, checkpoint: str | None) -> None:
    """🧪 Validate a trained model on the test split."""
    cfg = load_config(config)
    log_dir = cfg.logging.log_dir if cfg.logging.file else None
    setup_logging(cfg.logging.level, log_dir)
    log = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from cephnet.augmentations.pipeline import build_val_pipeline
    from cephnet.datasets.factory import get_dataset
    from cephnet.engines.checkpointer import Checkpointer
    from cephnet.engines.validator import Validator
    from cephnet.io.writers import ReportWriter
    from cephnet.models.regression import build_model

    img_size: tuple[int, int] = (cfg.dataset.image_size[0], cfg.dataset.image_size[1])
    test_ds = get_dataset(cfg.dataset.name, cfg.dataset.root, "test", transform=build_val_pipeline(img_size))
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False)
    log.info("🧪 Test split: %d samples", len(test_ds))

    model = build_model(cfg.model)
    ckpt = Checkpointer(cfg.training.checkpoint_dir, cfg.training.experiment_name)
    ckpt_path = Path(checkpoint) if checkpoint else None
    try:
        ckpt.load(model, checkpoint_path=ckpt_path)
    except FileNotFoundError:
        log.warning("⚠️  No checkpoint found — validating with random weights")

    heatmap_size = cfg.model.heatmap_size if isinstance(cfg.model, HeatmapModelConfig) else None
    validator = Validator(model, device, cfg.dataset.pixel_spacing_mm, img_size, heatmap_size)
    metrics = validator.evaluate(test_loader)
    metrics.log_summary(log)

    # Write reports
    report_dir = Path("/reports/metrics")
    writer = ReportWriter(report_dir)
    summary_dict = {"experiment": cfg.training.experiment_name, **metrics.to_dict()}
    writer.write_summary(summary_dict, cfg.training.experiment_name)

    # Landmark table — use config names if provided, else numbered fallback
    lm_names: list[str] | None = list(cfg.dataset.landmark_names) if cfg.dataset.landmark_names else None
    per_lm: list[float] | None = None
    if metrics.per_landmark_mre is not None:
        per_lm = metrics.per_landmark_mre.tolist()
        n = len(per_lm)
        if lm_names is None:
            lm_names = [f"L{i + 1:02d}" for i in range(n)]
        writer.write_landmark_table(lm_names, per_lm, cfg.training.experiment_name)

    # HTML report
    writer.write_html_report(
        experiment_name=cfg.training.experiment_name,
        summary=summary_dict,
        landmark_names=lm_names,
        per_landmark_mre=per_lm,
        title=f"Validation Report — {cfg.training.experiment_name}",
    )
    log.info("✅ Validation complete")


@click.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Training config YAML")
@click.option("--verification-dataset", "-v", default="aariz", show_default=True)
@click.option("--verification-root", default="/data/processed/aariz", show_default=True)
@click.option("--checkpoint", default=None, help="Path to specific checkpoint file")
def verify(
    config: str,
    verification_dataset: str,
    verification_root: str,
    checkpoint: str | None,
) -> None:
    """🔬 Cross-dataset verification: evaluate on a different dataset."""
    cfg = load_config(config)
    log_dir = cfg.logging.log_dir if cfg.logging.file else None
    setup_logging(cfg.logging.level, log_dir)
    log = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from cephnet.augmentations.pipeline import build_val_pipeline
    from cephnet.datasets.factory import get_dataset
    from cephnet.engines.checkpointer import Checkpointer
    from cephnet.evaluation.cross_dataset import CrossDatasetEvaluator
    from cephnet.models.regression import build_model

    img_size: tuple[int, int] = (cfg.dataset.image_size[0], cfg.dataset.image_size[1])
    ver_ds = get_dataset(
        verification_dataset, Path(verification_root), "test",
        transform=build_val_pipeline(img_size),
    )
    ver_loader = DataLoader(ver_ds, batch_size=cfg.training.batch_size, shuffle=False)
    log.info("🔬 Verification dataset: %s — %d samples", verification_dataset, len(ver_ds))

    model = build_model(cfg.model)
    ckpt = Checkpointer(cfg.training.checkpoint_dir, cfg.training.experiment_name)
    ckpt_path = Path(checkpoint) if checkpoint else None
    try:
        ckpt.load(model, checkpoint_path=ckpt_path)
    except FileNotFoundError:
        log.warning("⚠️  No checkpoint — verifying with random weights")

    heatmap_size = cfg.model.heatmap_size if isinstance(cfg.model, HeatmapModelConfig) else None
    evaluator = CrossDatasetEvaluator(
        model=model,
        device=device,
        report_dir=Path("/reports/verification"),
        experiment_name=cfg.training.experiment_name,
    )
    evaluator.evaluate(
        ver_loader,
        pixel_spacing_mm=cfg.dataset.pixel_spacing_mm,
        image_size=img_size,
        heatmap_size=heatmap_size,
        landmark_names=cfg.dataset.landmark_names or None,
        dataset_name=verification_dataset,
    )
