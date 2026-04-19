# schema.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: Pydantic v2 configuration schemas for cephnet training pipeline

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


class DatasetConfig(BaseModel):
    name: str
    root: Path
    raw_root: Path | None = None
    interim_root: Path | None = None
    image_size: tuple[int, int] = (512, 512)
    n_landmarks: int = Field(ge=1, le=100)
    pixel_spacing_mm: float = Field(gt=0.0, default=0.1)
    train_ratio: float = Field(ge=0.0, le=1.0, default=0.6)
    val_ratio: float = Field(ge=0.0, le=1.0, default=0.2)
    test_ratio: float = Field(ge=0.0, le=1.0, default=0.2)
    image_format: str = "png"
    annotation_format: str = "csv"
    landmark_names: list[str] = Field(default_factory=list)
    download_url: str | None = None

    @model_validator(mode="after")
    def validate_split_ratios(self) -> DatasetConfig:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")
        return self


class HeatmapModelConfig(BaseModel):
    type: Literal["heatmap"] = "heatmap"
    backbone: str = "resnet50"
    pretrained: bool = True
    n_landmarks: int = Field(ge=1, le=100)
    heatmap_sigma: float = Field(gt=0.0, default=3.0)
    heatmap_size: tuple[int, int] = (128, 128)
    dropout: float = Field(ge=0.0, le=1.0, default=0.0)


class RegressionModelConfig(BaseModel):
    type: Literal["regression"] = "regression"
    backbone: str = "resnet50"
    pretrained: bool = True
    n_landmarks: int = Field(ge=1, le=100)
    dropout: float = Field(ge=0.0, le=1.0, default=0.3)
    hidden_dim: int = Field(ge=64, default=512)


ModelConfig = Annotated[
    HeatmapModelConfig | RegressionModelConfig,
    Field(discriminator="type"),
]


class SchedulerConfig(BaseModel):
    type: Literal["cosine", "reduce_on_plateau", "step", "none"] = "cosine"
    params: dict[str, float | int | str] = Field(default_factory=dict)


class EarlyStoppingConfig(BaseModel):
    enabled: bool = True
    patience: int = Field(ge=1, default=20)
    min_delta: float = Field(ge=0.0, default=0.0001)


class TrainingConfig(BaseModel):
    epochs: int = Field(ge=1, default=100)
    batch_size: int = Field(ge=1, default=8)
    learning_rate: float = Field(gt=0.0, default=1e-4)
    weight_decay: float = Field(ge=0.0, default=1e-4)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    mixed_precision: bool = False
    gradient_clip_norm: float = Field(gt=0.0, default=1.0)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    seed: int = 42
    experiment_name: str = "cephnet_experiment"
    checkpoint_dir: Path = Path("/models/checkpoints")
    export_dir: Path = Path("/models/exports")
    num_workers: int = Field(ge=0, default=2)


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    console: bool = True
    file: bool = True
    log_dir: Path = Path("/reports/logs")
    log_interval_steps: int = Field(ge=1, default=10)
    save_metrics_every_n_epochs: int = Field(ge=1, default=1)
    rich_traceback: bool = True


class AugmentationConfig(BaseModel):
    enabled: bool = True
    p_horizontal_flip: float = Field(ge=0.0, le=1.0, default=0.0)
    p_rotation: float = Field(ge=0.0, le=1.0, default=0.5)
    rotation_limit: float = Field(ge=0.0, default=10.0)
    p_brightness_contrast: float = Field(ge=0.0, le=1.0, default=0.5)
    p_gaussian_noise: float = Field(ge=0.0, le=1.0, default=0.3)
    p_elastic_transform: float = Field(ge=0.0, le=1.0, default=0.2)


class CephnetConfig(BaseModel):
    """Root configuration for a cephnet experiment."""

    dataset: DatasetConfig
    model: HeatmapModelConfig | RegressionModelConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)

    @model_validator(mode="after")
    def validate_n_landmarks_consistency(self) -> CephnetConfig:
        if self.model.n_landmarks != self.dataset.n_landmarks:
            raise ValueError(
                f"model.n_landmarks ({self.model.n_landmarks}) must match "
                f"dataset.n_landmarks ({self.dataset.n_landmarks})"
            )
        return self
