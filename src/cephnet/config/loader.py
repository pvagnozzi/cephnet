# loader.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: YAML config loader that validates and returns CephnetConfig

from pathlib import Path

import yaml

from cephnet.config.schema import CephnetConfig


def load_config(config_path: str | Path) -> CephnetConfig:
    """Load and validate a YAML config file, returning a CephnetConfig."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ValueError(f"Empty config file: {path}")
    return CephnetConfig.model_validate(raw)
