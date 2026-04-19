# cephnet

> **Cephalometric Landmark Detection — Training Subsystem**

A Docker-first, reproducible training pipeline for cephalometric landmark detection.
All training logic runs inside Docker containers; OS wrapper scripts only call Docker.

[![Version](https://img.shields.io/github/v/release/pvagnozzi/cephnet?label=version&color=brightgreen&logo=github)](https://github.com/pvagnozzi/cephnet/releases/latest)
[![CI](https://github.com/pvagnozzi/cephnet/actions/workflows/ci.yml/badge.svg)](https://github.com/pvagnozzi/cephnet/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![GitFlow](https://img.shields.io/badge/branching-GitFlow-orange.svg)](https://nvie.com/posts/a-successful-git-branching-model/)

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Branching Strategy](#branching-strategy)
- [Running Training](#running-training)
- [Dataset Preparation](#dataset-preparation)
- [Validation & Verification](#validation--verification)
- [Running Tests](#running-tests)
- [Rebuilding Docker Images](#rebuilding-docker-images)
- [Environment Variables](#environment-variables)
- [Datasets](#datasets)
- [Releases](#releases)

---

## Overview

`cephnet` trains deep learning models to detect cephalometric landmarks on lateral skull X-rays.

**Key features:**

- 🐳 Docker-first: all training runs inside containers — no host Python for training
- 🔁 Reproducible: deterministic seeding, pinned dependencies via `uv.lock`
- 📊 Metrics: MRE, SDR at 2.0 / 2.5 / 3.0 / 4.0 mm, per-landmark error tables
- 🧪 Cross-dataset verification: train on ISBI 2015, verify on Aariz (or vice versa)
- 📦 Exportable: inference artifacts written to `models/exports/`
- 📈 MLflow-ready: optional experiment tracking

---

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) ≥ 24
- [Docker Compose](https://docs.docker.com/compose/) v2
- (Optional) NVIDIA GPU + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### 1. Clone and configure

```bash
git clone https://github.com/pvagnozzi/cephnet.git
cd cephnet
cp .env.example .env
# Edit .env as needed
```

### 2. Prepare datasets

```bash
# Linux / macOS
scripts/linux/run_prepare.sh

# Windows (PowerShell)
scripts\windows\run_prepare.ps1
```

### 3. Train (CPU)

```bash
# Linux / macOS
scripts/linux/run_train.sh

# Windows (PowerShell)
scripts\windows\run_train.ps1
```

### 4. Train with custom config

```bash
# Linux / macOS
scripts/linux/run_train.sh --config configs/training/default.yaml

# Windows (PowerShell)
scripts\windows\run_train.ps1 -Config configs/training/default.yaml
```

---

## Branching Strategy

This project follows **[GitFlow](https://nvie.com/posts/a-successful-git-branching-model/)** with automatic semantic versioning via [GitVersion](https://gitversion.net/).

| Branch | Purpose | Protection |
|--------|---------|------------|
| `main` | Production-ready releases | ✅ PR required · 1 approval · CI pass |
| `dev` | Active development (≡ develop) | ✅ PR required · 1 approval · CI pass |
| `feature/**` | New features | ✅ PR required · CI pass |
| `release/**` | Release preparation | ✅ PR required · 1 approval · CI pass |
| `hotfix/**` | Production hotfixes | ✅ PR required · 1 approval · CI pass |

**Version scheme:** `MAJOR.MINOR.PATCH[-pre]`
- PR merged to `main` → stable release (e.g. `1.2.0`)
- PR merged to `dev` → pre-release (e.g. `1.2.0-alpha.3`)

To configure branch protection (requires repo admin + `gh` CLI):

```bash
# Linux / macOS
scripts/linux/setup-branch-protection.sh

# Windows (PowerShell)
scripts\windows\setup-branch-protection.ps1
```

---

## Releases

Every PR merged to `main` or `dev` automatically:

1. 🏷️ Computes the version via **GitVersion** (GitFlow mode)
2. 📋 Generates a **CHANGELOG.md** from git history (emoji, grouped by type)
3. 📦 Publishes a **GitHub Release** with a zip artifact:
   ```
   cephnet-vX.Y.Z/
   ├── CHANGELOG.md        ← full git history changelog
   ├── models/README.md    ← placeholder for trained model artifacts
   └── reports/README.md   ← placeholder for training reports
   ```
4. 🔖 Updates the version badge in this README

Browse all releases: **[github.com/pvagnozzi/cephnet/releases](https://github.com/pvagnozzi/cephnet/releases)**

---



```
cephnet/
├── containers/
│   ├── base/               # Shared base image (uv, Python, common deps)
│   ├── cpu/                # CPU-only training image
│   ├── gpu/                # CUDA training image
│   └── compose/            # docker-compose files
├── .devcontainer/          # VS Code dev container
├── src/cephnet/
│   ├── cli/                # Thin CLI, orchestration only
│   ├── config/             # Config loading/validation (Pydantic)
│   ├── datasets/           # Dataset adapters
│   ├── augmentations/      # Augmentation pipeline
│   ├── models/             # Model definitions
│   ├── losses/             # Loss functions
│   ├── metrics/            # MRE, SDR, per-landmark tables
│   ├── engines/            # Training loop, validation, checkpointing
│   ├── evaluation/         # Cross-dataset verification, drift analysis
│   ├── logging/            # Structured + colored + emoji logging
│   ├── utils/              # Seeding, pathlib helpers
│   └── io/                 # Artifact/report writers
├── scripts/
│   ├── python/             # Entry-point scripts (delegate to src/)
│   ├── linux/              # Linux OS wrappers (Docker only)
│   ├── macos/              # macOS OS wrappers (Docker only)
│   └── windows/            # Windows OS wrappers (Docker only)
├── configs/
│   ├── training/           # Training YAML configs
│   ├── datasets/           # Dataset configs
│   ├── models/             # Model architecture configs
│   └── logging/            # Logging configs
├── data/                   # Host-mounted (gitignored)
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
├── models/                 # Host-mounted (gitignored)
│   ├── checkpoints/
│   ├── exports/
│   └── artifacts/
├── reports/                # Host-mounted (gitignored)
│   ├── metrics/
│   ├── plots/
│   ├── verification/
│   └── logs/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── verification/
│   └── fixtures/
├── tools/
│   ├── dataset_tools/
│   ├── quality/
│   └── release/
└── requirements/
    ├── base.txt
    ├── training.txt
    ├── dev.txt
    ├── packaging.txt
    └── extras.txt
```

---

## Running Training

All training runs **inside Docker**. The OS wrapper scripts handle container lifecycle.

### CPU training

```bash
# Linux / macOS
scripts/linux/run_train.sh
scripts/linux/run_train.sh --config configs/training/resnet50_heatmap.yaml

# macOS
scripts/macos/run_train.sh --config configs/training/default.yaml

# Windows (PowerShell)
scripts\windows\run_train.ps1
scripts\windows\run_train.ps1 -Config configs/training/resnet50_heatmap.yaml
```

### GPU training

The same scripts automatically select the GPU compose file when a GPU is detected:

```bash
scripts/linux/run_train.sh --config configs/training/default.yaml
```

### Inside the container (advanced)

If you need to run the training script directly inside a container shell:

```bash
docker compose -f containers/compose/docker-compose.cpu.yml run --rm trainer bash
# Inside the container:
uv run python scripts/python/train_model.py --config configs/training/default.yaml
```

---

## Dataset Preparation

Downloads, verifies, and preprocesses datasets into `data/processed/`.

```bash
# Linux / macOS
scripts/linux/run_prepare.sh

# Windows
scripts\windows\run_prepare.ps1
```

**Data flow:** `data/raw/` → `data/interim/` → `data/processed/`

---

## Validation & Verification

### Validate model performance

```bash
scripts/linux/run_validate.sh
```

Computes MRE and SDR metrics on the held-out test split.

### Cross-dataset verification

```bash
scripts/linux/run_verify.sh
```

Evaluates a model trained on one dataset against a different dataset to measure
distribution shift and landmark detection drift.

Reports are written to `reports/verification/`.

---

## Running Tests

Tests run **inside the dev container** or locally with `uv`.

```bash
# All unit tests
uv run pytest tests/unit/

# Single test file
uv run pytest tests/unit/test_metrics.py

# Single test by name
uv run pytest tests/ -k "test_mre_calculation"

# With coverage
uv run pytest tests/unit/ --cov=src/cephnet --cov-report=term-missing

# Skip slow tests
uv run pytest tests/ -m "not slow"
```

### Linting & type checking

```bash
uv run ruff check src/
uv run ruff format --check src/
uv run mypy src/
```

---

## Rebuilding Docker Images

```bash
# CPU image
docker compose -f containers/compose/docker-compose.cpu.yml build

# GPU image
docker compose -f containers/compose/docker-compose.gpu.yml build

# Base image only
docker build -t cephnet-base containers/base/

# Force rebuild (no cache)
docker compose -f containers/compose/docker-compose.cpu.yml build --no-cache
```

---

## Environment Variables

Copy `.env.example` to `.env` and adjust as needed.

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `./data` | Host path for data (raw/interim/processed) |
| `MODELS_DIR` | `./models` | Host path for model checkpoints and exports |
| `REPORTS_DIR` | `./reports` | Host path for metrics, plots, logs |
| `CEPHNET_CONFIG` | `configs/training/default.yaml` | Training config YAML path |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URI (optional) |
| `MLFLOW_EXPERIMENT_NAME` | `cephnet` | MLflow experiment name |
| `CEPHNET_SEED` | `42` | Global random seed for reproducibility |

---

## Datasets

### ISBI 2015 — Cephalometric Landmark Detection Challenge

- **Source:** [ISBI 2015 Grand Challenge](http://www-o.ntust.edu.tw/~cweiwang/ISBI2015/)
- **Images:** 400 lateral cephalograms (150 train / 250 test)
- **Landmarks:** 19 canonical cephalometric landmarks
- **Resolution:** 1935 × 2400 px, ~0.1 mm/pixel
- **Format:** BMP images + TXT annotation files

Place raw files under `data/raw/isbi2015/` before running `run_prepare.sh`.

### Aariz — Cephalometric Landmark Dataset

- **Source:** [Aariz Dataset](https://github.com/aarizimam/Aariz)
- **Images:** Lateral cephalograms with 19 landmark annotations
- **Format:** JPEG images + JSON annotations
- **Use:** Cross-dataset verification / drift analysis

Place raw files under `data/raw/aariz/` before running `run_prepare.sh`.

---

## License

MIT License — Copyright (c) 2026 Piergiorgio Vagnozzi. See [LICENSE](./LICENSE).
