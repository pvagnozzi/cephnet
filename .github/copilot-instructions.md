# Copilot Instructions — cephnet

## Project Purpose

**Training subsystem** for cephalometric landmark detection. Docker-first, reproducible, cross-platform. All training logic runs inside Docker containers; OS wrapper scripts only call Docker. The system trains, validates, and cross-validates models on public cephalometric datasets, exporting artifacts to host-mounted folders.

## Commands

All commands run **inside containers** via OS wrapper scripts. No direct host-side Python execution for training.

```bash
# CPU training
scripts/linux/run_train.sh [--config configs/training/...]
scripts/macos/run_train.sh [--config configs/training/...]
scripts/windows/run_train.ps1 [-Config configs/training/...]

# GPU training
# Same scripts but select docker-compose.gpu.yml internally

# Dataset preparation
scripts/linux/run_prepare.sh
scripts/windows/run_prepare.ps1

# Validation / cross-dataset verification
scripts/linux/run_validate.sh
scripts/linux/run_verify.sh
```

Inside a container (via `uv run`):
```bash
uv run python scripts/python/train_model.py --config configs/training/default.yaml
uv run pytest tests/unit/                         # unit tests only
uv run pytest tests/unit/test_<module>.py         # single test file
uv run pytest tests/ -k "test_name"               # single test by name
uv run ruff check src/
uv run mypy src/
```

## Architecture

```
containers/base/   → shared base image (uv, Python, common deps)
containers/cpu/    → derives from base, CPU-only training
containers/gpu/    → derives from base, CUDA training
containers/compose/ → docker-compose files (cpu, gpu, mlflow, dev)
.devcontainer/     → VS Code dev container (same stack as training)

src/cephnet/
├── cli/           → thin CLI, orchestration only
├── config/        → config loading/validation (Pydantic/dataclasses)
├── datasets/      → dataset adapters (independent from model code)
├── augmentations/ → augmentation pipeline
├── models/        → model definitions
├── losses/        → loss functions
├── metrics/       → MRE, SDR@2/2.5/3/4mm, per-landmark tables
├── engines/       → training loop, validation, checkpointing
├── evaluation/    → cross-dataset verification, drift analysis
├── logging/       → structured + colored + emoji logging
├── utils/         → seeding, pathlib helpers
└── io/            → artifact/report writers

scripts/python/    → entry-point scripts (delegate to src/)
scripts/{os}/      → OS wrappers that call docker/docker-compose only
configs/           → YAML configs for datasets, training, models, logging
data/              → host-mounted; raw/ → interim/ → processed/
models/            → host-mounted; checkpoints/, exports/, artifacts/
reports/           → host-mounted; metrics/, plots/, verification/, logs/
```

**Data flow:** OS script → `docker compose run --rm trainer uv run ...` → Python entry-point in `scripts/python/` → `src/cephnet/` logic → writes to host-mounted `data/`, `models/`, `reports/`.

## Key Conventions

### Docker-first rules
- OS wrapper scripts (`scripts/windows/`, `scripts/macos/`, `scripts/linux/`) must **only** call `docker` or `docker compose`. No Python, no pip, no direct execution on host.
- Example invocation from wrappers: `docker compose -f containers/compose/docker-compose.cpu.yml run --rm trainer uv run ...`
- Forward user-provided config paths into the container; return container exit code to shell.

### Dependency management — `uv`
- `pyproject.toml` is the source of truth.
- `uv.lock` must be committed.
- Use `uv sync` in container build/entrypoint; use `uv run` for all commands.
- `requirements/base.txt`, `training.txt`, `dev.txt`, `packaging.txt`, `extras.txt` exist for interoperability.

### Python style
- Type hints everywhere. PEP 8 + PEP 257.
- Use `pathlib.Path`; no hardcoded paths.
- No business logic in `scripts/python/` — delegate to `src/cephnet/`.
- Config-driven via YAML; fail fast on invalid configs.
- Deterministic seeding in all experiments.
- Dataclasses or Pydantic for DTOs and config validation.
- Single-responsibility modules; dataset adapters independent from model code.

### File header standard
Every source file must start with:
```python
# <filename>
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: YYYY-MM-DD
# Last modified: YYYY-MM-DD
# Description: <English description>
```

### Logging
- Structured logs using Rich or `coloredlogs`.
- ANSI-colored terminal output + tasteful emoji: `🚀 🧪 ✅ ⚠️`.
- Log: dataset loading, batch progress, epoch metrics, checkpointing, validation, verification.
- Artifacts to `reports/logs/` or `reports/verification/` only; never written exclusively inside the container layer.

### Artifacts and mounts
- `data/`, `models/`, `reports/` are host-mounted and gitignored.
- Production artifacts must never be written only to the container layer.

### Dataset handling
- Download/prepare into `data/raw/` → `data/interim/` → `data/processed/`.
- Verify annotation integrity; map dataset-specific landmarks to canonical schema.
- Support at least two datasets (primary training + distinct verification dataset).
- Avoid contamination between training/validation splits.

### Training features
- Heatmap keypoint baseline + coordinate regression baseline.
- Checkpointing, resume, early stopping, mixed precision, gradient clipping, scheduler support.
- Exportable inference artifacts to `models/exports/`.

### Metrics
- MRE, SDR at 2.0 / 2.5 / 3.0 / 4.0 mm, per-landmark error tables.
- Cross-dataset drift analysis, learning-curve plots.
- Reports exported to `reports/`.

### Code generation rules
- Generate **full files**, not fragments.
- Keep path names stable across the session.
- Emit Docker and config files before implementation files.
- Comments in English only; include only when they add clarity.
- Prefer maintainable architecture over clever shortcuts.
