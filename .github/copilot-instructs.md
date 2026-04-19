\## Copilot Instructions — Dockerized Training Program



\### Mission

Build only the \*\*training subsystem\*\* for cephalometric landmark detection. The program must be Docker-first, reproducible, cross-platform at the source level, and operable through OS scripts that only call Docker containers; no training logic should run directly on the host. The system must train, validate, and verify models on public cephalometric datasets and export artifacts and reports to host-mounted folders. Public cephalometric benchmark datasets and cross-dataset verification are essential for realistic evaluation. \[github](https://github.com/manwaarkhd/aariz-cephalometric-dataset)



\### Scope

Generate only code, scripts, configs, tests, and docs related to:

\- dataset download and preparation;

\- model training;

\- validation and cross-dataset verification;

\- artifact export;

\- report generation;

\- Docker/Dev Container workflow.



Do not generate GUI, end-user desktop packaging, or installer code.



\### Repository Layout

Use this exact structure:



```text

cephnet/

├─ LICENSE

├─ README.md

├─ pyproject.toml

├─ uv.lock

├─ requirements/

│  ├─ base.txt

│  ├─ training.txt

│  ├─ dev.txt

│  ├─ packaging.txt

│  └─ extras.txt

├─ .gitignore

├─ .gitattributes

├─ .editorconfig

├─ .flake8

├─ ruff.toml

├─ mypy.ini

├─ pytest.ini

├─ .python-version

├─ .env.example

├─ .devcontainer/

│  ├─ devcontainer.json

│  └─ Dockerfile

├─ containers/

│  ├─ base/

│  │  ├─ Dockerfile

│  │  └─ entrypoint.sh

│  ├─ cpu/

│  │  └─ Dockerfile

│  ├─ gpu/

│  │  └─ Dockerfile

│  ├─ compose/

│  │  ├─ docker-compose.cpu.yml

│  │  ├─ docker-compose.gpu.yml

│  │  ├─ docker-compose.mlflow.yml

│  │  └─ docker-compose.dev.yml

│  └─ scripts/

│     ├─ prepare.sh

│     ├─ train.sh

│     ├─ validate.sh

│     └─ verify.sh

├─ configs/

│  ├─ datasets/

│  ├─ training/

│  ├─ models/

│  ├─ logging/

│  └─ runtime/

├─ data/

│  ├─ raw/

│  ├─ interim/

│  ├─ processed/

│  └─ external/

├─ models/

│  ├─ checkpoints/

│  ├─ exports/

│  └─ artifacts/

├─ reports/

│  ├─ metrics/

│  ├─ plots/

│  └─ verification/

├─ src/

│  └─ cephnet/

│     ├─ \_\_init\_\_.py

│     ├─ cli/

│     ├─ config/

│     ├─ datasets/

│     ├─ augmentations/

│     ├─ models/

│     ├─ losses/

│     ├─ metrics/

│     ├─ engines/

│     ├─ evaluation/

│     ├─ logging/

│     ├─ utils/

│     └─ io/

├─ scripts/

│  ├─ python/

│  │  ├─ prepare\_dataset.py

│  │  ├─ train\_model.py

│  │  ├─ validate\_model.py

│  │  └─ verify\_cross\_dataset.py

│  ├─ windows/

│  │  ├─ run\_prepare.ps1

│  │  ├─ run\_train.ps1

│  │  ├─ run\_validate.ps1

│  │  └─ run\_verify.ps1

│  ├─ macos/

│  │  ├─ run\_prepare.sh

│  │  ├─ run\_train.sh

│  │  ├─ run\_validate.sh

│  │  └─ run\_verify.sh

│  └─ linux/

│     ├─ run\_prepare.sh

│     ├─ run\_train.sh

│     ├─ run\_validate.sh

│     └─ run\_verify.sh

├─ tests/

│  ├─ unit/

│  ├─ integration/

│  ├─ verification/

│  └─ fixtures/

├─ tools/

│  ├─ dataset\_tools/

│  ├─ quality/

│  └─ release/

└─ .github/

&#x20;  └─ workflows/

```



\### Core Principles

\- Keep all runtime training execution inside Docker containers.

\- The `scripts/windows`, `scripts/macos`, and `scripts/linux` wrappers must do nothing except invoke `docker` or `docker compose`.

\- Use `containers/base/` as the shared base layer for common dependencies.

\- Derive `containers/cpu/` and `containers/gpu/` from the base image.

\- Add a `.devcontainer/` setup that points to the same container stack as development.

\- Use `uv` for dependency management, environment sync, and command execution inside containers. uv provides project management, lockfiles, and reproducible `uv run` workflows. \[docs.astral](https://docs.astral.sh/uv/guides/projects/)

\- Prefer host-mounted directories for datasets, models, and reports so artifacts remain outside the container filesystem.



\### Docker and DevContainer Rules

\- The repository must be Docker-first, not Docker-optional.

\- The developer should be able to open the repository in a dev container using the same stack as the training environment. VS Code dev containers support Dockerfile and Docker Compose-based workflows, and `devcontainer.json` can point to a container service or build context. \[code.visualstudio](https://code.visualstudio.com/docs/devcontainers/create-dev-container)

\- Provide a base development image and use compose overrides for CPU, GPU, and optional MLflow services.

\- Scripts in OS folders must pass through Docker Compose, for example:

&#x20; - `docker compose -f containers/compose/docker-compose.cpu.yml run --rm trainer uv run ...`

&#x20; - `docker compose -f containers/compose/docker-compose.gpu.yml run --rm trainer uv run ...`

\- Do not require direct host-side Python installation for training.



\### Data and Artifact Routing

The host machine must contain these folders and they must be mounted into the container:

\- `data/` for inputs and processed datasets;

\- `models/` for checkpoints, exported artifacts, and serialized weights;

\- `reports/` for metrics, charts, verification results, and logs summaries.



The application must never write production artifacts only inside the container layer. Docker workflows for ML should bind-mount data and outputs so they persist on the host. \[apxml](https://apxml.com/courses/docker-for-ml-projects/chapter-3-managing-ml-data-containers/practice-mounting-data-saving-models)



\### .gitignore Requirements

Add `.gitignore` entries for at least:

\- `data/raw/`

\- `data/interim/`

\- `data/processed/`

\- `data/external/`

\- `models/`

\- `reports/`

\- `.venv/`

\- `uv.lock` only if project policy requires excluding it; otherwise commit it

\- cache directories

\- build directories

\- temporary logs

\- notebook checkpoints

\- OS junk files



\### .gitattributes Requirements

Create `.gitattributes` to:

\- normalize line endings;

\- mark generated files as `linguist-generated`;

\- keep documentation and scripts properly classified;

\- optionally force LF for shell scripts and Docker-related text files.  

GitHub documents `.gitattributes` as the correct place to control diff behavior and language statistics with `linguist-generated` attributes. \[docs.github](https://docs.github.com/en/repositories/working-with-files/managing-files/customizing-how-changed-files-appear-on-github)



\### Environment and Dependency Management

Use `uv` as the primary project tool.

\- `pyproject.toml` is the source of truth.

\- `requirements/base.txt`, `requirements/training.txt`, `requirements/dev.txt`, `requirements/packaging.txt`, and `requirements/extras.txt` should exist for interoperability and explicit pinned exports.

\- Keep `uv.lock` committed.

\- Provide `.python-version`.

\- Use `uv sync` in the container build or entrypoint where appropriate.

\- Use `uv run` for commands in scripts and container entrypoints. uv’s project model and lockfile workflow are designed for reproducible Python environments. \[docs.astral](https://docs.astral.sh/uv/)



\### Logging Requirements

The training application must be verbose and professional.

\- Use structured logs.

\- Add ANSI-colored terminal output.

\- Include tasteful emoji in operational messages, for example `🚀`, `🧪`, `✅`, `⚠️`.

\- Log dataset loading, batch processing, epoch progress, metrics, checkpoint saving, validation, and verification clearly.

\- Save logs to `reports/` or `models/` only if needed for run artifacts; otherwise keep them in `reports/logs` or `reports/verification`.



Use Rich or `coloredlogs` for colorized terminal output. Colorized logging and emojis are a practical way to make long-running training observable in terminal sessions. \[coloredlogs.readthedocs](https://coloredlogs.readthedocs.io/en/latest/readme.html)



\### Best Practices

Enforce these rules everywhere:

\- Type hints everywhere.

\- PEP 8 and PEP 257.

\- Single-responsibility modules.

\- No business logic inside scripts if it belongs in `src/`.

\- No hardcoded dataset paths.

\- Use `pathlib.Path`.

\- Config-driven workflows through YAML.

\- Deterministic seeding.

\- Fail fast on invalid configs.

\- Keep dataset adapters independent from model code.

\- Keep evaluation and reporting deterministic.

\- Use dataclasses or pydantic models for DTOs and config validation.

\- Keep CLI thin and orchestration-focused.

\- Use pytest for tests and make them fast.



\### Dataset Handling

Support at least two datasets:

\- one primary training dataset;

\- one distinct verification dataset.



The training program must:

\- download or prepare the dataset into `data/raw/`;

\- transform it into `data/interim/` and `data/processed/`;

\- verify annotation integrity;

\- map dataset-specific landmarks into a canonical schema.



\### Training Features

Implement:

\- heatmap keypoint baseline;

\- coordinate regression baseline;

\- optional coarse-to-fine extension;

\- checkpointing;

\- resume;

\- early stopping;

\- mixed precision when available;

\- gradient clipping;

\- scheduler support;

\- experiment metadata;

\- exportable inference artifacts.



\### Metrics and Reports

Implement:

\- MRE;

\- SDR at 2.0, 2.5, 3.0, and 4.0 mm;

\- per-landmark error tables;

\- confusion/quality summaries if applicable;

\- cross-dataset drift analysis;

\- learning-curve plots.



Export reports to `reports/` and artifacts to `models/`.



\### Cross-Dataset Verification

The verification flow must:

\- train on dataset A;

\- validate on dataset B;

\- generate a formal report;

\- save outputs into `reports/verification/`;

\- avoid contamination between splits.



\### Containers

Create at least these containers:

\- `base`: shared dependencies and tools.

\- `cpu`: training image for CPU.

\- `gpu`: training image for CUDA-enabled hosts.

\- optional `mlflow`: tracking stack if included.

\- optional `dev`: development container.



Each container must expose a consistent working directory and mount points for `data/`, `models/`, and `reports/`.



\### OS Wrapper Scripts

The scripts in:

\- `scripts/windows/`

\- `scripts/macos/`

\- `scripts/linux/`



must:

\- call Docker only;

\- not install Python locally;

\- not run the training directly on host;

\- select the correct compose file;

\- forward any user-provided config paths into the container;

\- return the container exit code to the shell.



\### Documentation Requirements

Add docs explaining:

\- Docker workflow;

\- uv usage;

\- CPU and GPU modes;

\- devcontainer setup;

\- how models and reports are exported to host directories;

\- how to run training, validation, and verification;

\- how to rebuild images;

\- how to clean caches safely.



\### File Header Standard

Every source file must start with an English header containing:

\- File name;

\- MIT license;

\- Author: Piergiorgio Vagnozzi;

\- Creation date;

\- Last modification date;

\- English description.



\### Acceptance Checklist

Complete only if all are true:

\- \[ ] Scripts in `scripts/windows`, `scripts/macos`, `scripts/linux` invoke Docker only.

\- \[ ] `containers/` replaces `docker/`.

\- \[ ] Base, CPU, GPU, and optional dev/mlflow containers exist.

\- \[ ] Devcontainer exists and uses the same stack.

\- \[ ] `uv` is the main project manager.

\- \[ ] `requirements/` files exist.

\- \[ ] `models/` and `reports/` are host-mounted and gitignored.

\- \[ ] `.gitignore`, `.gitattributes`, `.editorconfig`, `.flake8`, `ruff.toml`, `mypy.ini`, and `pytest.ini` exist.

\- \[ ] Logging is verbose, colored, and emoji-enhanced.

\- \[ ] Dataset download and preprocessing write into the expected folders.

\- \[ ] Training, validation, and cross-dataset verification are implemented.

\- \[ ] Tests exist and run in Docker.



\### Copilot Behavior Rules

When generating code:

\- generate full files, not fragments;

\- keep path names stable;

\- prefer maintainable architecture over clever shortcuts;

\- emit Docker and config files before implementation files;

\- include comments only when they add clarity;

\- keep all comments in English;

\- keep logs user-friendly and professional.



