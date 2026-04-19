#!/bin/bash
# run_train.sh
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: Linux wrapper — runs cephnet training via Docker Compose (CPU or GPU)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Select compose file based on --gpu flag
GPU=false
for arg in "$@"; do
    [[ "$arg" == "--gpu" ]] && GPU=true
done

if [[ "$GPU" == "true" ]]; then
    COMPOSE_FILE="${REPO_ROOT}/containers/compose/docker-compose.gpu.yml"
else
    COMPOSE_FILE="${REPO_ROOT}/containers/compose/docker-compose.cpu.yml"
fi

exec docker compose \
    -f "${COMPOSE_FILE}" \
    --project-directory "${REPO_ROOT}" \
    run --rm trainer \
    uv run python scripts/python/train_model.py "$@"
