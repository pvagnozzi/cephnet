#!/usr/bin/env bash
# run_train.sh
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-19
# Description: Linux wrapper — runs cephnet training via Docker Compose (CPU or GPU).
#              All training logic runs INSIDE Docker. Host receives only:
#                models/  → checkpoints, ONNX exports
#                reports/ → metrics, plots, logs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=scripts/linux/_common.sh
source "${SCRIPT_DIR}/_common.sh"

ensure_artifact_dirs "${REPO_ROOT}"
COMPOSE_FILE="$(get_compose_file "${REPO_ROOT}" "$@")"

echo "🚀 Starting training..."
echo "   Artifacts → ${REPO_ROOT}/models"
echo "   Reports   → ${REPO_ROOT}/reports"
echo "   Data      → Docker volume: cephnet-data"

exec docker compose \
    -f "${COMPOSE_FILE}" \
    --project-directory "${REPO_ROOT}" \
    run --rm trainer \
    uv run python scripts/python/train_model.py "$@"
