#!/usr/bin/env bash
# run_prepare.sh
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-19
# Description: macOS wrapper — prepares both the training dataset and the Aariz
#              verification dataset via Docker Compose. All logic runs INSIDE Docker.
#              Data is written to the cephnet-data named volume (NOT the host).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=scripts/macos/_common.sh
source "${SCRIPT_DIR}/_common.sh"

ensure_artifact_dirs "${REPO_ROOT}"
COMPOSE_FILE="${REPO_ROOT}/containers/compose/docker-compose.cpu.yml"

# --- Step 1: Prepare training dataset (all user args forwarded) ---
echo "📦 Step 1/2 — Preparing training dataset..."
echo "   Data → Docker volume: cephnet-data"
docker compose \
    -f "${COMPOSE_FILE}" \
    --project-directory "${REPO_ROOT}" \
    run --rm trainer \
    uv run python scripts/python/prepare_dataset.py "$@"

# --- Step 2: Prepare Aariz verification dataset ---
AARIZ_EXTRA=()
for arg in "$@"; do
    case "${arg}" in
        --force|--synthetic) AARIZ_EXTRA+=("${arg}") ;;
    esac
done

echo "📦 Step 2/2 — Preparing Aariz verification dataset..."
docker compose \
    -f "${COMPOSE_FILE}" \
    --project-directory "${REPO_ROOT}" \
    run --rm trainer \
    uv run python scripts/python/prepare_dataset.py \
    --config configs/training/verify_aariz.yaml \
    --n-train 0 --n-val 0 --n-test 50 \
    "${AARIZ_EXTRA[@]+"${AARIZ_EXTRA[@]}"}"

