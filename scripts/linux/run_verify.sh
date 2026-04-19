#!/usr/bin/env bash
# run_verify.sh
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-19
# Description: Linux wrapper — runs cross-dataset verification via Docker Compose.
#              All logic runs INSIDE Docker. Host receives only:
#                reports/verification/ → drift analysis, cross-dataset report

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=scripts/linux/_common.sh
source "${SCRIPT_DIR}/_common.sh"

ensure_artifact_dirs "${REPO_ROOT}"
COMPOSE_FILE="$(get_compose_file "${REPO_ROOT}" "$@")"

echo "🔍 Running cross-dataset verification..."
echo "   Reports → ${REPO_ROOT}/reports/verification"

exec docker compose \
    -f "${COMPOSE_FILE}" \
    --project-directory "${REPO_ROOT}" \
    run --rm trainer \
    uv run python scripts/python/verify_cross_dataset.py "$@"
