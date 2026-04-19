#!/usr/bin/env bash
# _common.sh
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-19
# Last modified: 2026-04-19
# Description: Shared helpers for Linux/macOS cephnet wrapper scripts.
#              Source this file with:  source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Ensure host-side artifact directories exist before Docker bind-mounts them.
# data/ is intentionally excluded — it lives in the cephnet-data named volume.
ensure_artifact_dirs() {
    local repo_root="$1"
    local dirs=(
        "models/checkpoints"
        "models/exports"
        "models/artifacts"
        "reports/metrics"
        "reports/plots"
        "reports/verification"
        "reports/logs"
    )
    for d in "${dirs[@]}"; do
        mkdir -p "${repo_root}/${d}"
    done
}

# Resolve the correct compose file: CPU (default) or GPU (--gpu flag present).
get_compose_file() {
    local repo_root="$1"
    shift
    local gpu=false
    for arg in "$@"; do
        [[ "$arg" == "--gpu" ]] && gpu=true
    done

    if [[ "$gpu" == "true" ]]; then
        echo "${repo_root}/containers/compose/docker-compose.gpu.yml"
    else
        echo "${repo_root}/containers/compose/docker-compose.cpu.yml"
    fi
}
