#!/bin/bash
# containers/scripts/verify.sh
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2024-01-01
# Last modified: 2024-01-01
# Description: Runs cross-dataset verification inside the container via uv run.

set -euo pipefail

uv run python scripts/python/verify_cross_dataset.py "$@"
