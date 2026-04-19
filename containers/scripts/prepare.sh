#!/bin/bash
# containers/scripts/prepare.sh
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2024-01-01
# Last modified: 2024-01-01
# Description: Runs dataset preparation inside the container via uv run.

set -euo pipefail

uv run python scripts/python/prepare_dataset.py "$@"
