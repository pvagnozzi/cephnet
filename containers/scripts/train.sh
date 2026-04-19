#!/bin/bash
# containers/scripts/train.sh
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2024-01-01
# Last modified: 2024-01-01
# Description: Runs model training inside the container via uv run.

set -euo pipefail

uv run python scripts/python/train_model.py "$@"
