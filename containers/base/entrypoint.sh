#!/bin/bash
# containers/base/entrypoint.sh
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2024-01-01
# Last modified: 2024-01-01
# Description: Container entrypoint. Syncs Python dependencies via uv, then executes
#              the provided command. Falls back to a full sync if the frozen lock fails.

set -euo pipefail

# The venv is pre-installed during the Docker image build.
# No sync needed at runtime — avoids downloading platform-mismatched packages.
exec "$@"
