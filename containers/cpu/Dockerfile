# containers/cpu/Dockerfile
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2024-01-01
# Last modified: 2026-04-18
# Description: CPU-only training image. Derives from cephnet-base and installs
#              CPU-only PyTorch to keep the image size minimal.

FROM cephnet-base:latest AS cpu

# README.md is required by hatchling for the project build
COPY README.md .

# Copy source code and entry-point scripts
COPY src/ /workspace/src/
COPY scripts/python/ /workspace/scripts/python/
COPY configs/ /workspace/configs/

# Create venv, install CPU torch + all deps, then install the project itself
COPY requirements/docker-cpu.txt requirements/docker-cpu.txt
RUN uv venv && \
    uv pip install --no-cache \
        "torch>=2.2.0" "torchvision>=0.17.0" \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match && \
    uv pip install --no-cache -r requirements/base.txt && \
    uv pip install --no-cache "onnx>=1.15.0" "onnxscript>=0.1.0" && \
    uv pip install --no-cache -e . --no-deps

# Expose venv on PATH so `python` resolves without uv run
ENV PATH="/workspace/.venv/bin:$PATH"

LABEL variant="cpu"
