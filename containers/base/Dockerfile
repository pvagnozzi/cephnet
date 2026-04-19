# containers/base/Dockerfile
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2024-01-01
# Last modified: 2024-01-01
# Description: Base image for cephnet training subsystem. Installs uv, Python 3.11,
#              and common system dependencies. Derived images (cpu, gpu) install PyTorch on top.

FROM python:3.11-slim AS base

# Metadata
LABEL maintainer="Piergiorgio Vagnozzi"
LABEL project="cephnet"
LABEL description="Base image for cephalometric landmark detection training"

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Working directory
WORKDIR /workspace

# Copy project metadata first (for layer caching)
COPY pyproject.toml .
COPY uv.lock .
COPY requirements/ requirements/

# Mount points (will be bind-mounted at runtime)
RUN mkdir -p /data /models /reports

# Entrypoint
COPY containers/base/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
