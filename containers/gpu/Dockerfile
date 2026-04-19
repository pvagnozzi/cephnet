# containers/gpu/Dockerfile
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2024-01-01
# Last modified: 2024-01-01
# Description: GPU-accelerated training image. Builds on the official NVIDIA CUDA
#              runtime image and installs Python 3.11, uv, and CUDA-enabled PyTorch.

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS gpu-base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System dependencies including Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    curl \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /workspace

# Copy project metadata first (for layer caching)
COPY pyproject.toml .
COPY requirements/ requirements/

# Mount points (will be bind-mounted at runtime)
RUN mkdir -p /data /models /reports

# Install CUDA-enabled PyTorch
RUN uv pip install --no-cache \
    torch==2.2.0 \
    torchvision==0.17.0 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Copy source code
COPY src/ /workspace/src/
COPY scripts/python/ /workspace/scripts/python/
COPY configs/ /workspace/configs/

# Install project in the uv-managed environment
RUN uv sync --frozen --no-dev || uv sync --no-dev

# Reuse the same entrypoint as the base image
COPY containers/base/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

LABEL variant="gpu"
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
