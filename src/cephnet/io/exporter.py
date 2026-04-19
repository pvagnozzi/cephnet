# exporter.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-19
# Last modified: 2026-04-19
# Description: ONNX model export utility for cephnet inference artifacts.

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: nn.Module,
    export_dir: Path,
    experiment_name: str,
    image_size: tuple[int, int],
    device: torch.device | None = None,
    opset_version: int = 12,
) -> Path:
    """Export *model* to ONNX format suitable for inference deployment.

    The model is exported with a dynamic batch dimension so the resulting
    artifact can run with any batch size at inference time.

    Args:
        model: Trained PyTorch model (eval mode is set internally).
        export_dir: Host directory where the ``.onnx`` file is written.
        experiment_name: Used as the stem of the output filename.
        image_size: ``(H, W)`` tuple used to construct the dummy input tensor.
        device: Device to run the export on; defaults to CPU.
        opset_version: ONNX opset version (default 17).

    Returns:
        Absolute path to the written ``.onnx`` file.
    """
    if device is None:
        device = torch.device("cpu")

    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    out_path = export_dir / f"{experiment_name}.onnx"

    model.eval()
    model.to(device)

    H, W = image_size
    dummy_input = torch.zeros(1, 3, H, W, device=device)

    dynamic_axes = {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    }

    logger.info("📦 Exporting ONNX model → %s", out_path)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input,),
            str(out_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

    size_mb = out_path.stat().st_size / (1024**2)
    logger.info("✅ ONNX export complete: %s (%.1f MB)", out_path.name, size_mb)
    return out_path
