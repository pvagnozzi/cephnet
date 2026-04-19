# preparer.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-19
# Last modified: 2026-04-19
# Description: Dataset download and preparation utilities for ISBI2015 and Aariz datasets.

from __future__ import annotations

import hashlib
import io
import logging
import shutil
import struct
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public download sources (checked; fallback to synthetic when unavailable)
# ---------------------------------------------------------------------------
_ISBI2015_SOURCES: list[dict[str, Any]] = [
    {
        "url": "https://figshare.com/ndownloader/articles/5045806/versions/1",
        "kind": "zip",
        "description": "ISBI 2015 Cephalometric Landmark Detection (Figshare)",
    },
]

_AARIZ_SOURCES: list[dict[str, Any]] = [
    {
        "url": "https://zenodo.org/record/8109511/files/aariz_dataset.zip",
        "kind": "zip",
        "description": "Aariz cephalometric dataset (Zenodo)",
    },
]

# Standard ISBI 2015 landmark names (19 landmarks)
ISBI2015_LANDMARK_NAMES: list[str] = [
    "sella", "nasion", "orbitale", "porion", "subspinale",
    "supramentale", "pogonion", "menton", "gnathion", "gonion",
    "incision_inferius", "incision_superius", "upper_lip", "lower_lip",
    "subnasale", "soft_tissue_pogonion", "posterior_nasal_spine",
    "anterior_nasal_spine", "articulare",
]

# Standard Aariz landmark names (19 landmarks, same canonical set)
AARIZ_LANDMARK_NAMES: list[str] = ISBI2015_LANDMARK_NAMES


def _try_download_zip(url: str, dest_dir: Path) -> Path | None:
    """Attempt to download a ZIP archive from *url* into *dest_dir*.

    Returns the path to the downloaded ZIP, or None on any failure.
    """
    try:
        import urllib.request

        dest_dir.mkdir(parents=True, exist_ok=True)
        zip_path = dest_dir / "_download.zip"
        logger.info("⬇️  Downloading from %s …", url)
        urllib.request.urlretrieve(url, zip_path)  # noqa: S310 (trusted academic sources)
        logger.info("✅ Download complete: %s (%.1f MB)", zip_path.name, zip_path.stat().st_size / 1e6)
        return zip_path
    except Exception as exc:
        logger.warning("⚠️  Download failed: %s", exc)
        return None


def _extract_zip(zip_path: Path, dest_dir: Path) -> bool:
    """Extract *zip_path* into *dest_dir*. Returns True on success."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        zip_path.unlink(missing_ok=True)
        return True
    except Exception as exc:
        logger.warning("⚠️  ZIP extraction failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Minimal PNG writer (no external deps at prepare time)
# ---------------------------------------------------------------------------

def _write_png(path: Path, array: np.ndarray) -> None:
    """Write an 8-bit grayscale or RGB PNG to *path* without PIL/cv2."""
    import zlib

    assert array.dtype == np.uint8
    h, w = array.shape[:2]
    is_gray = array.ndim == 2
    color_type = 0 if is_gray else 2  # grayscale or RGB
    channels = 1 if is_gray else 3

    def chunk(name: bytes, data: bytes) -> bytes:
        c = name + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    raw_rows = b""
    for row in range(h):
        if is_gray:
            raw_rows += b"\x00" + array[row].tobytes()
        else:
            raw_rows += b"\x00" + array[row].tobytes()

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw_rows, 6)))
        f.write(chunk(b"IEND", b""))


def _generate_synthetic_radiograph(
    rng: np.random.Generator,
    width: int = 512,
    height: int = 512,
) -> np.ndarray:
    """Generate a pseudo-cephalometric radiograph as a grayscale uint8 array.

    Uses gradient fields and elliptical blobs to approximate the characteristic
    appearance of a lateral skull X-ray — good enough for model sanity checks
    while remaining distinct from random noise.
    """
    img = np.zeros((height, width), dtype=np.float32)

    # Base gradient (brighter centre, like an exposed film)
    cx, cy = width * 0.55, height * 0.5
    yy, xx = np.mgrid[0:height, 0:width]
    dist = np.sqrt(((xx - cx) / (width * 0.45)) ** 2 + ((yy - cy) / (height * 0.5)) ** 2)
    img += np.clip(1.0 - dist * 0.7, 0, 1) * 160 + 40

    # Skull outline (large bright ellipse)
    skull_rx, skull_ry = width * 0.38, height * 0.46
    skull_cx, skull_cy = width * 0.52, height * 0.50
    skull_mask = ((xx - skull_cx) / skull_rx) ** 2 + ((yy - skull_cy) / skull_ry) ** 2
    img[skull_mask < 1.0] += 50

    # Mandible arc
    for angle in np.linspace(-0.6, 0.9, 80):
        bx = int(skull_cx + skull_rx * 0.7 * np.cos(angle + np.pi * 0.8))
        by = int(skull_cy + skull_ry * 0.6 * np.sin(angle + np.pi * 0.8) + height * 0.12)
        if 0 <= bx < width and 0 <= by < height:
            img[max(0, by - 4) : by + 4, max(0, bx - 4) : bx + 4] += 80

    # Teeth region
    teeth_y = int(skull_cy + skull_ry * 0.28)
    teeth_x_start = int(skull_cx - skull_rx * 0.25)
    teeth_x_end = int(skull_cx + skull_rx * 0.05)
    img[teeth_y : teeth_y + 30, teeth_x_start:teeth_x_end] += rng.uniform(
        60, 100, size=(30, teeth_x_end - teeth_x_start)
    ).astype(np.float32)

    # Cervical spine region (lower right)
    spine_cx = int(skull_cx + skull_rx * 0.72)
    spine_cy_start = int(skull_cy + skull_ry * 0.30)
    for i in range(5):
        sy = spine_cy_start + i * 28
        if sy + 20 < height:
            img[sy : sy + 18, spine_cx : spine_cx + 30] += 90

    # Random noise
    img += rng.normal(0, 8, size=(height, width)).astype(np.float32)

    return np.clip(img, 0, 255).astype(np.uint8)


def _canonical_landmark_positions(
    rng: np.random.Generator,
    width: int = 512,
    height: int = 512,
) -> np.ndarray:
    """Return 19 realistic landmark positions for a synthetic lateral ceph."""
    cx, cy = width * 0.52, height * 0.50
    rx, ry = width * 0.38, height * 0.46

    # Approximate anatomical positions (fraction of skull bounding box)
    # (x_frac, y_frac) relative to skull centre
    relative: list[tuple[float, float]] = [
        (-0.10, -0.20),  # sella
        (-0.35, -0.38),  # nasion
        (-0.55, -0.10),  # orbitale
        (0.65, -0.05),   # porion
        (-0.48, 0.08),   # subspinale
        (-0.42, 0.22),   # supramentale
        (-0.52, 0.36),   # pogonion
        (-0.18, 0.62),   # menton
        (-0.38, 0.58),   # gnathion
        (0.45, 0.45),    # gonion
        (-0.38, 0.18),   # incision_inferius
        (-0.36, 0.12),   # incision_superius
        (-0.44, 0.00),   # upper_lip
        (-0.44, 0.10),   # lower_lip
        (-0.48, 0.00),   # subnasale
        (-0.55, 0.38),   # soft_tissue_pogonion
        (0.22, -0.05),   # posterior_nasal_spine
        (-0.38, -0.05),  # anterior_nasal_spine
        (0.32, -0.15),   # articulare
    ]

    pts = np.array([
        [cx + dx * rx + rng.uniform(-8, 8), cy + dy * ry + rng.uniform(-8, 8)]
        for dx, dy in relative
    ], dtype=np.float32)

    # Clamp to valid range
    pts[:, 0] = np.clip(pts[:, 0], 20, width - 20)
    pts[:, 1] = np.clip(pts[:, 1], 20, height - 20)
    return pts


def generate_synthetic_dataset(
    root: Path,
    n_train: int = 80,
    n_val: int = 20,
    n_test: int = 20,
    image_size: tuple[int, int] = (512, 512),
    seed: int = 42,
    force: bool = False,
) -> None:
    """Generate a synthetic dataset with realistic-looking cephalometric images.

    Creates the same directory layout expected by :class:`ISBI2015Dataset`:
    - ``<root>/images/<split>/<id>.png``
    - ``<root>/annotations/landmarks.csv``

    Args:
        root: Dataset root directory (e.g. ``/data/processed/isbi2015``).
        n_train: Number of training images to generate.
        n_val: Number of validation images.
        n_test: Number of test images.
        image_size: ``(W, H)`` of generated images.
        seed: RNG seed for reproducibility.
        force: If True, overwrite existing data.
    """
    ann_path = root / "annotations" / "landmarks.csv"
    if ann_path.exists() and not force:
        logger.info("✅ Dataset already exists at %s — skipping generation", root)
        return

    logger.info("🖼️  Generating synthetic cephalometric dataset → %s", root)
    W, H = image_size
    rng = np.random.default_rng(seed)
    splits = {"train": n_train, "val": n_val, "test": n_test}

    rows: list[dict] = []
    for split, n in splits.items():
        img_dir = root / "images" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img_array = _generate_synthetic_radiograph(rng, W, H)
            img_id = f"{split}_{i:04d}"
            img_path = img_dir / f"{img_id}.png"
            _write_png(img_path, img_array)

            pts = _canonical_landmark_positions(rng, W, H)
            row: dict = {"image_id": img_id, "split": split}
            for j, (px, py) in enumerate(pts):
                row[f"lm_{j}_x"] = round(float(px), 2)
                row[f"lm_{j}_y"] = round(float(py), 2)
            rows.append(row)

        logger.info("   ✅ %s split: %d images written", split, n)

    ann_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(ann_path, index=False)
    logger.info("📋 Annotations saved → %s (%d rows)", ann_path, len(rows))
    logger.info("✅ Synthetic dataset ready at %s", root)


class DatasetPreparer:
    """Orchestrates download (or synthetic generation) of a cephnet dataset."""

    def __init__(self, dataset_name: str, root: Path, raw_root: Path | None = None) -> None:
        self.dataset_name = dataset_name.lower()
        self.root = Path(root)
        self.raw_root = Path(raw_root) if raw_root else self.root.parent.parent / "raw" / dataset_name

    def prepare(
        self,
        image_size: tuple[int, int] = (512, 512),
        force: bool = False,
        n_train: int = 80,
        n_val: int = 20,
        n_test: int = 20,
    ) -> None:
        """Prepare the dataset: download if possible, else generate synthetic data."""
        sources = _ISBI2015_SOURCES if self.dataset_name == "isbi2015" else _AARIZ_SOURCES

        # 1. Check if data already exists
        ann_path = self.root / "annotations" / "landmarks.csv"
        if ann_path.exists() and not force:
            n_rows = len(pd.read_csv(ann_path))
            logger.info("✅ Dataset '%s' already prepared (%d samples) at %s", self.dataset_name, n_rows, self.root)
            return

        # 2. Try downloading from each source
        downloaded = False
        for source in sources:
            logger.info("🔎 Trying source: %s", source["description"])
            zip_path = _try_download_zip(source["url"], self.raw_root)
            if zip_path is None:
                continue
            if _extract_zip(zip_path, self.raw_root):
                # Attempt to locate and reformat into expected layout
                if self._reformat_downloaded(self.raw_root, image_size):
                    downloaded = True
                    break
            logger.warning("⚠️  Could not use source '%s', trying next…", source["description"])

        # 3. Fall back to synthetic generation
        if not downloaded:
            logger.warning(
                "⚠️  No downloadable source succeeded for '%s'. "
                "Generating synthetic data in %s for development/testing.",
                self.dataset_name,
                self.root,
            )
            logger.warning(
                "📌 To use the real dataset, download it manually and place images/annotations "
                "under: %s/images/ and %s/annotations/landmarks.csv",
                self.root,
                self.root,
            )
            generate_synthetic_dataset(
                self.root,
                n_train=n_train,
                n_val=n_val,
                n_test=n_test,
                image_size=image_size,
                force=force,
            )

    def _reformat_downloaded(self, src_dir: Path, image_size: tuple[int, int]) -> bool:
        """Reformat a downloaded archive into the canonical layout.

        Subclasses or future implementations can override this method.
        Returns True if reformatting succeeded.
        """
        try:
            # Walk and look for PNG/BMP/TIFF images + annotation files
            images = list(src_dir.rglob("*.png")) + list(src_dir.rglob("*.bmp")) + list(src_dir.rglob("*.tiff"))
            txt_files = list(src_dir.rglob("*.txt"))
            if not images:
                logger.warning("⚠️  No images found in downloaded archive — cannot reformat")
                return False

            logger.info("🔧 Reformatting %d downloaded images into canonical layout…", len(images))
            out_img_dir = self.root / "images" / "train"
            out_img_dir.mkdir(parents=True, exist_ok=True)

            # Copy images
            for i, img in enumerate(images):
                shutil.copy2(img, out_img_dir / f"train_{i:04d}{img.suffix}")

            # Parse landmark text files if they look like coordinate lists
            rows: list[dict] = []
            for i, img in enumerate(images):
                stem = img.stem
                txt_match = next((t for t in txt_files if t.stem == stem), None)
                row: dict = {"image_id": f"train_{i:04d}", "split": "train"}
                if txt_match:
                    coords = txt_match.read_text().strip().split()
                    pts = [float(c) for c in coords]
                    n_lm = len(pts) // 2
                    for j in range(n_lm):
                        row[f"lm_{j}_x"] = pts[j * 2]
                        row[f"lm_{j}_y"] = pts[j * 2 + 1]
                rows.append(row)

            ann_dir = self.root / "annotations"
            ann_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(ann_dir / "landmarks.csv", index=False)
            logger.info("✅ Reformatted dataset written to %s", self.root)
            return True
        except Exception as exc:
            logger.error("❌ Reformat failed: %s", exc)
            return False
