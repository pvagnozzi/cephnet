# paths.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: Path resolution and directory management utilities

from pathlib import Path


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    """Resolve a path, optionally relative to a base directory."""
    p = Path(path)
    if base is not None and not p.is_absolute():
        p = base / p
    return p.resolve()


def ensure_dirs(*paths: str | Path) -> None:
    """Create directories (and parents) if they do not exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
