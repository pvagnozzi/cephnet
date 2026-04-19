# test_utils.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: Tests for seed_everything, resolve_path and ensure_dirs

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cephnet.utils.paths import ensure_dirs, resolve_path
from cephnet.utils.seeding import seed_everything


def test_seed_everything_reproducible() -> None:
    """The same seed produces identical random tensors across two calls."""
    seed_everything(42)
    a = torch.rand(10)

    seed_everything(42)
    b = torch.rand(10)

    assert torch.allclose(a, b)


def test_seed_everything_different_seeds() -> None:
    """Different seeds produce different random tensors."""
    seed_everything(42)
    a = torch.rand(10)

    seed_everything(999)
    b = torch.rand(10)

    assert not torch.allclose(a, b)


def test_resolve_path_absolute(tmp_path: Path) -> None:
    """An absolute path is returned unchanged (after .resolve())."""
    p = tmp_path / "some" / "file.txt"
    resolved = resolve_path(p)
    assert resolved == p.resolve()
    assert resolved.is_absolute()


def test_resolve_path_relative(tmp_path: Path) -> None:
    """A relative path joined with base resolves correctly."""
    resolved = resolve_path("subdir/file.txt", base=tmp_path)
    expected = (tmp_path / "subdir" / "file.txt").resolve()
    assert resolved == expected


def test_resolve_path_no_base() -> None:
    """A relative path without base resolves against the current directory."""
    resolved = resolve_path("some_relative_file.txt")
    assert resolved.is_absolute()
    assert resolved.name == "some_relative_file.txt"


def test_ensure_dirs(tmp_path: Path) -> None:
    """ensure_dirs creates all nested directories that do not yet exist."""
    nested = tmp_path / "a" / "b" / "c"
    assert not nested.exists()

    ensure_dirs(nested)

    assert nested.exists()
    assert nested.is_dir()


def test_ensure_dirs_multiple(tmp_path: Path) -> None:
    """ensure_dirs accepts multiple paths and creates all of them."""
    d1 = tmp_path / "dir1"
    d2 = tmp_path / "deep" / "dir2"

    ensure_dirs(d1, d2)

    assert d1.is_dir()
    assert d2.is_dir()
