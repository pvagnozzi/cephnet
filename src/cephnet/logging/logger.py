# logger.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: Centralized logging setup using Rich for colored, emoji-enhanced output

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

_console = Console(stderr=True)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger configured by setup_logging."""
    return logging.getLogger(name)


def setup_logging(
    level: str = "INFO",
    log_dir: Path | None = None,
    rich_traceback: bool = True,
) -> None:
    """Configure root logger with Rich console handler and optional file handler."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [
        RichHandler(
            console=_console,
            show_time=True,
            show_path=False,
            rich_tracebacks=rich_traceback,
            markup=True,
            log_time_format="[%H:%M:%S]",
        )
    ]

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "training.log", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s: %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        format="%(message)s",
        datefmt="[%H:%M:%S]",
    )

    # Suppress noisy third-party loggers
    for noisy in ("PIL", "matplotlib", "albumentations", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
