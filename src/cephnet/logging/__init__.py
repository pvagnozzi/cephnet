# __init__.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: logging package — exports get_logger and setup_logging

from cephnet.logging.logger import get_logger, setup_logging

__all__ = ["get_logger", "setup_logging"]
