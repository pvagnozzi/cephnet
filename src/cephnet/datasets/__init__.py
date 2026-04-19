# __init__.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: datasets package — exports dataset adapters and factory

from cephnet.datasets.aariz import AarizDataset
from cephnet.datasets.base import BaseDataset
from cephnet.datasets.factory import get_dataset
from cephnet.datasets.isbi2015 import ISBI2015Dataset

__all__ = [
    "BaseDataset",
    "ISBI2015Dataset",
    "AarizDataset",
    "get_dataset",
]
