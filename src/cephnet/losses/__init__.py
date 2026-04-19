# __init__.py
# MIT License
# Author: Piergiorgio Vagnozzi
# Created: 2026-04-18
# Last modified: 2026-04-18
# Description: losses package — exports all loss functions

from cephnet.losses.heatmap_loss import AdaptiveWingLoss, HeatmapLoss, generate_gaussian_heatmaps
from cephnet.losses.regression_loss import RegressionLoss, WingLoss

__all__ = [
    "HeatmapLoss",
    "AdaptiveWingLoss",
    "RegressionLoss",
    "WingLoss",
    "generate_gaussian_heatmaps",
]
