"""MedEval: Medical Imaging Evaluation Metrics Library.

A PyTorch-native library for computing, aggregating, and visualizing
evaluation metrics for medical imaging tasks with 2D/3D support,
correct physical spacing handling, confidence intervals, and clean APIs.

Example
-------
>>> import torch
>>> from medeval.metrics.segmentation import dice_score
>>> pred = torch.rand(1, 1, 64, 64, 64) > 0.5
>>> target = torch.rand(1, 1, 64, 64, 64) > 0.5
>>> dice = dice_score(pred, target)
"""

from importlib.metadata import version as _version
__version__ = _version("medeval")

# Convenience imports
from medeval.core import (
    as_tensor,
    load_nifti,
    save_nifti,
    aggregate_metrics,
    bootstrap_ci,
    MedicalPrediction,
)

__all__ = [
    "__version__",
    "as_tensor",
    "load_nifti",
    "save_nifti",
    "aggregate_metrics",
    "bootstrap_ci",
    "MedicalPrediction",
]
