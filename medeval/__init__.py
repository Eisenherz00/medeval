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

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _dist_version
from pathlib import Path
import re


def _read_version_from_pyproject(pyproject_path: Path) -> str | None:
    """Best-effort read of `[project].version` from `pyproject.toml` without extra deps."""
    try:
        text = pyproject_path.read_text(encoding="utf-8")
    except OSError:
        return None

    in_project = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_project = line == "[project]"
            continue
        if not in_project:
            continue

        m = re.match(r'version\s*=\s*"([^"]+)"\s*$', line)
        if m:
            return m.group(1)
    return None


def _is_site_packages_path(p: Path) -> bool:
    s = str(p)
    return ("site-packages" in s) or ("dist-packages" in s)


def _resolve_version() -> str:
    """Resolve version for both installed-package and source-checkout workflows."""
    module_path = Path(__file__).resolve()
    repo_root = module_path.parents[1]
    pyproject = repo_root / "pyproject.toml"

    # If running from a source checkout (not in site-packages), prefer the repo version.
    if pyproject.exists() and not _is_site_packages_path(module_path):
        v = _read_version_from_pyproject(pyproject)
        if v:
            return v

    # Otherwise, fall back to installed distribution metadata.
    try:
        return _dist_version("medeval")
    except PackageNotFoundError:
        # Last resort: if we're in a repo but parsing failed, still avoid crashing imports.
        v = _read_version_from_pyproject(pyproject) if pyproject.exists() else None
        return v or "0.0.0"


__version__ = _resolve_version()

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
