"""Core utilities: spacing handling, label mapping, one-hot, reduction, weighting."""

from __future__ import annotations

from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch

from medeval.core.typing import Tensor, as_tensor

# Public type used across the package (imported by medeval.core.__init__)
ReductionType = Literal["none", "mean-case", "mean-class", "global"]


def apply_spacing(
    coords: Tensor,
    spacing: Tuple[float, ...],
    device: Optional[torch.device] = None,
) -> Tensor:
    """Convert voxel coordinates to physical coordinates using spacing.

    Parameters
    ----------
    coords : Tensor
        Voxel coordinates, shape (..., D) where D is spatial dimension
    spacing : Tuple[float, ...]
        Physical spacing per dimension (dx, dy, dz) or (dx, dy) for 2D
    device : torch.device, optional
        Device for output tensor

    Returns
    -------
    Tensor
        Physical coordinates, same shape as coords
    """
    coords = as_tensor(coords, device=device)
    spacing_tensor = as_tensor(list(spacing), dtype=torch.float32, device=coords.device)

    spatial_dims = coords.shape[-1]
    if len(spacing) != spatial_dims:
        raise ValueError(
            f"Spacing dimension {len(spacing)} does not match "
            f"coordinate dimension {spatial_dims}"
        )

    return coords * spacing_tensor


def sample_with_spacing(
    image: Tensor,
    spacing: Tuple[float, ...],
    target_spacing: Optional[Tuple[float, ...]] = None,
    mode: str = "trilinear",
) -> Tuple[Tensor, Tuple[float, ...]]:
    """Resample image to target spacing using interpolation.

    Parameters
    ----------
    image : Tensor
        Input image, shape (B, C, Z, Y, X) or (B, C, Y, X) or (B, Z, Y, X)
    spacing : Tuple[float, ...]
        Current spacing (dz, dy, dx) or (dy, dx)
    target_spacing : Tuple[float, ...], optional
        Target spacing. If None, uses isotropic spacing based on min spacing.
    mode : str
        Interpolation mode: 'nearest', 'bilinear', 'trilinear'

    Returns
    -------
    Tensor
        Resampled image
    Tuple[float, ...]
        Actual target spacing used
    """
    if target_spacing is None:
        target_spacing = tuple([min(spacing)] * len(spacing))

    scale_factors = [s / ts for s, ts in zip(spacing, target_spacing)]
    ndim = len(spacing)

    if image.dim() == 4:
        if ndim == 2:
            scale = (scale_factors[0], scale_factors[1])
        elif ndim == 3:
            scale = (scale_factors[0], scale_factors[1], scale_factors[2])
        else:
            raise ValueError(f"Image dims {image.dim()} incompatible with spacing dims {ndim}")
    elif image.dim() == 5:
        if ndim == 3:
            scale = (scale_factors[0], scale_factors[1], scale_factors[2])
        else:
            raise ValueError(f"Image dims {image.dim()} incompatible with spacing dims {ndim}")
    else:
        raise ValueError(f"Unsupported image dimensions: {image.dim()}")

    if ndim == 2:
        mode_map = {"nearest": "nearest", "bilinear": "bilinear", "trilinear": "bilinear"}
        interp_mode = mode_map.get(mode, "bilinear")
        resampled = torch.nn.functional.interpolate(
            image, scale_factor=scale, mode=interp_mode, align_corners=False
        )
        return resampled, target_spacing

    # 3D
    if image.dim() == 4:
        image = image.unsqueeze(1)
        needs_squeeze = True
    else:
        needs_squeeze = False

    resampled = torch.nn.functional.interpolate(
        image, scale_factor=scale, mode="trilinear", align_corners=False
    )

    if needs_squeeze:
        resampled = resampled.squeeze(1)

    return resampled, target_spacing
def _reduce_mean_case(metrics: Tensor, per_class: bool) -> Tensor:
    """Reduce by averaging over cases (batch dimension).

    Expected input is `(B, ...)` or `(B, C, ...)`.
    - per_class=False: returns a scalar (averages over batch + class + any extra dims)
    - per_class=True: returns per-class values (averages over batch + any extra dims, keeps class dim)
    """
    if metrics.dim() == 0:
        return metrics

    if metrics.dim() == 1:
        # (B,)
        return metrics.mean(dim=0)

    # (B, C, ...)
    if per_class:
        x = metrics.mean(dim=0)  # -> (C, ...)
        if x.dim() > 1:
            x = x.mean(dim=tuple(range(1, x.dim())))  # -> (C,)
        return x

    # scalar
    return metrics.mean()


def _reduce_mean_class(metrics: Tensor, per_class: bool) -> Tensor:
    """Reduce by averaging over classes.

    Note: The project tests expect `reduction="mean-class"` with `per_class=True`
    to return per-class values averaged over cases (i.e., keep class dim).

    For input `(B, C, ...)`:
    - per_class=True: mean over batch + extra dims, keep class -> `(C,)`
    - per_class=False: mean over class then over batch/extra dims -> scalar

    For input `(B,)`:
    - returns mean over batch -> scalar
    """
    if metrics.dim() == 0:
        return metrics

    if metrics.dim() == 1:
        return metrics.mean(dim=0)

    # (B, C, ...)
    if per_class:
        x = metrics.mean(dim=0)  # -> (C, ...)
        if x.dim() > 1:
            x = x.mean(dim=tuple(range(1, x.dim())))  # -> (C,)
        return x

    # scalar: average over classes, then over batch/extra dims
    return metrics.mean(dim=1).mean()


def _reduce_global(metrics: Tensor) -> Tensor:
    """Reduce by averaging over all dimensions (global mean)."""
    if metrics.dim() == 0:
        return metrics
    return metrics.mean()


def reduce_metrics(
    metrics: Tensor,
    reduction: ReductionType = "mean-case",
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    per_class: bool = False,
) -> Tensor:
    """Reduce metrics according to specified reduction strategy.

    Conventions in this repo (aligned to tests):
    - Input is usually `(B, C, ...)` or `(B, ...)`.
    - `reduction="none"` returns the input unchanged.
    - `reduction="mean-case"` averages over cases; if `per_class=True`, keeps class dim.
    - `reduction="mean-class"` averages over classes; if `per_class=True`, returns per-class
      values averaged over cases (i.e., keeps class dim).
    - `reduction="global"` returns a scalar global mean.

    Parameters
    ----------
    metrics : Tensor
        Metric values.
    reduction : ReductionType
        Reduction strategy.
    dim : int or Tuple[int, ...], optional
        Explicit dimensions to reduce over. If provided, overrides `reduction`.
    per_class : bool
        If True, keep/return per-class values where applicable.

    Returns
    -------
    Tensor
        Reduced metrics.
    """
    metrics = as_tensor(metrics)

    if reduction == "none":
        return metrics

    # Explicit dims override the named reduction
    if dim is not None:
        return metrics.mean(dim=dim)

    if reduction == "mean-case":
        return _reduce_mean_case(metrics, per_class=per_class)

    if reduction == "mean-class":
        return _reduce_mean_class(metrics, per_class=per_class)

    if reduction == "global":
        return _reduce_global(metrics)

    raise ValueError(f"Unknown reduction type: {reduction}")


def label_mapping(
    labels: Tensor,
    mapping: dict[int, int],
    ignore_index: Optional[int] = None,
) -> Tensor:
    """Map label values according to a dictionary mapping.

    Parameters
    ----------
    labels : Tensor
        Input labels to map
    mapping : dict[int, int]
        Dictionary mapping old label values to new values
    ignore_index : int, optional
        Label value to ignore (preserved as-is)

    Returns
    -------
    Tensor
        Mapped labels with same shape as input
    """
    labels = as_tensor(labels)
    mapped = labels.clone()

    for old_val, new_val in mapping.items():
        mask = labels == old_val
        if ignore_index is not None:
            mask = mask & (labels != ignore_index)
        mapped[mask] = new_val

    return mapped


def compute_one_hot(labels: Tensor, num_classes: int) -> Tensor:
    """Convert integer labels to one-hot encoding.

    Parameters
    ----------
    labels : Tensor
        Integer labels, shape (...,)
    num_classes : int
        Number of classes

    Returns
    -------
    Tensor
        One-hot encoded labels, shape (..., num_classes)
    """
    labels = as_tensor(labels, dtype=torch.long)
    shape = labels.shape
    one_hot = torch.zeros(*shape, num_classes, dtype=torch.float32, device=labels.device)
    one_hot.scatter_(-1, labels.unsqueeze(-1), 1.0)
    return one_hot


def compute_weights(
    labels: Tensor,
    method: Literal["uniform", "inverse_freq"] = "uniform",
) -> Tensor:
    """Compute sample weights based on label distribution.

    Parameters
    ----------
    labels : Tensor
        Integer labels, shape (...,)
    method : {"uniform", "inverse_freq"}
        Weight computation method:
        - "uniform": All samples have equal weight (1.0)
        - "inverse_freq": Weight inversely proportional to class frequency

    Returns
    -------
    Tensor
        Sample weights, same shape as labels
    """
    labels = as_tensor(labels, dtype=torch.long)

    if method == "uniform":
        return torch.ones_like(labels, dtype=torch.float32)

    if method == "inverse_freq":
        # Flatten to compute frequencies
        labels_flat = labels.flatten()
        unique_labels, counts = torch.unique(labels_flat, return_counts=True)
        total = labels_flat.numel()

        # Compute inverse frequency weights
        freq_weights = total / (len(unique_labels) * counts.float())
        weight_map = torch.zeros(
            labels_flat.max().item() + 1, dtype=torch.float32, device=labels.device
        )
        weight_map[unique_labels] = freq_weights

        # Map weights back to original shape
        weights = weight_map[labels]
        return weights

    raise ValueError(f"Unknown weight method: {method}")
