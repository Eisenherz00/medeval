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


def normalize_input_shapes(
    pred: Tensor,
    target: Tensor,
    spacing: Optional[Tuple[float, ...]] = None,
    require_spacing: bool = False,
) -> Tuple[Tensor, Tensor, int, Optional[Tuple[float, ...]]]:
    """Normalize input shapes to standard (B, C, ...) format.

    This function handles various input conventions:
    - 2D: (H, W) -> (1, 1, H, W)
    - 2D batched: (B, H, W) -> (B, 1, H, W)
    - 2D with channel: (B, C, H, W) -> unchanged
    - 3D: (Z, Y, X) -> (1, 1, Z, Y, X)
    - 3D batched: (B, Z, Y, X) -> (B, 1, Z, Y, X)
    - 3D with channel: (B, C, Z, Y, X) -> unchanged

    Heuristic for distinguishing (B, Z, Y, X) from (B, C, Y, X):
    - If axis-1 size is in {1, 2, 3, 4}, treat as channel dim (2D spatial)
    - Otherwise treat as spatial dim (3D spatial, needs channel added)

    Output Contract
    ---------------
    - pred is torch.Tensor with dtype=float32 (for metric computation)
    - target keeps its original dtype (may be int for label maps, float for soft targets)
    - Both outputs are on the same device as pred
    - Both outputs have the same shape: (B, C, *spatial_dims)

    Spacing Convention
    ------------------
    This library uses the following spacing axis order:
    - 2D: (dy, dx) where dy is row spacing, dx is column spacing
    - 3D: (dz, dy, dx) where dz is slice spacing

    Parameters
    ----------
    pred : Tensor
        Prediction tensor
    target : Tensor
        Target tensor
    spacing : Tuple[float, ...], optional
        Physical spacing. If provided, validates against spatial dims.
        For 2D: (dy, dx), for 3D: (dz, dy, dx)
    require_spacing : bool
        If True and spacing is None, raises ValueError (for surface metrics)

    Returns
    -------
    pred : Tensor
        Normalized prediction, shape (B, C, ...), dtype float32
    target : Tensor
        Normalized target, shape (B, C, ...), dtype float32, same device as pred
    spatial_dims : int
        Number of spatial dimensions (2 or 3)
    spacing : Tuple[float, ...] or None
        Validated spacing (or None if not provided)

    Raises
    ------
    ValueError
        If shapes are incompatible or spacing is required but not provided
    """
    pred = as_tensor(pred)
    target = as_tensor(target)

    # Ensure float32 dtype for pred (for metric computation)
    # Keep target's original dtype (may be int for label maps, float for soft targets)
    pred = pred.float()

    # Ensure same device
    if target.device != pred.device:
        target = target.to(pred.device)

    def _normalize_single(x: Tensor) -> Tuple[Tensor, int]:
        """Normalize a single tensor and return spatial dims."""
        d = x.dim()

        if d == 2:
            # (H, W) -> (1, 1, H, W)
            return x.unsqueeze(0).unsqueeze(0), 2

        if d == 3:
            # Could be (Z, Y, X) or (B, H, W)
            # Heuristic: if first dim is small (<= 4), treat as batch
            if x.shape[0] <= 4:
                # (B, H, W) -> (B, 1, H, W)
                return x.unsqueeze(1), 2
            else:
                # (Z, Y, X) -> (1, 1, Z, Y, X)
                return x.unsqueeze(0).unsqueeze(0), 3

        if d == 4:
            # Could be (B, C, H, W) or (B, Z, Y, X)
            # Heuristic: if axis-1 is in {1,2,3,4}, treat as channel (2D)
            if x.shape[1] in (1, 2, 3, 4):
                # (B, C, H, W) -> unchanged, 2D
                return x, 2
            else:
                # (B, Z, Y, X) -> (B, 1, Z, Y, X), 3D
                return x.unsqueeze(1), 3

        if d == 5:
            # (B, C, Z, Y, X) -> unchanged, 3D
            return x, 3

        # Fallback for higher dims: assume (B, C, ...)
        return x, max(d - 2, 2)

    pred, pred_spatial = _normalize_single(pred)
    target, target_spatial = _normalize_single(target)

    # Use the larger spatial dim if they differ (conservative)
    spatial_dims = max(pred_spatial, target_spatial)

    # Ensure same number of dimensions
    while pred.dim() < target.dim():
        pred = pred.unsqueeze(0)
    while target.dim() < pred.dim():
        target = target.unsqueeze(0)

    # Validate spacing if provided
    if spacing is not None:
        if len(spacing) != spatial_dims:
            raise ValueError(
                f"Spacing dimension {len(spacing)} does not match "
                f"inferred spatial dimensions {spatial_dims}. "
                f"Expected {spatial_dims}D spacing (dz,dy,dx for 3D or dy,dx for 2D)."
            )

    if require_spacing and spacing is None:
        raise ValueError(
            "Spacing is required for this metric (e.g., surface distances). "
            "Provide spacing as (dy, dx) for 2D or (dz, dy, dx) for 3D."
        )

    return pred, target, spatial_dims, spacing


def get_spatial_dims_from_spacing(spacing: Optional[Tuple[float, ...]]) -> Optional[int]:
    """Get number of spatial dimensions from spacing tuple.

    Parameters
    ----------
    spacing : Tuple[float, ...], optional
        Physical spacing

    Returns
    -------
    int or None
        Number of spatial dimensions (2 or 3), or None if spacing not provided
    """
    if spacing is None:
        return None
    return len(spacing)
