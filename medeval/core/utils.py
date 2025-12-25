"""Core utilities: spacing handling, label mapping, one-hot, reduction, weighting."""

from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np
import torch

from medeval.core.typing import ArrayLike, Device, DType, Tensor, as_tensor

ReductionType = Literal["none", "mean-case", "mean-class", "global"]


def apply_spacing(
    coords: Tensor,
    spacing: Tuple[float, ...],
    device: Optional[Device] = None,
) -> Tensor:
    """
    Convert voxel coordinates to physical coordinates using spacing.

    Parameters
    ----------
    coords : Tensor
        Voxel coordinates, shape (..., D) where D is spatial dimension
    spacing : Tuple[float, ...]
        Physical spacing per dimension (dx, dy, dz) or (dx, dy) for 2D
    device : Device, optional
        Device for output tensor

    Returns
    -------
    Tensor
        Physical coordinates, same shape as coords
    """
    coords = as_tensor(coords, device=device)
    spacing_tensor = as_tensor(spacing, dtype=torch.float32, device=coords.device)

    # Ensure spacing matches spatial dimension
    spatial_dims = coords.shape[-1]
    if len(spacing) != spatial_dims:
        raise ValueError(
            f"Spacing dimension {len(spacing)} does not match "
            f"coordinate dimension {spatial_dims} for coords shape {coords.shape}. "
            f"Expected spacing with {spatial_dims} values, got {len(spacing)}."
        )

    # Apply spacing: physical = voxel * spacing
    return coords * spacing_tensor


def _get_spatial_dims_and_scale(
    image: Tensor,
    spacing: Tuple[float, ...],
    target_spacing: Tuple[float, ...],
) -> Tuple[int, Tuple[float, ...], bool]:
    """
    Determine spatial dimensions, scale factors, and if channel dim needs squeeze.

    Parameters
    ----------
    image : Tensor
        Input image tensor
    spacing : Tuple[float, ...]
        Current spacing
    target_spacing : Tuple[float, ...]
        Target spacing

    Returns
    -------
    Tuple[int, Tuple[float, ...], bool]
        (spatial_dims, scale_factors, needs_squeeze)
    """
    ndim = len(spacing)
    image_dim = image.dim()
    scale_factors = [s / ts for s, ts in zip(spacing, target_spacing)]

    if image_dim == 4:  # (B, C, Y, X) or (B, Z, Y, X)
        if ndim == 2:
            # 2D: (B, C, Y, X)
            return 2, tuple(scale_factors[:2]), False
        elif ndim == 3:
            # 3D without channel: (B, Z, Y, X)
            return 3, tuple(scale_factors), True
        else:
            raise ValueError(
                f"Image dims {image_dim} incompatible with spacing dims {ndim}"
            )
    elif image_dim == 5:  # (B, C, Z, Y, X)
        if ndim == 3:
            return 3, tuple(scale_factors), False
        else:
            raise ValueError(
                f"Image dims {image_dim} incompatible with spacing dims {ndim}"
            )
    else:
        raise ValueError(f"Unsupported image dimensions: {image_dim}")


def sample_with_spacing(
    image: Tensor,
    spacing: Tuple[float, ...],
    target_spacing: Optional[Tuple[float, ...]] = None,
    mode: str = "trilinear",
) -> Tuple[Tensor, Tuple[float, ...]]:
    """
    Resample image to target spacing using interpolation.

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
        # Use isotropic spacing based on minimum
        target_spacing = tuple([min(spacing)] * len(spacing))

    # Get spatial dimensions and scale factors
    spatial_dims, scale_factors, needs_squeeze = _get_spatial_dims_and_scale(
        image, spacing, target_spacing
    )

    # Prepare image for interpolation
    work_image = image
    if spatial_dims == 3 and needs_squeeze:
        # (B, Z, Y, X) - add channel dimension for 3D interpolation
        work_image = image.unsqueeze(1)  # (B, 1, Z, Y, X)

    # Resample using torch's interpolate
    if spatial_dims == 2:
        # 2D interpolation: (B, C, Y, X)
        mode_map = {"nearest": "nearest", "bilinear": "bilinear", "trilinear": "bilinear"}
        interp_mode = mode_map.get(mode, "bilinear")
        resampled = torch.nn.functional.interpolate(
            work_image, scale_factor=scale_factors, mode=interp_mode, align_corners=False
        )
    else:  # 3D: (B, C, Z, Y, X) or (B, 1, Z, Y, X)
        # Compute target size
        target_size = tuple(
            int(work_image.shape[i + 2] * scale_factors[i]) for i in range(3)
        )

        # Use trilinear interpolation for 3D
        mode_map = {"nearest": "nearest", "bilinear": "trilinear", "trilinear": "trilinear"}
        interp_mode = mode_map.get(mode, "trilinear")
        resampled = torch.nn.functional.interpolate(
            work_image, size=target_size, mode=interp_mode, align_corners=False
        )

        # Remove channel dimension if we added it
        if needs_squeeze:
            resampled = resampled.squeeze(1)  # (B, Z, Y, X)

    return resampled, target_spacing


def label_mapping(
    labels: Tensor,
    mapping: Dict[int, int],
    ignore_index: Optional[int] = None,
) -> Tensor:
    """
    Map label values according to a dictionary.

    Parameters
    ----------
    labels : Tensor
        Input labels, integer tensor
    mapping : Dict[int, int]
        Mapping from old label values to new label values
    ignore_index : int, optional
        Label value to ignore (not mapped)

    Returns
    -------
    Tensor
        Mapped labels
    """
    labels = labels.clone()
    for old_val, new_val in mapping.items():
        if ignore_index is None or old_val != ignore_index:
            labels[labels == old_val] = new_val
    return labels


def compute_one_hot(
    labels: Tensor,
    num_classes: Optional[int] = None,
    ignore_index: Optional[int] = None,
) -> Tensor:
    """
    Convert integer labels to one-hot encoding.

    Parameters
    ----------
    labels : Tensor
        Integer labels, shape (..., H, W, ...) or (..., D, H, W)
    num_classes : int, optional
        Number of classes. If None, inferred from labels.
    ignore_index : int, optional
        Label value to ignore (set to all zeros in one-hot)

    Returns
    -------
    Tensor
        One-hot encoding, shape (..., num_classes, H, W, ...)
    """
    if num_classes is None:
        if ignore_index is not None:
            unique_labels = labels[labels != ignore_index].unique()
        else:
            unique_labels = labels.unique()
        num_classes = int(unique_labels.max().item()) + 1

    # Create one-hot tensor
    shape = list(labels.shape)
    one_hot = torch.zeros(*shape, num_classes, dtype=torch.float32, device=labels.device)

    # Use scatter_ for efficient one-hot encoding
    labels_long = labels.long()
    
    if ignore_index is not None:
        # Mask out ignore_index values
        valid_mask = labels_long != ignore_index
        labels_valid = labels_long * valid_mask.long()  # Set ignored to 0
        # Create indices for scatter: expand labels to match one_hot shape
        labels_expanded = labels_valid.unsqueeze(-1)
        # Scatter ones at the appropriate positions
        one_hot.scatter_(-1, labels_expanded, 1.0)
        # Zero out ignored positions
        ignore_mask = (~valid_mask).unsqueeze(-1).expand_as(one_hot)
        one_hot[ignore_mask] = 0.0
    else:
        # Simple case: use scatter directly
        labels_expanded = labels_long.unsqueeze(-1)
        one_hot.scatter_(-1, labels_expanded, 1.0)

    return one_hot


def _reduce_mean_case(
    metrics: Tensor, batch_dim: int, class_dim: Optional[int], per_class: bool
) -> Tensor:
    """Reduce by averaging over cases (batch dimension)."""
    if per_class and class_dim is not None:
        # Average over cases, keep classes
        return metrics.mean(dim=batch_dim)
    else:
        # Average over cases (and classes if present)
        if class_dim is not None:
            return metrics.mean(dim=(batch_dim, class_dim))
        else:
            return metrics.mean(dim=batch_dim)


def _reduce_mean_class(metrics: Tensor, class_dim: Optional[int]) -> Tensor:
    """Reduce by averaging over classes."""
    if class_dim is not None:
        return metrics.mean(dim=class_dim)
    else:
        return metrics  # No class dimension to reduce


def _reduce_global(metrics: Tensor, batch_dim: int, class_dim: Optional[int]) -> Tensor:
    """Reduce by averaging over both cases and classes."""
    if class_dim is not None:
        return metrics.mean(dim=(batch_dim, class_dim))
    else:
        return metrics.mean(dim=batch_dim)


def reduce_metrics(
    metrics: Tensor,
    reduction: ReductionType = "mean-case",
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    per_class: bool = False,
) -> Tensor:
    """
    Reduce metrics according to specified reduction strategy.

    Parameters
    ----------
    metrics : Tensor
        Metric values, shape (B, C, ...) where B is batch, C is classes
    reduction : ReductionType
        Reduction strategy:
        - "none": no reduction
        - "mean-case": average over cases (batch dimension)
        - "mean-class": average over classes
        - "global": average over both cases and classes
    dim : int or Tuple[int, ...], optional
        Dimensions to reduce over. If None, inferred from reduction type.
    per_class : bool
        If True and reduction is "mean-case", returns per-class metrics

    Returns
    -------
    Tensor
        Reduced metrics
    """
    if reduction == "none":
        return metrics

    # Use provided dim if specified
    if dim is not None:
        return metrics.mean(dim=dim)

    # Infer dimensions based on shape
    if metrics.dim() >= 2:
        # Assume (B, C, ...) structure
        batch_dim = 0
        class_dim = 1
    else:
        batch_dim = 0
        class_dim = None

    # Dispatch based on reduction type
    reduction_map = {
        "mean-case": lambda: _reduce_mean_case(metrics, batch_dim, class_dim, per_class),
        "mean-class": lambda: _reduce_mean_class(metrics, class_dim),
        "global": lambda: _reduce_global(metrics, batch_dim, class_dim),
    }

    if reduction not in reduction_map:
        raise ValueError(f"Unknown reduction type: {reduction}")

    return reduction_map[reduction]()


def compute_weights(
    labels: Tensor,
    method: Literal["uniform", "inverse_freq", "balanced"] = "uniform",
) -> Tensor:
    """
    Compute sample weights for imbalanced datasets.

    Parameters
    ----------
    labels : Tensor
        Integer labels, shape (B, ...)
    method : str
        Weighting method:
        - "uniform": equal weights
        - "inverse_freq": inverse frequency weighting
        - "balanced": balanced class weights (sklearn style)

    Returns
    -------
    Tensor
        Weights, shape (B,)
    """
    if method == "uniform":
        return torch.ones(labels.shape[0], device=labels.device, dtype=torch.float32)

    # Flatten to get all label values
    flat_labels = labels.flatten()
    unique_labels, counts = torch.unique(flat_labels, return_counts=True)
    n_samples = flat_labels.numel()
    n_classes = len(unique_labels)

    # Both inverse_freq and balanced use the same formula
    if method in ("inverse_freq", "balanced"):
        # Weight inversely proportional to frequency: n_samples / (n_classes * class_count)
        class_weights = n_samples / (n_classes * counts.float())
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    # Map weights to samples
    weight_map = {int(label.item()): float(weight.item()) for label, weight in zip(unique_labels, class_weights)}
    sample_weights = torch.tensor(
        [weight_map.get(int(label.item()), 1.0) for label in flat_labels],
        device=labels.device,
        dtype=torch.float32,
    )

    # Reshape to (B,) by taking mean over spatial dimensions
    if labels.dim() > 1:
        sample_weights = sample_weights.view(labels.shape).mean(dim=tuple(range(1, labels.dim())))

    return sample_weights

