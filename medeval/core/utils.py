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
            f"coordinate dimension {spatial_dims}"
        )

    # Apply spacing: physical = voxel * spacing
    return coords * spacing_tensor


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

    # Compute scale factors
    scale_factors = [s / ts for s, ts in zip(spacing, target_spacing)]

    # Determine spatial dimensions
    ndim = len(spacing)
    if image.dim() == 4:  # (B, C, Y, X) or (B, Z, Y, X)
        if ndim == 2:
            # 2D: (B, C, Y, X)
            scale = (scale_factors[0], scale_factors[1])
        elif ndim == 3:
            # 3D without channel: (B, Z, Y, X)
            scale = (scale_factors[0], scale_factors[1], scale_factors[2])
        else:
            raise ValueError(f"Image dims {image.dim()} incompatible with spacing dims {ndim}")
    elif image.dim() == 5:  # (B, C, Z, Y, X)
        if ndim == 3:
            scale = (scale_factors[0], scale_factors[1], scale_factors[2])
        else:
            raise ValueError(f"Image dims {image.dim()} incompatible with spacing dims {ndim}")
    else:
        raise ValueError(f"Unsupported image dimensions: {image.dim()}")

    # Resample using torch's interpolate
    if ndim == 2:
        # 2D interpolation: (B, C, Y, X)
        mode_map = {"nearest": "nearest", "bilinear": "bilinear", "trilinear": "bilinear"}
        interp_mode = mode_map.get(mode, "bilinear")
        resampled = torch.nn.functional.interpolate(
            image, scale_factor=scale, mode=interp_mode, align_corners=False
        )
    else:  # 3D: (B, C, Z, Y, X) or (B, Z, Y, X)
        # PyTorch's interpolate supports 5D tensors (B, C, D, H, W) with trilinear mode
        if image.dim() == 4:  # (B, Z, Y, X) - add channel dimension
            image = image.unsqueeze(1)  # (B, 1, Z, Y, X)
            needs_squeeze = True
        else:  # image.dim() == 5: (B, C, Z, Y, X)
            needs_squeeze = False
        
        # Compute target size
        target_size = tuple(int(image.shape[i + 2] * scale_factors[i]) for i in range(3))
        
        # Use trilinear interpolation for 3D
        mode_map = {"nearest": "nearest", "bilinear": "trilinear", "trilinear": "trilinear"}
        interp_mode = mode_map.get(mode, "trilinear")
        resampled = torch.nn.functional.interpolate(
            image, size=target_size, mode=interp_mode, align_corners=False
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

    # Set ones for valid labels
    if ignore_index is not None:
        valid_mask = labels != ignore_index
        indices = labels[valid_mask].long()
        valid_indices = torch.nonzero(valid_mask, as_tuple=False)
        for idx, label_val in zip(valid_indices, indices):
            one_hot[tuple(idx) + (label_val,)] = 1.0
    else:
        indices = labels.long()
        # Use scatter_ or advanced indexing
        for i in range(num_classes):
            mask = indices == i
            one_hot[..., i][mask] = 1.0

    return one_hot


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

    if dim is None:
        # Infer dimensions based on shape
        if metrics.dim() >= 2:
            # Assume (B, C, ...) structure
            batch_dim = 0
            class_dim = 1 if metrics.dim() >= 2 else None
        else:
            batch_dim = 0
            class_dim = None

    if reduction == "mean-case":
        if per_class and class_dim is not None:
            # Average over cases, keep classes
            return metrics.mean(dim=batch_dim)
        else:
            # Average over cases
            if class_dim is not None:
                return metrics.mean(dim=(batch_dim, class_dim))
            else:
                return metrics.mean(dim=batch_dim)
    elif reduction == "mean-class":
        if class_dim is not None:
            return metrics.mean(dim=class_dim)
        else:
            return metrics  # No class dimension to reduce
    elif reduction == "global":
        if class_dim is not None:
            return metrics.mean(dim=(batch_dim, class_dim))
        else:
            return metrics.mean(dim=batch_dim)
    else:
        raise ValueError(f"Unknown reduction type: {reduction}")


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

    if method == "inverse_freq":
        # Weight inversely proportional to frequency
        class_weights = n_samples / (n_classes * counts.float())
    elif method == "balanced":
        # Balanced weights: n_samples / (n_classes * class_count)
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

