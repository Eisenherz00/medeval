"""Segmentation metrics for medical imaging evaluation.

This module provides overlap metrics, surface-based metrics, and calibration
metrics for 2D/3D segmentation tasks with proper spacing handling.
"""

from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.spatial.distance import cdist

from medeval.core.typing import ArrayLike, Device, Tensor, as_tensor
from medeval.core.utils import ReductionType, apply_spacing, compute_one_hot, reduce_metrics


try:
    from scipy.ndimage import distance_transform_edt
except ImportError:
    # Fallback for older scipy versions
    distance_transform_edt = None


# Helper to ensure batch dimension
def _ensure_batch(pred: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """Ensure a leading batch dimension.

    This repo currently mixes conventions. For metrics we standardize to:
    - single 2D mask: (H, W)  -> (1, H, W)
    - single 3D mask: (H, W, D)/(Z,Y,X) -> (1, H, W, D)

    We do NOT try to infer channels here; channel handling is done elsewhere.
    """
    if pred.dim() in (2, 3):
        pred = pred.unsqueeze(0)
    if target.dim() in (2, 3):
        target = target.unsqueeze(0)
    return pred, target



def _ensure_binary(pred: Tensor, target: Tensor, threshold: float = 0.5) -> Tuple[Tensor, Tensor]:
    """Ensure binary masks from predictions and targets.

    Notes
    -----
    - Ensures a batch dimension for unbatched inputs.
    - Only squeezes a channel dimension when we are in a clear (B, C, ...) layout
      (i.e., tensors with >=4 dims).
    """
    pred, target = _ensure_batch(pred, target)

    # Ensure same number of dimensions
    while pred.dim() < target.dim():
        pred = pred.unsqueeze(0)
    while target.dim() < pred.dim():
        target = target.unsqueeze(0)

    # Remove channel dimension if it's size 1 (only for (B, C, ...))
    if pred.dim() >= 4 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    if target.dim() >= 4 and target.shape[1] == 1:
        target = target.squeeze(1)

    if pred.dtype.is_floating_point:
        pred = (pred > threshold).float()
    else:
        pred = pred.float()

    if target.dtype.is_floating_point:
        target = (target > threshold).float()
    else:
        target = target.float()

    return pred, target


def _get_surface_points(mask: np.ndarray, spacing: Optional[Tuple[float, ...]] = None) -> np.ndarray:
    """
    Extract surface points from a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (2D or 3D)
    spacing : Tuple[float, ...], optional
        Physical spacing for coordinates

    Returns
    -------
    np.ndarray
        Surface points, shape (N, D) where D is spatial dimension
    """
    if spacing is None:
        spacing = (1.0,) * mask.ndim

    # Validate spacing
    if len(spacing) != mask.ndim:
        raise ValueError(
            f"Spacing dimension {len(spacing)} does not match mask.ndim {mask.ndim}"
        )

    # Use morphological operations to find surface
    # Surface = mask - eroded(mask)
    structure = np.ones((3,) * mask.ndim, dtype=bool)
    eroded = ndimage.binary_erosion(mask.astype(bool), structure=structure)
    surface = mask.astype(bool) & (~eroded)

    # Get coordinates of surface points
    coords = np.argwhere(surface)

    # Apply spacing
    if coords.size == 0:
        return coords.astype(np.float32)

    coords = coords.astype(np.float32)
    for i, s in enumerate(spacing):
        coords[:, i] *= float(s)

    return coords


def _compute_hausdorff_distance(
    pred_surface: np.ndarray, target_surface: np.ndarray, percentile: Optional[float] = None
) -> float:
    """
    Compute Hausdorff distance between two surfaces.

    Parameters
    ----------
    pred_surface : np.ndarray
        Surface points from prediction, shape (N, D)
    target_surface : np.ndarray
        Surface points from target, shape (M, D)
    percentile : float, optional
        If provided, compute percentile instead of maximum (e.g., 95 for HD95)

    Returns
    -------
    float
        Hausdorff distance (or percentile)
    """
    # Undefined if one of the surfaces is empty.
    # Convention in this repo: caller maps (both empty)->0.0, (one empty)->NaN,
    # and higher-level aggregation should be NaN-aware.
    if len(pred_surface) == 0 or len(target_surface) == 0:
        return float("nan")

    # Compute distances from pred to target
    dists_pred_to_target = cdist(pred_surface, target_surface, metric="euclidean")
    min_dists_pred = np.min(dists_pred_to_target, axis=1)

    # Compute distances from target to pred
    dists_target_to_pred = cdist(target_surface, pred_surface, metric="euclidean")
    min_dists_target = np.min(dists_target_to_pred, axis=1)

    # Symmetric Hausdorff distance
    if percentile is not None:
        # Use percentile of all distances
        all_dists = np.concatenate([min_dists_pred, min_dists_target])
        return float(np.percentile(all_dists, percentile))
    else:
        # Maximum distance
        return float(max(np.max(min_dists_pred), np.max(min_dists_target)))


def dice_score(
    pred: ArrayLike,
    target: ArrayLike,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
    reduction: ReductionType = "mean-case",
) -> Tensor:
    """
    Compute Dice coefficient (F1 score) for binary or multi-class segmentation.

    Formula: Dice = 2 * |A ∩ B| / (|A| + |B|)

    Parameters
    ----------
    pred : ArrayLike
        Predictions, shape (B, C, ...) or (B, ...) for binary
    target : ArrayLike
        Ground truth, same shape as pred
    threshold : float
        Threshold for binary predictions (default: 0.5)
    ignore_index : int, optional
        Label index to ignore
    reduction : ReductionType
        Reduction strategy: "none", "mean-case", "mean-class", "global"

    Returns
    -------
    Tensor
        Dice scores, shape depends on reduction
    """
    pred = as_tensor(pred)
    target = as_tensor(target)
    pred, target = _ensure_batch(pred, target)

    # Handle ignore_index
    if ignore_index is not None:
        mask = target != ignore_index
        pred = pred * mask.float()
        target = target * mask.float()

    # Check if multi-class (has channel dimension with multiple classes)
    is_multi_class = (
        pred.dim() >= 3
        and target.dim() >= 3
        and pred.shape[1] > 1
        and (target.shape[1] > 1 if target.dim() > 2 else False)
    )

    if is_multi_class:
        # Multi-class case: assume one-hot or logits format (B, C, ...)
        # Convert to class indices
        if pred.dtype.is_floating_point:
            pred_classes = pred.argmax(dim=1)  # (B, ...)
        else:
            pred_classes = pred.argmax(dim=1)

        if target.dtype.is_floating_point and target.shape[1] > 1:
            target_classes = target.argmax(dim=1)  # (B, ...)
        else:
            target_classes = target.squeeze(1) if target.shape[1] == 1 else target

        # Compute per-class Dice
        num_classes = pred.shape[1]
        dice_scores = []
        for c in range(num_classes):
            pred_c = (pred_classes == c).float()
            target_c = (target_classes == c).float()

            intersection = (pred_c * target_c).sum(dim=tuple(range(1, pred_c.dim())))
            union = pred_c.sum(dim=tuple(range(1, pred_c.dim()))) + target_c.sum(
                dim=tuple(range(1, target_c.dim()))
            )

            # Handle empty sets: both empty = 1.0, otherwise 0.0 if union is 0
            pred_c_sum = pred_c.sum(dim=tuple(range(1, pred_c.dim())))
            target_c_sum = target_c.sum(dim=tuple(range(1, target_c.dim())))
            both_empty = (pred_c_sum == 0) & (target_c_sum == 0)
            dice = torch.where(
                both_empty,
                torch.tensor(1.0, device=pred.device),
                torch.where(
                    union > 0, 2.0 * intersection / union, torch.tensor(0.0, device=pred.device)
                ),
            )
            dice_scores.append(dice)

        dice_tensor = torch.stack(dice_scores, dim=1)  # (B, C)
    else:
        # Binary case: ensure same shape
        if pred.dim() > target.dim():
            target = target.unsqueeze(1)
        elif target.dim() > pred.dim():
            pred = pred.unsqueeze(1)

        pred, target = _ensure_binary(pred, target, threshold)

        intersection = (pred * target).sum(dim=tuple(range(1, pred.dim())))
        union = pred.sum(dim=tuple(range(1, pred.dim()))) + target.sum(
            dim=tuple(range(1, target.dim()))
        )

        # Handle empty sets: if both are empty, return 1.0 (perfect match)
        # If only one is empty, return 0.0
        pred_sum = pred.sum(dim=tuple(range(1, pred.dim())))
        target_sum = target.sum(dim=tuple(range(1, target.dim())))
        both_empty = (pred_sum == 0) & (target_sum == 0)
        dice = torch.where(
            both_empty,
            torch.tensor(1.0, device=pred.device),
            torch.where(
                union > 0, 2.0 * intersection / union, torch.tensor(0.0, device=pred.device)
            ),
        )
        dice_tensor = dice.unsqueeze(1) if reduction != "none" else dice

    return reduce_metrics(dice_tensor, reduction=reduction)


def jaccard_index(
    pred: ArrayLike,
    target: ArrayLike,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
    reduction: ReductionType = "mean-case",
) -> Tensor:
    """
    Compute Jaccard index (IoU) for segmentation.

    Formula: IoU = |A ∩ B| / |A ∪ B|

    Parameters
    ----------
    pred : ArrayLike
        Predictions
    target : ArrayLike
        Ground truth
    threshold : float
        Threshold for binary predictions
    ignore_index : int, optional
        Label index to ignore
    reduction : ReductionType
        Reduction strategy

    Returns
    -------
    Tensor
        Jaccard indices
    """
    pred = as_tensor(pred)
    target = as_tensor(target)
    pred, target = _ensure_batch(pred, target)

    if ignore_index is not None:
        mask = target != ignore_index
        pred = pred * mask.float()
        target = target * mask.float()

    # Check if multi-class
    is_multi_class = (
        pred.dim() >= 3
        and target.dim() >= 3
        and pred.shape[1] > 1
        and (target.shape[1] > 1 if target.dim() > 2 else False)
    )

    if is_multi_class:
        if pred.dtype.is_floating_point:
            pred_classes = pred.argmax(dim=1)
        else:
            pred_classes = pred.argmax(dim=1)
        target_classes = target.argmax(dim=1) if target.dim() > 2 and target.shape[1] > 1 else target.squeeze(1) if target.dim() > 2 else target

        num_classes = pred.shape[1]
        jaccard_scores = []
        for c in range(num_classes):
            pred_c = (pred_classes == c).float()
            target_c = (target_classes == c).float()

            intersection = (pred_c * target_c).sum(dim=tuple(range(1, pred_c.dim())))
            union = (pred_c + target_c).clamp(0, 1).sum(dim=tuple(range(1, pred_c.dim())))

            pred_c_sum = pred_c.sum(dim=tuple(range(1, pred_c.dim())))
            target_c_sum = target_c.sum(dim=tuple(range(1, target_c.dim())))
            both_empty = (pred_c_sum == 0) & (target_c_sum == 0)

            jaccard = torch.where(
                both_empty,
                torch.tensor(1.0, device=pred.device),
                torch.where(
                    union > 0, intersection / union, torch.tensor(0.0, device=pred.device)
                ),
            )
            jaccard_scores.append(jaccard)

        jaccard_tensor = torch.stack(jaccard_scores, dim=1)
    else:
        # Ensure same shape
        if pred.dim() > target.dim():
            target = target.unsqueeze(1)
        elif target.dim() > pred.dim():
            pred = pred.unsqueeze(1)

        pred, target = _ensure_binary(pred, target, threshold)

        intersection = (pred * target).sum(dim=tuple(range(1, pred.dim())))
        union = (pred + target).clamp(0, 1).sum(dim=tuple(range(1, pred.dim())))

        pred_sum = pred.sum(dim=tuple(range(1, pred.dim())))
        target_sum = target.sum(dim=tuple(range(1, target.dim())))
        both_empty = (pred_sum == 0) & (target_sum == 0)

        jaccard = torch.where(
            both_empty,
            torch.tensor(1.0, device=pred.device),
            torch.where(
                union > 0, intersection / union, torch.tensor(0.0, device=pred.device)
            ),
        )
        jaccard_tensor = jaccard.unsqueeze(1) if reduction != "none" else jaccard

    return reduce_metrics(jaccard_tensor, reduction=reduction)


def precision_score(
    pred: ArrayLike,
    target: ArrayLike,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
    reduction: ReductionType = "mean-case",
) -> Tensor:
    """
    Compute precision (positive predictive value).

    Formula: Precision = TP / (TP + FP)

    Parameters
    ----------
    pred : ArrayLike
        Predictions
    target : ArrayLike
        Ground truth
    threshold : float
        Threshold for binary predictions
    ignore_index : int, optional
        Label index to ignore
    reduction : ReductionType
        Reduction strategy

    Returns
    -------
    Tensor
        Precision scores
    """
    pred = as_tensor(pred)
    target = as_tensor(target)
    pred, target = _ensure_batch(pred, target)

    if ignore_index is not None:
        mask = target != ignore_index
        pred = pred * mask.float()
        target = target * mask.float()

    pred, target = _ensure_binary(pred, target, threshold)

    tp = (pred * target).sum(dim=tuple(range(1, pred.dim())))
    fp = (pred * (1 - target)).sum(dim=tuple(range(1, pred.dim())))

    pred_sum = pred.sum(dim=tuple(range(1, pred.dim())))
    target_sum = target.sum(dim=tuple(range(1, target.dim())))
    both_empty = (pred_sum == 0) & (target_sum == 0)

    precision = torch.where(
        both_empty,
        torch.tensor(1.0, device=pred.device),
        torch.where(
            (tp + fp) > 0, tp / (tp + fp), torch.tensor(0.0, device=pred.device)
        ),
    )

    return reduce_metrics(precision.unsqueeze(1), reduction=reduction)


def recall_score(
    pred: ArrayLike,
    target: ArrayLike,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
    reduction: ReductionType = "mean-case",
) -> Tensor:
    """
    Compute recall (sensitivity, true positive rate).

    Formula: Recall = TP / (TP + FN)

    Parameters
    ----------
    pred : ArrayLike
        Predictions
    target : ArrayLike
        Ground truth
    threshold : float
        Threshold for binary predictions
    ignore_index : int, optional
        Label index to ignore
    reduction : ReductionType
        Reduction strategy

    Returns
    -------
    Tensor
        Recall scores
    """
    pred = as_tensor(pred)
    target = as_tensor(target)
    pred, target = _ensure_batch(pred, target)

    if ignore_index is not None:
        mask = target != ignore_index
        pred = pred * mask.float()
        target = target * mask.float()

    pred, target = _ensure_binary(pred, target, threshold)

    tp = (pred * target).sum(dim=tuple(range(1, pred.dim())))
    fn = ((1 - pred) * target).sum(dim=tuple(range(1, pred.dim())))

    pred_sum = pred.sum(dim=tuple(range(1, pred.dim())))
    target_sum = target.sum(dim=tuple(range(1, target.dim())))
    both_empty = (pred_sum == 0) & (target_sum == 0)

    recall = torch.where(
        both_empty,
        torch.tensor(1.0, device=pred.device),
        torch.where(
            (tp + fn) > 0, tp / (tp + fn), torch.tensor(0.0, device=pred.device)
        ),
    )

    return reduce_metrics(recall.unsqueeze(1), reduction=reduction)


def volumetric_similarity(
    pred: ArrayLike,
    target: ArrayLike,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
    reduction: ReductionType = "mean-case",
) -> Tensor:
    """
    Compute volumetric similarity.

    Formula: VS = 1 - |A - B| / (A + B)

    Parameters
    ----------
    pred : ArrayLike
        Predictions
    target : ArrayLike
        Ground truth
    threshold : float
        Threshold for binary predictions
    ignore_index : int, optional
        Label index to ignore
    reduction : ReductionType
        Reduction strategy

    Returns
    -------
    Tensor
        Volumetric similarity scores
    """
    pred = as_tensor(pred)
    target = as_tensor(target)
    pred, target = _ensure_batch(pred, target)

    if ignore_index is not None:
        mask = target != ignore_index
        pred = pred * mask.float()
        target = target * mask.float()

    pred, target = _ensure_binary(pred, target, threshold)

    vol_pred = pred.sum(dim=tuple(range(1, pred.dim())))
    vol_target = target.sum(dim=tuple(range(1, target.dim())))

    both_empty = (vol_pred == 0) & (vol_target == 0)
    vs = torch.where(
        both_empty,
        torch.tensor(1.0, device=pred.device),
        torch.where(
            (vol_pred + vol_target) > 0,
            1.0 - torch.abs(vol_pred - vol_target) / (vol_pred + vol_target),
            torch.tensor(0.0, device=pred.device),
        ),
    )

    return reduce_metrics(vs.unsqueeze(1), reduction=reduction)


def hausdorff_distance(
    pred: ArrayLike,
    target: ArrayLike,
    spacing: Optional[Tuple[float, ...]] = None,
    percentile: Optional[float] = None,
    ignore_index: Optional[int] = None,
    reduction: ReductionType = "mean-case",
) -> Tensor:
    """
    Compute Hausdorff distance between prediction and target surfaces.

    Parameters
    ----------
    pred : ArrayLike
        Binary predictions, shape (B, ...) or (B, C, ...)
    target : ArrayLike
        Binary ground truth, same shape
    spacing : Tuple[float, ...], optional
        Physical spacing for distance computation
    percentile : float, optional
        If provided (e.g., 95), compute HD95 instead of maximum HD
    ignore_index : int, optional
        Label index to ignore
    reduction : ReductionType
        Reduction strategy

    Returns
    -------
    Tensor
        Hausdorff distances
    """
    pred = as_tensor(pred)
    target = as_tensor(target)
    pred, target = _ensure_batch(pred, target)

    # Convert to numpy for surface computation
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    batch_size = pred_np.shape[0]
    hd_scores = []

    for b in range(batch_size):
        pred_b = pred_np[b]
        target_b = target_np[b]

        # Handle multi-class: compute per-class HD
        if pred_b.ndim > 1 and pred_b.shape[0] > 1:
            # Multi-class: one-hot or channel-first
            class_hds = []
            for c in range(pred_b.shape[0]):
                pred_c = pred_b[c] > 0.5
                target_c = target_b[c] > 0.5 if target_b.ndim > 1 else target_b > 0.5

                if ignore_index is not None and c == ignore_index:
                    # Skip ignored class index
                    continue

                pred_surface = _get_surface_points(pred_c, spacing)
                target_surface = _get_surface_points(target_c, spacing)

                # Empty-set handling
                if len(pred_surface) == 0 and len(target_surface) == 0:
                    class_hds.append(0.0)
                    continue
                if len(pred_surface) == 0 or len(target_surface) == 0:
                    class_hds.append(float("nan"))
                    continue

                hd = _compute_hausdorff_distance(pred_surface, target_surface, percentile)
                class_hds.append(hd)

            if class_hds:
                # NaN-aware mean: ignore undefined classes (e.g., one empty surface)
                if np.all(np.isnan(class_hds)):
                    hd_scores.append(float("nan"))
                else:
                    hd_scores.append(float(np.nanmean(class_hds)))
            else:
                hd_scores.append(float("nan"))
        else:
            # Binary case
            if pred_b.ndim > 1:
                pred_b = pred_b[0] if pred_b.shape[0] == 1 else pred_b.squeeze(0)
            if target_b.ndim > 1:
                target_b = target_b[0] if target_b.shape[0] == 1 else target_b.squeeze(0)

            pred_binary = pred_b > 0.5
            target_binary = target_b > 0.5

            if ignore_index is not None:
                mask = target_binary != ignore_index
                pred_binary = pred_binary & mask
                target_binary = target_binary & mask

            pred_surface = _get_surface_points(pred_binary, spacing)
            target_surface = _get_surface_points(target_binary, spacing)

            # Empty-set handling
            if len(pred_surface) == 0 and len(target_surface) == 0:
                hd_scores.append(0.0)
                continue
            if len(pred_surface) == 0 or len(target_surface) == 0:
                hd_scores.append(float("nan"))
                continue

            hd = _compute_hausdorff_distance(pred_surface, target_surface, percentile)
            hd_scores.append(hd)

    hd_tensor = torch.tensor(hd_scores, device=pred.device, dtype=torch.float32)
    return reduce_metrics(hd_tensor.unsqueeze(1), reduction=reduction)


def hausdorff_distance_95(
    pred: ArrayLike,
    target: ArrayLike,
    spacing: Optional[Tuple[float, ...]] = None,
    ignore_index: Optional[int] = None,
    reduction: ReductionType = "mean-case",
) -> Tensor:
    """
    Compute 95th percentile Hausdorff distance.

    Parameters
    ----------
    pred : ArrayLike
        Predictions
    target : ArrayLike
        Ground truth
    spacing : Tuple[float, ...], optional
        Physical spacing
    ignore_index : int, optional
        Label index to ignore
    reduction : ReductionType
        Reduction strategy

    Returns
    -------
    Tensor
        HD95 scores
    """
    return hausdorff_distance(pred, target, spacing, percentile=95.0, ignore_index=ignore_index, reduction=reduction)


def average_symmetric_surface_distance(
    pred: ArrayLike,
    target: ArrayLike,
    spacing: Optional[Tuple[float, ...]] = None,
    ignore_index: Optional[int] = None,
    reduction: ReductionType = "mean-case",
) -> Tensor:
    """
    Compute Average Symmetric Surface Distance (ASSD).

    ASSD = (mean distance from pred to target + mean distance from target to pred) / 2

    Parameters
    ----------
    pred : ArrayLike
        Predictions
    target : ArrayLike
        Ground truth
    spacing : Tuple[float, ...], optional
        Physical spacing
    ignore_index : int, optional
        Label index to ignore
    reduction : ReductionType
        Reduction strategy

    Returns
    -------
    Tensor
        ASSD scores
    """
    pred = as_tensor(pred)
    target = as_tensor(target)
    pred, target = _ensure_batch(pred, target)

    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    batch_size = pred_np.shape[0]
    assd_scores = []

    for b in range(batch_size):
        pred_b = pred_np[b]
        target_b = target_np[b]

        # Handle binary case
        if pred_b.ndim > 1 and pred_b.shape[0] > 1:
            # Multi-class: average over classes
            class_assds = []
            for c in range(pred_b.shape[0]):
                pred_c = pred_b[c] > 0.5
                target_c = target_b[c] > 0.5 if target_b.ndim > 1 else target_b > 0.5

                if ignore_index is not None and c == ignore_index:
                    continue

                pred_surface = _get_surface_points(pred_c, spacing)
                target_surface = _get_surface_points(target_c, spacing)

                # Empty-set handling:
                # - both empty => 0.0 (perfect match)
                # - one empty  => NaN (undefined)
                if len(pred_surface) == 0 and len(target_surface) == 0:
                    class_assds.append(0.0)
                    continue
                if len(pred_surface) == 0 or len(target_surface) == 0:
                    class_assds.append(float("nan"))
                    continue

                # Compute distances
                dists_pred_to_target = cdist(pred_surface, target_surface, metric="euclidean")
                dists_target_to_pred = cdist(target_surface, pred_surface, metric="euclidean")

                mean_dist_pred = np.mean(np.min(dists_pred_to_target, axis=1))
                mean_dist_target = np.mean(np.min(dists_target_to_pred, axis=1))

                assd = (mean_dist_pred + mean_dist_target) / 2.0
                class_assds.append(assd)

            if class_assds:
                if np.all(np.isnan(class_assds)):
                    assd_scores.append(float("nan"))
                else:
                    assd_scores.append(float(np.nanmean(class_assds)))
            else:
                assd_scores.append(float("nan"))
        else:
            # Binary case
            if pred_b.ndim > 1:
                pred_b = pred_b[0] if pred_b.shape[0] == 1 else pred_b.squeeze(0)
            if target_b.ndim > 1:
                target_b = target_b[0] if target_b.shape[0] == 1 else target_b.squeeze(0)

            pred_binary = pred_b > 0.5
            target_binary = target_b > 0.5

            if ignore_index is not None:
                mask = target_binary != ignore_index
                pred_binary = pred_binary & mask
                target_binary = target_binary & mask

            pred_surface = _get_surface_points(pred_binary, spacing)
            target_surface = _get_surface_points(target_binary, spacing)

            # Empty-set handling:
            # - both empty => 0.0
            # - one empty  => NaN
            if len(pred_surface) == 0 and len(target_surface) == 0:
                assd_scores.append(0.0)
                continue
            if len(pred_surface) == 0 or len(target_surface) == 0:
                assd_scores.append(float("nan"))
                continue

            dists_pred_to_target = cdist(pred_surface, target_surface, metric="euclidean")
            dists_target_to_pred = cdist(target_surface, pred_surface, metric="euclidean")

            mean_dist_pred = np.mean(np.min(dists_pred_to_target, axis=1))
            mean_dist_target = np.mean(np.min(dists_target_to_pred, axis=1))

            assd = (mean_dist_pred + mean_dist_target) / 2.0
            assd_scores.append(assd)

    assd_tensor = torch.tensor(assd_scores, device=pred.device, dtype=torch.float32)
    return reduce_metrics(assd_tensor.unsqueeze(1), reduction=reduction)


def surface_dice(
    pred: ArrayLike,
    target: ArrayLike,
    tolerance: float = 1.0,
    spacing: Optional[Tuple[float, ...]] = None,
    ignore_index: Optional[int] = None,
    reduction: ReductionType = "mean-case",
) -> Tensor:
    """
    Compute Surface Dice at a given tolerance.

    Surface Dice = fraction of surface points within tolerance distance.

    Parameters
    ----------
    pred : ArrayLike
        Predictions
    target : ArrayLike
        Ground truth
    tolerance : float
        Distance tolerance (in physical units if spacing provided)
    spacing : Tuple[float, ...], optional
        Physical spacing
    ignore_index : int, optional
        Label index to ignore
    reduction : ReductionType
        Reduction strategy

    Returns
    -------
    Tensor
        Surface Dice scores
    """
    pred = as_tensor(pred)
    target = as_tensor(target)
    pred, target = _ensure_batch(pred, target)

    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    batch_size = pred_np.shape[0]
    surface_dice_scores = []

    for b in range(batch_size):
        pred_b = pred_np[b]
        target_b = target_np[b]

        # Handle binary case
        if pred_b.ndim > 1 and pred_b.shape[0] > 1:
            class_scores = []
            for c in range(pred_b.shape[0]):
                pred_c = pred_b[c] > 0.5
                target_c = target_b[c] > 0.5 if target_b.ndim > 1 else target_b > 0.5

                if ignore_index is not None and c == ignore_index:
                    continue

                pred_surface = _get_surface_points(pred_c, spacing)
                target_surface = _get_surface_points(target_c, spacing)

                # Empty-set handling:
                # - both empty => 1.0 (perfect match)
                # - one empty  => 0.0 (miss)
                if len(pred_surface) == 0 and len(target_surface) == 0:
                    class_scores.append(1.0)
                    continue
                if len(pred_surface) == 0 or len(target_surface) == 0:
                    class_scores.append(0.0)
                    continue

                # Compute distances
                dists_pred_to_target = cdist(pred_surface, target_surface, metric="euclidean")
                min_dists_pred = np.min(dists_pred_to_target, axis=1)

                dists_target_to_pred = cdist(target_surface, pred_surface, metric="euclidean")
                min_dists_target = np.min(dists_target_to_pred, axis=1)

                # Fraction within tolerance
                within_tol_pred = np.sum(min_dists_pred <= tolerance) / len(min_dists_pred)
                within_tol_target = np.sum(min_dists_target <= tolerance) / len(min_dists_target)

                surface_dice = (within_tol_pred + within_tol_target) / 2.0
                class_scores.append(surface_dice)

            if class_scores:
                surface_dice_scores.append(float(np.mean(class_scores)))
            else:
                surface_dice_scores.append(0.0)
        else:
            # Binary case
            if pred_b.ndim > 1:
                pred_b = pred_b[0] if pred_b.shape[0] == 1 else pred_b.squeeze(0)
            if target_b.ndim > 1:
                target_b = target_b[0] if target_b.shape[0] == 1 else target_b.squeeze(0)

            pred_binary = pred_b > 0.5
            target_binary = target_b > 0.5

            if ignore_index is not None:
                mask = target_binary != ignore_index
                pred_binary = pred_binary & mask
                target_binary = target_binary & mask

            pred_surface = _get_surface_points(pred_binary, spacing)
            target_surface = _get_surface_points(target_binary, spacing)

            # Empty-set handling:
            # - both empty => 1.0
            # - one empty  => 0.0
            if len(pred_surface) == 0 and len(target_surface) == 0:
                surface_dice_scores.append(1.0)
                continue
            if len(pred_surface) == 0 or len(target_surface) == 0:
                surface_dice_scores.append(0.0)
                continue

            dists_pred_to_target = cdist(pred_surface, target_surface, metric="euclidean")
            min_dists_pred = np.min(dists_pred_to_target, axis=1)

            dists_target_to_pred = cdist(target_surface, pred_surface, metric="euclidean")
            min_dists_target = np.min(dists_target_to_pred, axis=1)

            within_tol_pred = np.sum(min_dists_pred <= tolerance) / len(min_dists_pred)
            within_tol_target = np.sum(min_dists_target <= tolerance) / len(min_dists_target)

            surface_dice = (within_tol_pred + within_tol_target) / 2.0
            surface_dice_scores.append(surface_dice)

    surface_dice_tensor = torch.tensor(surface_dice_scores, device=pred.device, dtype=torch.float32)
    return reduce_metrics(surface_dice_tensor.unsqueeze(1), reduction=reduction)


def soft_dice_score(
    pred: ArrayLike,
    target: ArrayLike,
    smooth: float = 1e-6,
    ignore_index: Optional[int] = None,
    reduction: ReductionType = "mean-case",
) -> Tensor:
    """
    Compute soft Dice score for probabilistic predictions.

    Formula: soft-Dice = 2 * sum(p * t) / (sum(p) + sum(t))

    Parameters
    ----------
    pred : ArrayLike
        Probabilistic predictions, shape (B, C, ...) or (B, ...)
    target : ArrayLike
        Ground truth (can be binary or probabilistic), same shape
    smooth : float
        Smoothing factor to avoid division by zero
    ignore_index : int, optional
        Label index to ignore
    reduction : ReductionType
        Reduction strategy

    Returns
    -------
    Tensor
        Soft Dice scores
    """
    pred = as_tensor(pred).float()
    target = as_tensor(target).float()

    if ignore_index is not None:
        mask = target != ignore_index
        pred = pred * mask.float()
        target = target * mask.float()

    # Ensure values in [0, 1]
    pred = torch.clamp(pred, 0, 1)
    target = torch.clamp(target, 0, 1)

    intersection = (pred * target).sum(dim=tuple(range(1, pred.dim())))
    pred_sum = pred.sum(dim=tuple(range(1, pred.dim())))
    target_sum = target.sum(dim=tuple(range(1, target.dim())))

    soft_dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)

    return reduce_metrics(soft_dice.unsqueeze(1), reduction=reduction)


def brier_score(
    pred: ArrayLike,
    target: ArrayLike,
    ignore_index: Optional[int] = None,
    reduction: ReductionType = "mean-case",
) -> Tensor:
    """
    Compute Brier score (per-voxel mean squared error).

    Formula: Brier = mean((p - t)^2)

    Parameters
    ----------
    pred : ArrayLike
        Probabilistic predictions
    target : ArrayLike
        Binary or probabilistic ground truth
    ignore_index : int, optional
        Label index to ignore
    reduction : ReductionType
        Reduction strategy

    Returns
    -------
    Tensor
        Brier scores
    """
    pred = as_tensor(pred).float()
    target = as_tensor(target).float()

    if ignore_index is not None:
        mask = target != ignore_index
        pred = pred * mask.float()
        target = target * mask.float()

    # Ensure values in [0, 1]
    pred = torch.clamp(pred, 0, 1)
    target = torch.clamp(target, 0, 1)

    # Compute per-voxel squared error
    squared_error = (pred - target) ** 2

    # Mean over spatial dimensions
    brier = squared_error.mean(dim=tuple(range(1, pred.dim())))

    return reduce_metrics(brier.unsqueeze(1), reduction=reduction)


def threshold_sweep(
    pred: ArrayLike,
    target: ArrayLike,
    thresholds: Optional[List[float]] = None,
    metric_fn: str = "dice",
    ignore_index: Optional[int] = None,
) -> Dict[str, Tensor]:
    """
    Compute metric across multiple thresholds.

    Parameters
    ----------
    pred : ArrayLike
        Probabilistic predictions
    target : ArrayLike
        Ground truth
    thresholds : List[float], optional
        Threshold values to test. If None, uses [0.1, 0.2, ..., 0.9]
    metric_fn : str
        Metric to compute: "dice", "jaccard", "precision", "recall"
    ignore_index : int, optional
        Label index to ignore

    Returns
    -------
    Dict[str, Tensor]
        Dictionary with threshold values and corresponding metric scores
    """
    if thresholds is None:
        thresholds = [0.1 * i for i in range(1, 10)]

    metric_map = {
        "dice": dice_score,
        "jaccard": jaccard_index,
        "precision": precision_score,
        "recall": recall_score,
    }

    if metric_fn not in metric_map:
        raise ValueError(f"Unknown metric: {metric_fn}. Choose from {list(metric_map.keys())}")

    metric_func = metric_map[metric_fn]
    results = {"thresholds": thresholds, "scores": []}

    for threshold in thresholds:
        score = metric_func(pred, target, threshold=threshold, ignore_index=ignore_index, reduction="none")
        results["scores"].append(score)

    results["scores"] = torch.stack(results["scores"], dim=0)  # (n_thresholds, B, ...)
    return results


def compute_segmentation_metrics(
    pred: ArrayLike,
    target: ArrayLike,
    spacing: Optional[Tuple[float, ...]] = None,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
    include_surface: bool = True,
    include_calibration: bool = False,
    reduction: ReductionType = "mean-case",
    per_class: bool = False,
) -> Dict[str, Tensor]:
    """
    Compute comprehensive segmentation metrics.

    Parameters
    ----------
    pred : ArrayLike
        Predictions
    target : ArrayLike
        Ground truth
    spacing : Tuple[float, ...], optional
        Physical spacing for surface metrics
    threshold : float
        Threshold for binary predictions
    ignore_index : int, optional
        Label index to ignore
    include_surface : bool
        If True, compute surface-based metrics (slower)
    include_calibration : bool
        If True, compute calibration metrics
    reduction : ReductionType
        Reduction strategy
    per_class : bool
        If True, return per-class metrics

    Returns
    -------
    Dict[str, Tensor]
        Dictionary of metric names to values
    """
    results = {}

    # Overlap metrics
    results["dice"] = dice_score(pred, target, threshold, ignore_index, reduction)
    results["jaccard"] = jaccard_index(pred, target, threshold, ignore_index, reduction)
    results["precision"] = precision_score(pred, target, threshold, ignore_index, reduction)
    results["recall"] = recall_score(pred, target, threshold, ignore_index, reduction)
    results["volumetric_similarity"] = volumetric_similarity(
        pred, target, threshold, ignore_index, reduction
    )

    # Surface metrics
    if include_surface and spacing is not None:
        results["hausdorff"] = hausdorff_distance(pred, target, spacing, ignore_index=ignore_index, reduction=reduction)
        results["hausdorff_95"] = hausdorff_distance_95(pred, target, spacing, ignore_index=ignore_index, reduction=reduction)
        results["assd"] = average_symmetric_surface_distance(
            pred, target, spacing, ignore_index=ignore_index, reduction=reduction
        )
        results["surface_dice"] = surface_dice(
            pred, target, tolerance=1.0, spacing=spacing, ignore_index=ignore_index, reduction=reduction
        )

    # Calibration metrics
    if include_calibration:
        results["soft_dice"] = soft_dice_score(pred, target, ignore_index=ignore_index, reduction=reduction)
        results["brier"] = brier_score(pred, target, ignore_index=ignore_index, reduction=reduction)

    return results
