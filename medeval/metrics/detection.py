"""Detection metrics for bounding boxes and instance segmentation."""

from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from medeval.core.typing import ArrayLike, Tensor, as_tensor

try:
    from scipy.stats import entropy
except ImportError:
    entropy = None


def box_iou_2d(
    boxes1: ArrayLike,
    boxes2: ArrayLike,
    format: Literal["xyxy", "xywh", "cxcywh"] = "xyxy",
) -> Tensor:
    """
    Compute IoU for 2D bounding boxes.

    Parameters
    ----------
    boxes1 : ArrayLike
        First set of boxes, shape (N, 4)
    boxes2 : ArrayLike
        Second set of boxes, shape (M, 4)
    format : str
        Box format: "xyxy" (x1, y1, x2, y2), "xywh" (x, y, w, h), "cxcywh" (cx, cy, w, h)

    Returns
    -------
    Tensor
        IoU matrix, shape (N, M)
    """
    boxes1 = as_tensor(boxes1).float()
    boxes2 = as_tensor(boxes2).float()

    # Convert to xyxy format
    if format == "xywh":
        boxes1 = _xywh_to_xyxy(boxes1)
        boxes2 = _xywh_to_xyxy(boxes2)
    elif format == "cxcywh":
        boxes1 = _cxcywh_to_xyxy(boxes1)
        boxes2 = _cxcywh_to_xyxy(boxes2)

    # Compute areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    # Compute union
    union = area1[:, None] + area2[None, :] - inter

    # Compute IoU
    iou = inter / union.clamp(min=1e-6)

    return iou


def box_iou_3d(
    boxes1: ArrayLike,
    boxes2: ArrayLike,
    format: Literal["xyzxyz", "xyzwhd", "cxcyczwhd"] = "xyzxyz",
) -> Tensor:
    """
    Compute IoU for 3D bounding boxes.

    Parameters
    ----------
    boxes1 : ArrayLike
        First set of boxes, shape (N, 6)
    boxes2 : ArrayLike
        Second set of boxes, shape (M, 6)
    format : str
        Box format: "xyzxyz" (x1, y1, z1, x2, y2, z2), "xyzwhd" (x, y, z, w, h, d),
                    "cxcyczwhd" (cx, cy, cz, w, h, d)

    Returns
    -------
    Tensor
        IoU matrix, shape (N, M)
    """
    boxes1 = as_tensor(boxes1).float()
    boxes2 = as_tensor(boxes2).float()

    # Convert to xyzxyz format
    if format == "xyzwhd":
        boxes1 = _xyzwhd_to_xyzxyz(boxes1)
        boxes2 = _xyzwhd_to_xyzxyz(boxes2)
    elif format == "cxcyczwhd":
        boxes1 = _cxcyczwhd_to_xyzxyz(boxes1)
        boxes2 = _cxcyczwhd_to_xyzxyz(boxes2)

    # Compute volumes
    vol1 = (
        (boxes1[:, 3] - boxes1[:, 0])
        * (boxes1[:, 4] - boxes1[:, 1])
        * (boxes1[:, 5] - boxes1[:, 2])
    )
    vol2 = (
        (boxes2[:, 3] - boxes2[:, 0])
        * (boxes2[:, 4] - boxes2[:, 1])
        * (boxes2[:, 5] - boxes2[:, 2])
    )

    # Compute intersection
    lt = torch.max(boxes1[:, None, :3], boxes2[:, :3])  # (N, M, 3)
    rb = torch.min(boxes1[:, None, 3:], boxes2[:, 3:])  # (N, M, 3)

    whd = (rb - lt).clamp(min=0)  # (N, M, 3)
    inter = whd[:, :, 0] * whd[:, :, 1] * whd[:, :, 2]  # (N, M)

    # Compute union
    union = vol1[:, None] + vol2[None, :] - inter

    # Compute IoU
    iou = inter / union.clamp(min=1e-6)

    return iou


def _xywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert xywh to xyxy format."""
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)


def _cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert cxcywh to xyxy format."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _xyzwhd_to_xyzxyz(boxes: Tensor) -> Tensor:
    """Convert xyzwhd to xyzxyz format."""
    x, y, z, w, h, d = boxes.unbind(-1)
    return torch.stack([x, y, z, x + w, y + h, z + d], dim=-1)


def _cxcyczwhd_to_xyzxyz(boxes: Tensor) -> Tensor:
    """Convert cxcyczwhd to xyzxyz format."""
    cx, cy, cz, w, h, d = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    z1 = cz - d / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    z2 = cz + d / 2
    return torch.stack([x1, y1, z1, x2, y2, z2], dim=-1)


def compute_ap(
    recalls: np.ndarray,
    precisions: np.ndarray,
    use_11_point: bool = False,
) -> float:
    """
    Compute Average Precision from precision-recall curve.

    Parameters
    ----------
    recalls : np.ndarray
        Recall values
    precisions : np.ndarray
        Precision values
    use_11_point : bool
        If True, use 11-point interpolation

    Returns
    -------
    float
        Average Precision
    """
    if use_11_point:
        # 11-point interpolation
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0
        return ap
    else:
        # Area under curve
        # Add sentinel values
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))

        # Compute precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # Find indices where recall changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # Sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return float(ap)


def mean_average_precision(
    pred_boxes: List[ArrayLike],
    pred_scores: List[ArrayLike],
    pred_labels: List[ArrayLike],
    target_boxes: List[ArrayLike],
    target_labels: List[ArrayLike],
    iou_thresholds: Optional[np.ndarray] = None,
    class_aware: bool = True,
    use_3d: bool = False,
) -> Dict[str, float]:
    """
    Compute mean Average Precision (mAP).

    Parameters
    ----------
    pred_boxes : List[ArrayLike]
        Predicted boxes per image, each shape (N_i, 4) or (N_i, 6) for 3D
    pred_scores : List[ArrayLike]
        Prediction scores per image, each shape (N_i,)
    pred_labels : List[ArrayLike]
        Prediction labels per image, each shape (N_i,)
    target_boxes : List[ArrayLike]
        Ground truth boxes per image, each shape (M_i, 4) or (M_i, 6) for 3D
    target_labels : List[ArrayLike]
        Ground truth labels per image, each shape (M_i,)
    iou_thresholds : np.ndarray, optional
        IoU thresholds. If None, uses [0.50:0.95:0.05] for mAP@[.50:.95]
    class_aware : bool
        If True, compute per-class AP and average (class-aware mAP)
    use_3d : bool
        If True, use 3D box IoU

    Returns
    -------
    Dict[str, float]
        Dictionary with mAP scores at different IoU thresholds
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.50, 1.0, 0.05)

    # Get unique classes
    all_labels = []
    for labels in pred_labels + target_labels:
        all_labels.extend(as_tensor(labels).cpu().numpy().tolist())
    unique_classes = sorted(set(all_labels))

    results = {}

    for iou_thresh in iou_thresholds:
        aps = []

        for class_id in unique_classes:
            # Collect predictions and targets for this class
            class_pred_boxes = []
            class_pred_scores = []
            class_target_boxes = []

            for img_idx in range(len(pred_boxes)):
                pred_labels_img = as_tensor(pred_labels[img_idx]).cpu().numpy()
                target_labels_img = as_tensor(target_labels[img_idx]).cpu().numpy()

                # Filter by class
                pred_mask = pred_labels_img == class_id
                target_mask = target_labels_img == class_id

                if np.sum(pred_mask) > 0:
                    class_pred_boxes.append(as_tensor(pred_boxes[img_idx])[pred_mask])
                    class_pred_scores.append(as_tensor(pred_scores[img_idx])[pred_mask])

                if np.sum(target_mask) > 0:
                    class_target_boxes.append(as_tensor(target_boxes[img_idx])[target_mask])

            if len(class_target_boxes) == 0:
                continue

            # Sort predictions by score (descending)
            if class_pred_boxes:
                all_pred_boxes = torch.cat(class_pred_boxes, dim=0)
                all_pred_scores = torch.cat(class_pred_scores, dim=0)
            else:
                all_pred_boxes = torch.empty(0, 4 if not use_3d else 6)
                all_pred_scores = torch.empty(0)
            all_target_boxes = torch.cat(class_target_boxes, dim=0)

            if len(all_pred_boxes) == 0:
                ap = 0.0
            else:
                # Sort by score
                sorted_indices = torch.argsort(all_pred_scores, descending=True)
                all_pred_boxes = all_pred_boxes[sorted_indices]
                all_pred_scores = all_pred_scores[sorted_indices]

                # Compute IoU
                if use_3d:
                    iou_matrix = box_iou_3d(all_pred_boxes, all_target_boxes)
                else:
                    iou_matrix = box_iou_2d(all_pred_boxes, all_target_boxes)

                # Match predictions to targets
                matched_targets = set()
                tp = np.zeros(len(all_pred_boxes))
                fp = np.zeros(len(all_pred_boxes))

                for i in range(len(all_pred_boxes)):
                    if len(all_target_boxes) == 0:
                        fp[i] = 1
                        continue

                    # Find best matching target
                    ious = iou_matrix[i].cpu().numpy()
                    best_iou_idx = np.argmax(ious)
                    best_iou = ious[best_iou_idx]

                    if best_iou >= iou_thresh and best_iou_idx not in matched_targets:
                        tp[i] = 1
                        matched_targets.add(best_iou_idx)
                    else:
                        fp[i] = 1

                # Compute precision and recall
                tp_cumsum = np.cumsum(tp)
                fp_cumsum = np.cumsum(fp)
                recalls = tp_cumsum / len(all_target_boxes) if len(all_target_boxes) > 0 else np.zeros_like(tp_cumsum)
                precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

                # Compute AP
                ap = compute_ap(recalls, precisions)

            aps.append(ap)

        if class_aware and len(aps) > 0:
            map_score = np.mean(aps)
        else:
            map_score = aps[0] if len(aps) > 0 else 0.0

        results[f"mAP@{iou_thresh:.2f}"] = float(map_score)

    # Compute mAP@[.50:.95]
    if len(iou_thresholds) > 1:
        results["mAP@[.50:.95]"] = float(np.mean([results[f"mAP@{t:.2f}"] for t in iou_thresholds]))

    return results


def froc(
    pred_boxes: List[ArrayLike],
    pred_scores: List[ArrayLike],
    pred_labels: List[ArrayLike],
    target_boxes: List[ArrayLike],
    target_labels: List[ArrayLike],
    iou_threshold: float = 0.5,
    use_3d: bool = False,
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Compute Free-Response ROC (FROC) curve.

    FROC plots sensitivity vs. average number of false positives per image.

    Parameters
    ----------
    pred_boxes : List[ArrayLike]
        Predicted boxes per image
    pred_scores : List[ArrayLike]
        Prediction scores per image
    pred_labels : List[ArrayLike]
        Prediction labels per image
    target_boxes : List[ArrayLike]
        Ground truth boxes per image
    target_labels : List[ArrayLike]
        Ground truth labels per image
    iou_threshold : float
        IoU threshold for matching
    use_3d : bool
        If True, use 3D box IoU

    Returns
    -------
    Dict[str, Union[np.ndarray, float]]
        Dictionary with FROC curve data and average FPs
    """
    n_images = len(pred_boxes)

    # Collect all predictions and targets, tracking image indices
    all_pred_boxes_list = []
    all_pred_scores_list = []
    all_target_boxes = []
    image_indices = []  # Track which image each prediction belongs to

    for img_idx in range(n_images):
        pred_boxes_img = as_tensor(pred_boxes[img_idx])
        pred_scores_img = as_tensor(pred_scores[img_idx])
        n_preds = len(pred_boxes_img)

        all_pred_boxes_list.append(pred_boxes_img)
        all_pred_scores_list.append(pred_scores_img)
        all_target_boxes.append(as_tensor(target_boxes[img_idx]))
        image_indices.extend([img_idx] * n_preds)

    # Sort all predictions by score
    if all_pred_boxes_list:
        all_pred_boxes_cat = torch.cat(all_pred_boxes_list, dim=0)
        all_pred_scores_cat = torch.cat(all_pred_scores_list, dim=0)
    else:
        all_pred_boxes_cat = torch.empty(0, 4 if not use_3d else 6)
        all_pred_scores_cat = torch.empty(0)

    if len(all_pred_boxes_cat) == 0:
        return {
            "sensitivity": np.array([0.0]),
            "avg_fps_per_image": np.array([0.0]),
            "thresholds": np.array([0.0]),
        }

    sorted_indices = torch.argsort(all_pred_scores_cat, descending=True)
    all_pred_boxes_sorted = all_pred_boxes_cat[sorted_indices]
    all_pred_scores_sorted = all_pred_scores_cat[sorted_indices]
    image_indices_sorted = [image_indices[i] for i in sorted_indices.cpu().numpy()]

    # Compute total number of targets
    total_targets = sum(len(boxes) for boxes in all_target_boxes)

    # Compute FROC curve
    sensitivities = []
    avg_fps = []
    thresholds = []

    matched_targets_per_image = [set() for _ in range(n_images)]

    for i in range(len(all_pred_boxes_sorted)):
        # Get image index for this prediction
        img_idx = image_indices_sorted[i]

        pred_box = all_pred_boxes_sorted[i]
        pred_score = all_pred_scores_sorted[i].item()

        # Compute IoU with targets in this image
        if len(all_target_boxes[img_idx]) > 0:
            if use_3d:
                ious = box_iou_3d(pred_box.unsqueeze(0), all_target_boxes[img_idx])
            else:
                ious = box_iou_2d(pred_box.unsqueeze(0), all_target_boxes[img_idx])

            best_iou_idx = torch.argmax(ious).item()
            best_iou = ious[0, best_iou_idx].item()

            if best_iou >= iou_threshold and best_iou_idx not in matched_targets_per_image[img_idx]:
                matched_targets_per_image[img_idx].add(best_iou_idx)

        # Compute current metrics
        total_tp = sum(len(matched) for matched in matched_targets_per_image)
        sensitivity = total_tp / total_targets if total_targets > 0 else 0.0

        total_fp = (i + 1) - total_tp
        avg_fp = total_fp / n_images

        sensitivities.append(sensitivity)
        avg_fps.append(avg_fp)
        thresholds.append(pred_score)

    return {
        "sensitivity": np.array(sensitivities),
        "avg_fps_per_image": np.array(avg_fps),
        "thresholds": np.array(thresholds),
    }


def average_recall(
    pred_boxes: List[ArrayLike],
    pred_scores: List[ArrayLike],
    target_boxes: List[ArrayLike],
    max_detections: int = 100,
    iou_thresholds: Optional[np.ndarray] = None,
    use_3d: bool = False,
) -> float:
    """
    Compute Average Recall (AR).

    Parameters
    ----------
    pred_boxes : List[ArrayLike]
        Predicted boxes per image
    pred_scores : List[ArrayLike]
        Prediction scores per image
    target_boxes : List[ArrayLike]
        Ground truth boxes per image
    max_detections : int
        Maximum number of detections to consider
    iou_thresholds : np.ndarray, optional
        IoU thresholds. If None, uses [0.50:1.00:0.05]
    use_3d : bool
        If True, use 3D box IoU

    Returns
    -------
    float
        Average Recall
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.50, 1.0, 0.05)

    recalls = []

    for iou_thresh in iou_thresholds:
        total_tp = 0
        total_targets = 0

        for img_idx in range(len(pred_boxes)):
            pred_boxes_img = as_tensor(pred_boxes[img_idx])
            pred_scores_img = as_tensor(pred_scores[img_idx])
            target_boxes_img = as_tensor(target_boxes[img_idx])

            total_targets += len(target_boxes_img)

            if len(pred_boxes_img) == 0:
                continue

            # Sort by score and take top max_detections
            sorted_indices = torch.argsort(pred_scores_img, descending=True)[:max_detections]
            pred_boxes_sorted = pred_boxes_img[sorted_indices]

            # Compute IoU
            if len(target_boxes_img) > 0:
                if use_3d:
                    iou_matrix = box_iou_3d(pred_boxes_sorted, target_boxes_img)
                else:
                    iou_matrix = box_iou_2d(pred_boxes_sorted, target_boxes_img)

                # Match predictions to targets
                matched_targets = set()
                for i in range(len(pred_boxes_sorted)):
                    ious = iou_matrix[i].cpu().numpy()
                    best_iou_idx = np.argmax(ious)
                    best_iou = ious[best_iou_idx]

                    if best_iou >= iou_thresh and best_iou_idx not in matched_targets:
                        total_tp += 1
                        matched_targets.add(best_iou_idx)

        recall = total_tp / total_targets if total_targets > 0 else 0.0
        recalls.append(recall)

    return float(np.mean(recalls))


def hungarian_matching(
    cost_matrix: ArrayLike,
    maximize: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match instances using Hungarian algorithm.

    Parameters
    ----------
    cost_matrix : ArrayLike
        Cost matrix, shape (N, M)
    maximize : bool
        If True, maximize instead of minimize

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Row indices and column indices of matches
    """
    cost_matrix = as_tensor(cost_matrix).cpu().numpy()

    if maximize:
        # Convert to minimization problem
        cost_matrix = -cost_matrix + cost_matrix.max()

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    return row_indices, col_indices


def instance_segmentation_matching(
    pred_masks: List[ArrayLike],
    pred_scores: List[ArrayLike],
    target_masks: List[ArrayLike],
    iou_threshold: float = 0.5,
) -> Dict[str, Union[List, float]]:
    """
    Match instance segmentation masks using Hungarian algorithm.

    Parameters
    ----------
    pred_masks : List[ArrayLike]
        Predicted masks per image, each shape (N_i, H, W) or (N_i, D, H, W)
    pred_scores : List[ArrayLike]
        Prediction scores per image
    target_masks : List[ArrayLike]
        Ground truth masks per image, each shape (M_i, H, W) or (M_i, D, H, W)
    iou_threshold : float
        IoU threshold for matching

    Returns
    -------
    Dict[str, Union[List, float]]
        Matching results and metrics
    """
    all_matches = []
    all_tp = 0
    all_fp = 0
    all_fn = 0

    for img_idx in range(len(pred_masks)):
        pred_masks_img = as_tensor(pred_masks[img_idx])
        pred_scores_img = as_tensor(pred_scores[img_idx])
        target_masks_img = as_tensor(target_masks[img_idx])

        n_pred = len(pred_masks_img)
        n_target = len(target_masks_img)

        if n_pred == 0:
            all_fn += n_target
            continue

        if n_target == 0:
            all_fp += n_pred
            continue

        # Compute IoU matrix between all pred and target masks
        iou_matrix = np.zeros((n_pred, n_target))
        for i in range(n_pred):
            for j in range(n_target):
                pred_mask = pred_masks_img[i] > 0.5
                target_mask = target_masks_img[j] > 0.5

                intersection = (pred_mask & target_mask).sum().item()
                union = (pred_mask | target_mask).sum().item()

                if union > 0:
                    iou_matrix[i, j] = intersection / union

        # Use Hungarian algorithm to find optimal matching
        # Cost = 1 - IoU (to minimize)
        cost_matrix = 1.0 - iou_matrix
        row_indices, col_indices = hungarian_matching(cost_matrix, maximize=False)

        # Filter matches by IoU threshold
        matches = []
        matched_targets = set()
        for i, j in zip(row_indices, col_indices):
            if iou_matrix[i, j] >= iou_threshold:
                matches.append((i, j))
                matched_targets.add(j)

        all_matches.append(matches)
        all_tp += len(matches)
        all_fp += n_pred - len(matches)
        all_fn += n_target - len(matched_targets)

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0

    return {
        "matches": all_matches,
        "precision": float(precision),
        "recall": float(recall),
        "tp": all_tp,
        "fp": all_fp,
        "fn": all_fn,
    }


def compute_detection_metrics(
    pred_boxes: ArrayLike,
    target_boxes: ArrayLike,
    iou_thresholds: Optional[List[float]] = None,
    use_3d: bool = False,
    include_froc: bool = False,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compute comprehensive detection metrics.

    This is a high-level function that computes multiple detection metrics
    from predicted and ground truth bounding boxes.

    Parameters
    ----------
    pred_boxes : ArrayLike
        Predicted boxes with format [x1, y1, (z1,) x2, y2, (z2,) score, class_id]
        Shape: (N, 8) for 3D or (N, 6) for 2D
    target_boxes : ArrayLike
        Ground truth boxes with format [x1, y1, (z1,) x2, y2, (z2,) 1.0, class_id]
        Shape: (M, 8) for 3D or (M, 6) for 2D
    iou_thresholds : List[float], optional
        IoU thresholds for mAP computation. If None, uses [0.5, 0.75]
    use_3d : bool
        If True, use 3D box IoU (boxes have 6 coordinates instead of 4)
    include_froc : bool
        If True, include FROC curve data in results

    Returns
    -------
    Dict[str, Union[float, np.ndarray]]
        Dictionary containing:
        - iou_mean: Mean IoU between matched boxes
        - mAP@{threshold}: Mean Average Precision at each threshold
        - mAP@[.50:.95]: Mean AP across all thresholds (if multiple thresholds)
        - precision: Overall precision
        - recall: Overall recall
        - froc_* (if include_froc): FROC curve data

    Example
    -------
    >>> pred_boxes = torch.tensor([
    ...     [10, 10, 5, 30, 30, 15, 0.9, 0],  # 3D box with score 0.9, class 0
    ...     [50, 50, 10, 70, 70, 20, 0.8, 0],
    ... ])
    >>> gt_boxes = torch.tensor([
    ...     [12, 12, 6, 32, 32, 16, 1.0, 0],
    ... ])
    >>> results = compute_detection_metrics(pred_boxes, gt_boxes, use_3d=True)
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75]

    pred_boxes = as_tensor(pred_boxes).float()
    target_boxes = as_tensor(target_boxes).float()

    results = {}

    # Handle empty inputs
    if len(pred_boxes) == 0 and len(target_boxes) == 0:
        results["iou_mean"] = 1.0
        for thresh in iou_thresholds:
            results[f"mAP@{thresh:.2f}"] = 1.0
        results["precision"] = 1.0
        results["recall"] = 1.0
        return results

    if len(pred_boxes) == 0:
        results["iou_mean"] = 0.0
        for thresh in iou_thresholds:
            results[f"mAP@{thresh:.2f}"] = 0.0
        results["precision"] = 0.0
        results["recall"] = 0.0
        return results

    if len(target_boxes) == 0:
        results["iou_mean"] = 0.0
        for thresh in iou_thresholds:
            results[f"mAP@{thresh:.2f}"] = 0.0
        results["precision"] = 0.0
        results["recall"] = 0.0
        return results

    # Determine box format based on shape
    box_dim = 6 if use_3d else 4
    
    # Extract box coordinates, scores, and class IDs
    pred_coords = pred_boxes[:, :box_dim]
    pred_scores = pred_boxes[:, box_dim]
    pred_classes = pred_boxes[:, box_dim + 1].long()

    target_coords = target_boxes[:, :box_dim]
    target_classes = target_boxes[:, box_dim + 1].long()

    # Compute IoU matrix
    if use_3d:
        iou_matrix = box_iou_3d(pred_coords, target_coords)
    else:
        iou_matrix = box_iou_2d(pred_coords, target_coords)

    # Compute mean IoU of best matches
    if iou_matrix.numel() > 0:
        max_ious_per_pred = iou_matrix.max(dim=1).values
        results["iou_mean"] = float(max_ious_per_pred.mean().item())
    else:
        results["iou_mean"] = 0.0

    # Prepare data for mAP computation (wrap in lists as mAP expects per-image lists)
    pred_boxes_list = [pred_coords]
    pred_scores_list = [pred_scores]
    pred_labels_list = [pred_classes]
    target_boxes_list = [target_coords]
    target_labels_list = [target_classes]

    # Compute mAP at each threshold
    map_results = mean_average_precision(
        pred_boxes_list,
        pred_scores_list,
        pred_labels_list,
        target_boxes_list,
        target_labels_list,
        iou_thresholds=np.array(iou_thresholds),
        class_aware=True,
        use_3d=use_3d,
    )
    results.update(map_results)

    # Compute precision and recall at default IoU threshold (0.5)
    default_thresh = 0.5
    matched_targets = set()
    tp = 0
    fp = 0

    # Sort predictions by score
    sorted_indices = torch.argsort(pred_scores, descending=True)
    
    for idx in sorted_indices:
        pred_class = pred_classes[idx].item()
        
        # Find best matching target of same class
        best_iou = 0.0
        best_target_idx = -1
        
        for t_idx in range(len(target_boxes)):
            if t_idx in matched_targets:
                continue
            if target_classes[t_idx].item() != pred_class:
                continue
            
            iou = iou_matrix[idx, t_idx].item()
            if iou > best_iou:
                best_iou = iou
                best_target_idx = t_idx
        
        if best_iou >= default_thresh and best_target_idx >= 0:
            tp += 1
            matched_targets.add(best_target_idx)
        else:
            fp += 1

    fn = len(target_boxes) - len(matched_targets)
    
    results["precision"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    results["recall"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # Include FROC if requested
    if include_froc:
        froc_results = froc(
            pred_boxes_list,
            pred_scores_list,
            pred_labels_list,
            target_boxes_list,
            target_labels_list,
            iou_threshold=default_thresh,
            use_3d=use_3d,
        )
        results["froc_sensitivity"] = froc_results["sensitivity"]
        results["froc_avg_fps"] = froc_results["avg_fps_per_image"]
        results["froc_thresholds"] = froc_results["thresholds"]

    return results

