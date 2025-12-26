"""Classification and calibration metrics for medical imaging evaluation.

This module provides classification metrics (AUROC, AUPRC, accuracy, etc.),
calibration metrics (ECE, AECE, TACE), and decision curve analysis.
"""

from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from medeval.core.aggregate import bootstrap_ci
from medeval.core.typing import ArrayLike, Device, Tensor, as_tensor
from medeval.core.utils import ReductionType, reduce_metrics


try:
    from scipy.stats import norm
except ImportError:
    norm = None


# Helper: trapezoidal integration compatible with NumPy 1.x and 2.x
def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration compatible with NumPy 1.x and 2.x."""
    # NumPy 2.0 removed np.trapz; replacement is np.trapezoid
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _delong_auc_variance(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute variance of AUC using DeLong's method.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth labels
    y_scores : np.ndarray
        Prediction scores

    Returns
    -------
    float
        Variance of AUC
    """
    # Sort by scores
    order = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[order]
    y_scores_sorted = y_scores[order]

    n_pos = np.sum(y_true_sorted == 1)
    n_neg = len(y_true_sorted) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.0

    # Compute V10 and V01 (DeLong et al., 1988)
    pos_scores = y_scores_sorted[y_true_sorted == 1]
    neg_scores = y_scores_sorted[y_true_sorted == 0]

    # V10: average of indicator functions for positive samples
    v10 = np.array([np.mean(neg_scores < pos_score) for pos_score in pos_scores])
    v01 = np.array([np.mean(pos_scores > neg_score) for neg_score in neg_scores])

    # Variance formula from DeLong et al.
    s10 = np.var(v10, ddof=1) / n_pos
    s01 = np.var(v01, ddof=1) / n_neg
    variance = s10 / n_pos + s01 / n_neg

    return variance


def auroc(
    pred: ArrayLike,
    target: ArrayLike,
    average: Literal["macro", "micro", "weighted"] = "macro",
    multi_class: Literal["ovr", "ovo"] = "ovr",
    compute_ci: bool = False,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: Optional[int] = None,
) -> Union[float, Tuple[float, float, float]]:
    """
    Compute Area Under ROC Curve (AUROC).

    Parameters
    ----------
    pred : ArrayLike
        Prediction scores/logits, shape (N,) for binary or (N, C) for multi-class
    target : ArrayLike
        Ground truth labels, shape (N,) for binary or (N, C) for multi-class
    average : str
        Averaging strategy: "macro", "micro", "weighted"
    multi_class : str
        Multi-class strategy: "ovr" (one-vs-rest) or "ovo" (one-vs-one)
    compute_ci : bool
        If True, compute confidence interval using DeLong's method
    confidence : float
        Confidence level for CI
    seed : int, optional
        Random seed for bootstrap (if compute_ci=True and multi-class)

    Returns
    -------
    float or Tuple[float, float, float]
        AUROC score, optionally with (score, lower, upper) CI
    """
    pred = as_tensor(pred).cpu().numpy()
    target = as_tensor(target).cpu().numpy()

    # Handle multi-class
    if pred.ndim > 1 and pred.shape[1] > 1:
        # Multi-class: convert to probabilities if logits
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.softmax(torch.from_numpy(pred), dim=1).numpy()

        # Convert target to class indices if one-hot
        if target.ndim > 1 and target.shape[1] > 1:
            target = np.argmax(target, axis=1)

        try:
            auroc_score = roc_auc_score(
                target, pred, average=average, multi_class=multi_class
            )
        except ValueError:
            # Fallback: compute per-class and average
            num_classes = pred.shape[1]
            class_scores = []
            for c in range(num_classes):
                y_true_c = (target == c).astype(int)
                if np.sum(y_true_c) > 0 and np.sum(1 - y_true_c) > 0:
                    try:
                        score = roc_auc_score(y_true_c, pred[:, c])
                        class_scores.append(score)
                    except ValueError:
                        pass
            auroc_score = np.mean(class_scores) if class_scores else 0.0
    else:
        # Binary case
        if pred.ndim > 1:
            pred = pred.squeeze(1) if pred.shape[1] == 1 else pred[:, 1]
        if target.ndim > 1:
            target = target.squeeze(1) if target.shape[1] == 1 else np.argmax(target, axis=1)

        # Ensure binary labels
        target = (target > 0.5).astype(int) if target.dtype == float else target.astype(int)

        try:
            auroc_score = roc_auc_score(target, pred)
        except ValueError:
            # Only one class present
            auroc_score = 0.0

    if compute_ci:
        # Binary: use DeLong's method when available; otherwise bootstrap resampling.
        if pred.ndim == 1 or pred.shape[1] == 1:
            if norm is None:
                rng = np.random.default_rng(seed)
                n = len(target)
                if n == 0:
                    return float(auroc_score), float("nan"), float("nan")
                boot = []
                for _ in range(n_bootstrap):
                    idx = rng.integers(0, n, size=n)
                    try:
                        boot.append(roc_auc_score(target[idx], pred[idx]))
                    except Exception:
                        boot.append(float("nan"))
                boot = np.asarray(boot, dtype=float)
                boot = boot[np.isfinite(boot)]
                if boot.size == 0:
                    return float(auroc_score), float("nan"), float("nan")
                alpha = 1.0 - confidence
                lower = float(np.percentile(boot, 100 * (alpha / 2)))
                upper = float(np.percentile(boot, 100 * (1 - alpha / 2)))
            else:
                variance = _delong_auc_variance(target, pred)
                std_error = np.sqrt(variance)
                z_critical = norm.ppf(1 - (1 - confidence) / 2)
                lower = float(max(0.0, auroc_score - z_critical * std_error))
                upper = float(min(1.0, auroc_score + z_critical * std_error))
        else:
            # Multi-class: bootstrap resampling over samples.
            rng = np.random.default_rng(seed)
            n = len(target)
            if n == 0:
                return float(auroc_score), float("nan"), float("nan")
            boot = []
            for _ in range(n_bootstrap):
                idx = rng.integers(0, n, size=n)
                try:
                    boot.append(
                        roc_auc_score(
                            target[idx],
                            pred[idx],
                            average=average,
                            multi_class=multi_class,
                        )
                    )
                except Exception:
                    boot.append(float("nan"))
            boot = np.asarray(boot, dtype=float)
            boot = boot[np.isfinite(boot)]
            if boot.size == 0:
                return float(auroc_score), float("nan"), float("nan")
            alpha = 1.0 - confidence
            lower = float(np.percentile(boot, 100 * (alpha / 2)))
            upper = float(np.percentile(boot, 100 * (1 - alpha / 2)))

        return float(auroc_score), float(lower), float(upper)

    return float(auroc_score)


def auprc(
    pred: ArrayLike,
    target: ArrayLike,
    average: Literal["macro", "micro"] = "macro",
    compute_ci: bool = False,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: Optional[int] = None,
) -> Union[float, Tuple[float, float, float]]:
    """
    Compute Area Under Precision-Recall Curve (AUPRC).

    Parameters
    ----------
    pred : ArrayLike
        Prediction scores/logits
    target : ArrayLike
        Ground truth labels
    average : str
        Averaging strategy: "macro" or "micro"
    compute_ci : bool
        If True, compute confidence interval using bootstrap
    confidence : float
        Confidence level
    n_bootstrap : int
        Number of bootstrap samples
    seed : int, optional
        Random seed

    Returns
    -------
    float or Tuple[float, float, float]
        AUPRC score, optionally with CI
    """
    pred = as_tensor(pred).cpu().numpy()
    target = as_tensor(target).cpu().numpy()

    # Handle multi-class
    if pred.ndim > 1 and pred.shape[1] > 1:
        # Convert to probabilities if logits
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.softmax(torch.from_numpy(pred), dim=1).numpy()

        # Convert target to class indices if one-hot
        if target.ndim > 1 and target.shape[1] > 1:
            target = np.argmax(target, axis=1)

        # Compute per-class AUPRC
        num_classes = pred.shape[1]
        class_scores = []
        for c in range(num_classes):
            y_true_c = (target == c).astype(int)
            if np.sum(y_true_c) > 0:
                precision, recall, _ = precision_recall_curve(y_true_c, pred[:, c])
                # Compute AUC using trapezoidal rule
                auprc_c = _trapz(precision, recall)
                class_scores.append(auprc_c)

        if average == "macro":
            auprc_score = np.mean(class_scores) if class_scores else 0.0
        else:  # micro
            # Micro-averaged: combine all classes
            y_true_binary = np.zeros((len(target), num_classes))
            for c in range(num_classes):
                y_true_binary[:, c] = (target == c).astype(int)
            # Use micro-averaged precision-recall
            precision, recall, _ = precision_recall_curve(
                y_true_binary.ravel(), pred.ravel()
            )
            auprc_score = _trapz(precision, recall)
    else:
        # Binary case
        if pred.ndim > 1:
            pred = pred.squeeze(1) if pred.shape[1] == 1 else pred[:, 1]
        if target.ndim > 1:
            target = target.squeeze(1) if target.shape[1] == 1 else np.argmax(target, axis=1)

        target = (target > 0.5).astype(int) if target.dtype == float else target.astype(int)

        if np.sum(target) > 0:
            precision, recall, _ = precision_recall_curve(target, pred)
            # Sort by recall ascending for proper trapz integration
            sorted_idx = np.argsort(recall)
            auprc_score = _trapz(precision[sorted_idx], recall[sorted_idx])
        else:
            auprc_score = 0.0

    if compute_ci:
        # Bootstrap resampling over samples (not over a single scalar).
        rng = np.random.default_rng(seed)
        n = len(target)
        if n == 0:
            return float(auprc_score), float("nan"), float("nan")

        boot = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            try:
                boot.append(auprc(pred[idx], target[idx], average=average, compute_ci=False))
            except Exception:
                boot.append(float("nan"))

        boot = np.asarray(boot, dtype=float)
        boot = boot[np.isfinite(boot)]
        if boot.size == 0:
            return float(auprc_score), float("nan"), float("nan")

        alpha = 1.0 - confidence
        lower = float(np.percentile(boot, 100 * (alpha / 2)))
        upper = float(np.percentile(boot, 100 * (1 - alpha / 2)))
        return float(auprc_score), float(lower), float(upper)

    return float(auprc_score)


def accuracy(
    pred: ArrayLike,
    target: ArrayLike,
    threshold: float = 0.5,
    reduction: ReductionType = "mean-case",
) -> Tensor:
    """
    Compute accuracy.

    Parameters
    ----------
    pred : ArrayLike
        Predictions (logits or probabilities)
    target : ArrayLike
        Ground truth labels
    threshold : float
        Threshold for binary classification
    reduction : ReductionType
        Reduction strategy

    Returns
    -------
    Tensor
        Accuracy scores
    """
    pred = as_tensor(pred)
    target = as_tensor(target)

    # Convert to class predictions
    if pred.dim() > 1 and pred.shape[-1] > 1:
        pred_classes = pred.argmax(dim=-1)
    else:
        pred_classes = (pred > threshold).long().flatten()

    if target.dim() > 1 and target.shape[-1] > 1:
        target_classes = target.argmax(dim=-1)
    else:
        target_classes = target.long().flatten()

    # Compute accuracy
    correct = (pred_classes == target_classes).float()
    
    # For flat 1D input, return overall accuracy
    if correct.dim() == 1:
        accuracy_score = correct.mean()
        if reduction == "none":
            return correct  # Per-sample correctness
        return accuracy_score.unsqueeze(0)
    
    # For batched input with spatial dims, compute per-sample accuracy
    accuracy_scores = correct.mean(dim=tuple(range(1, correct.dim())))
    return reduce_metrics(accuracy_scores.unsqueeze(1), reduction=reduction)


def balanced_accuracy(
    pred: ArrayLike,
    target: ArrayLike,
    threshold: float = 0.5,
    reduction: ReductionType = "mean-case",
) -> Tensor:
    """
    Compute balanced accuracy.

    Parameters
    ----------
    pred : ArrayLike
        Predictions
    target : ArrayLike
        Ground truth labels
    threshold : float
        Threshold for binary classification
    reduction : ReductionType
        Reduction strategy

    Returns
    -------
    Tensor
        Balanced accuracy scores
    """
    pred = as_tensor(pred).cpu().numpy()
    target = as_tensor(target).cpu().numpy()

    # Handle batch dimension
    if pred.ndim > 1 and pred.shape[0] > 1:
        scores = []
        for i in range(pred.shape[0]):
            pred_i = pred[i]
            target_i = target[i]

            # Convert to class predictions
            if pred_i.ndim > 0 and (pred_i.ndim > 1 or len(pred_i.shape) > 0):
                if pred_i.ndim > 1 and pred_i.shape[-1] > 1:
                    pred_classes = np.argmax(pred_i, axis=-1)
                else:
                    pred_classes = (pred_i > threshold).astype(int)
            else:
                pred_classes = (pred_i > threshold).astype(int) if pred_i.ndim == 0 else (pred_i > threshold).astype(int)

            if target_i.ndim > 1 and target_i.shape[-1] > 1:
                target_classes = np.argmax(target_i, axis=-1)
            else:
                target_classes = target_i.astype(int)

            score = balanced_accuracy_score(target_classes.flatten(), pred_classes.flatten())
            scores.append(score)

        scores_tensor = torch.tensor(scores, dtype=torch.float32)
    else:
        # Single sample
        pred_flat = pred.flatten()
        target_flat = target.flatten()

        if pred_flat.ndim == 0 or len(pred_flat.shape) == 0:
            pred_classes = (pred_flat > threshold).astype(int)
        else:
            if pred_flat.ndim > 1 or (hasattr(pred_flat, 'shape') and len(pred_flat.shape) > 1 and pred_flat.shape[-1] > 1):
                pred_classes = np.argmax(pred_flat, axis=-1)
            else:
                pred_classes = (pred_flat > threshold).astype(int)

        target_classes = target_flat.astype(int)
        score = balanced_accuracy_score(target_classes, pred_classes)
        scores_tensor = torch.tensor([score], dtype=torch.float32)

    return reduce_metrics(scores_tensor.unsqueeze(1), reduction=reduction)


def sensitivity(
    pred: ArrayLike,
    target: ArrayLike,
    threshold: float = 0.5,
    reduction: ReductionType = "mean-case",
) -> Tensor:
    """
    Compute sensitivity (recall, true positive rate).

    Parameters
    ----------
    pred : ArrayLike
        Predictions
    target : ArrayLike
        Ground truth labels
    threshold : float
        Threshold for binary classification
    reduction : ReductionType
        Reduction strategy

    Returns
    -------
    Tensor
        Sensitivity scores
    """
    pred = as_tensor(pred)
    target = as_tensor(target)

    # Flatten for simple binary classification
    pred_binary = (pred > threshold).float().flatten()
    target_binary = target.float().flatten()

    # Compute TP and FN over all samples
    tp = (pred_binary * target_binary).sum()
    fn = ((1 - pred_binary) * target_binary).sum()

    sensitivity_score = torch.where(
        (tp + fn) > 0, tp / (tp + fn), torch.tensor(0.0, device=pred.device)
    )

    if reduction == "none":
        return sensitivity_score.unsqueeze(0)
    return sensitivity_score.unsqueeze(0)


def specificity(
    pred: ArrayLike,
    target: ArrayLike,
    threshold: float = 0.5,
    reduction: ReductionType = "mean-case",
) -> Tensor:
    """
    Compute specificity (true negative rate).

    Parameters
    ----------
    pred : ArrayLike
        Predictions
    target : ArrayLike
        Ground truth labels
    threshold : float
        Threshold for binary classification
    reduction : ReductionType
        Reduction strategy

    Returns
    -------
    Tensor
        Specificity scores
    """
    pred = as_tensor(pred)
    target = as_tensor(target)

    # Flatten for simple binary classification
    pred_binary = (pred > threshold).float().flatten()
    target_binary = target.float().flatten()

    # Compute TN and FP over all samples
    tn = ((1 - pred_binary) * (1 - target_binary)).sum()
    fp = (pred_binary * (1 - target_binary)).sum()

    specificity_score = torch.where(
        (tn + fp) > 0, tn / (tn + fp), torch.tensor(0.0, device=pred.device)
    )

    if reduction == "none":
        return specificity_score.unsqueeze(0)
    return specificity_score.unsqueeze(0)


def f1_score_metric(
    pred: ArrayLike,
    target: ArrayLike,
    threshold: float = 0.5,
    average: Literal["macro", "micro", "weighted"] = "macro",
    reduction: ReductionType = "mean-case",
) -> Tensor:
    """
    Compute F1 score.

    Parameters
    ----------
    pred : ArrayLike
        Predictions
    target : ArrayLike
        Ground truth labels
    threshold : float
        Threshold for binary classification
    average : str
        Averaging strategy for multi-class
    reduction : ReductionType
        Reduction strategy

    Returns
    -------
    Tensor
        F1 scores
    """
    pred = as_tensor(pred).cpu().numpy()
    target = as_tensor(target).cpu().numpy()

    # Convert to class predictions
    if pred.ndim > 1 and pred.shape[-1] > 1:
        pred_classes = np.argmax(pred, axis=-1)
    else:
        pred_classes = (pred > threshold).astype(int)

    if target.ndim > 1 and target.shape[-1] > 1:
        target_classes = np.argmax(target, axis=-1)
    else:
        target_classes = target.astype(int)

    # Handle batch dimension
    if pred.ndim > 1 and pred.shape[0] > 1:
        scores = []
        for i in range(pred.shape[0]):
            score = f1_score(
                target_classes[i].flatten(),
                pred_classes[i].flatten(),
                average=average,
                zero_division=0,
            )
            scores.append(score)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
    else:
        score = f1_score(
            target_classes.flatten(), pred_classes.flatten(), average=average, zero_division=0
        )
        scores_tensor = torch.tensor([score], dtype=torch.float32)

    return reduce_metrics(scores_tensor.unsqueeze(1), reduction=reduction)


def mcc(
    pred: ArrayLike,
    target: ArrayLike,
    threshold: float = 0.5,
    reduction: ReductionType = "mean-case",
) -> Tensor:
    """
    Compute Matthews Correlation Coefficient (MCC).

    Parameters
    ----------
    pred : ArrayLike
        Predictions
    target : ArrayLike
        Ground truth labels
    threshold : float
        Threshold for binary classification
    reduction : ReductionType
        Reduction strategy

    Returns
    -------
    Tensor
        MCC scores
    """
    pred = as_tensor(pred).cpu().numpy()
    target = as_tensor(target).cpu().numpy()

    # Convert to class predictions
    if pred.ndim > 1 and pred.shape[-1] > 1:
        pred_classes = np.argmax(pred, axis=-1)
    else:
        pred_classes = (pred > threshold).astype(int)

    if target.ndim > 1 and target.shape[-1] > 1:
        target_classes = np.argmax(target, axis=-1)
    else:
        target_classes = target.astype(int)

    # Handle batch dimension
    if pred.ndim > 1 and pred.shape[0] > 1:
        scores = []
        for i in range(pred.shape[0]):
            score = matthews_corrcoef(
                target_classes[i].flatten(), pred_classes[i].flatten()
            )
            scores.append(score)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
    else:
        score = matthews_corrcoef(target_classes.flatten(), pred_classes.flatten())
        scores_tensor = torch.tensor([score], dtype=torch.float32)

    return reduce_metrics(scores_tensor.unsqueeze(1), reduction=reduction)


def cohen_kappa(
    pred: ArrayLike,
    target: ArrayLike,
    threshold: float = 0.5,
    reduction: ReductionType = "mean-case",
) -> Tensor:
    """
    Compute Cohen's kappa coefficient.

    Parameters
    ----------
    pred : ArrayLike
        Predictions
    target : ArrayLike
        Ground truth labels
    threshold : float
        Threshold for binary classification
    reduction : ReductionType
        Reduction strategy

    Returns
    -------
    Tensor
        Cohen's kappa scores
    """
    pred = as_tensor(pred).cpu().numpy()
    target = as_tensor(target).cpu().numpy()

    # Convert to class predictions
    if pred.ndim > 1 and pred.shape[-1] > 1:
        pred_classes = np.argmax(pred, axis=-1)
    else:
        pred_classes = (pred > threshold).astype(int)

    if target.ndim > 1 and target.shape[-1] > 1:
        target_classes = np.argmax(target, axis=-1)
    else:
        target_classes = target.astype(int)

    # Handle batch dimension
    if pred.ndim > 1 and pred.shape[0] > 1:
        scores = []
        for i in range(pred.shape[0]):
            score = cohen_kappa_score(
                target_classes[i].flatten(), pred_classes[i].flatten()
            )
            scores.append(score)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
    else:
        score = cohen_kappa_score(target_classes.flatten(), pred_classes.flatten())
        scores_tensor = torch.tensor([score], dtype=torch.float32)

    return reduce_metrics(scores_tensor.unsqueeze(1), reduction=reduction)


def expected_calibration_error(
    pred: ArrayLike,
    target: ArrayLike,
    n_bins: int = 10,
    reduction: ReductionType = "mean-case",
) -> Tensor:
    """
    Compute Expected Calibration Error (ECE).

    ECE = sum(|acc(b) - conf(b)| * |b| / N) over bins b

    Parameters
    ----------
    pred : ArrayLike
        Prediction probabilities, shape (N,) or (N, C)
    target : ArrayLike
        Ground truth labels
    n_bins : int
        Number of bins for calibration
    reduction : ReductionType
        Reduction strategy

    Returns
    -------
    Tensor
        ECE scores
    """
    pred = as_tensor(pred).cpu().numpy()
    target = as_tensor(target).cpu().numpy()

    # Handle multi-class: use max probability
    if pred.ndim > 1 and pred.shape[1] > 1:
        # Convert to probabilities if logits
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.softmax(torch.from_numpy(pred), dim=1).numpy()
        pred_conf = np.max(pred, axis=1)
        pred_classes = np.argmax(pred, axis=1)

        if target.ndim > 1 and target.shape[1] > 1:
            target_classes = np.argmax(target, axis=1)
        else:
            target_classes = target.astype(int)
    else:
        # Binary case
        if pred.ndim > 1:
            pred = pred.squeeze(1) if pred.shape[1] == 1 else pred[:, 1]
        pred_conf = pred
        pred_classes = (pred > 0.5).astype(int)

        if target.ndim > 1:
            target_classes = np.argmax(target, axis=1) if target.shape[1] > 1 else target.squeeze(1)
        else:
            target_classes = (target > 0.5).astype(int) if target.dtype == float else target.astype(int)

    # Compute ECE
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (pred_conf > bin_lower) & (pred_conf <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            # Accuracy in this bin
            accuracy_in_bin = (pred_classes[in_bin] == target_classes[in_bin]).mean()
            # Average confidence in this bin
            avg_confidence_in_bin = pred_conf[in_bin].mean()
            # Add to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return reduce_metrics(torch.tensor([ece], dtype=torch.float32).unsqueeze(1), reduction=reduction)


def adaptive_expected_calibration_error(
    pred: ArrayLike,
    target: ArrayLike,
    n_bins: int = 10,
    reduction: ReductionType = "mean-case",
) -> Tensor:
    """
    Compute Adaptive Expected Calibration Error (AECE).

    Similar to ECE but uses adaptive binning (equal number of samples per bin).

    Parameters
    ----------
    pred : ArrayLike
        Prediction probabilities
    target : ArrayLike
        Ground truth labels
    n_bins : int
        Number of bins
    reduction : ReductionType
        Reduction strategy

    Returns
    -------
    Tensor
        AECE scores
    """
    pred = as_tensor(pred).cpu().numpy()
    target = as_tensor(target).cpu().numpy()

    # Handle multi-class
    if pred.ndim > 1 and pred.shape[1] > 1:
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.softmax(torch.from_numpy(pred), dim=1).numpy()
        pred_conf = np.max(pred, axis=1)
        pred_classes = np.argmax(pred, axis=1)

        if target.ndim > 1 and target.shape[1] > 1:
            target_classes = np.argmax(target, axis=1)
        else:
            target_classes = target.astype(int)
    else:
        if pred.ndim > 1:
            pred = pred.squeeze(1) if pred.shape[1] == 1 else pred[:, 1]
        pred_conf = pred
        pred_classes = (pred > 0.5).astype(int)

        if target.ndim > 1:
            target_classes = np.argmax(target, axis=1) if target.shape[1] > 1 else target.squeeze(1)
        else:
            target_classes = (target > 0.5).astype(int) if target.dtype == float else target.astype(int)

    # Adaptive binning: equal number of samples per bin
    sorted_indices = np.argsort(pred_conf)
    n_samples = len(pred_conf)
    samples_per_bin = n_samples // n_bins

    aece = 0.0
    for i in range(n_bins):
        start_idx = i * samples_per_bin
        end_idx = (i + 1) * samples_per_bin if i < n_bins - 1 else n_samples
        bin_indices = sorted_indices[start_idx:end_idx]

        if len(bin_indices) > 0:
            # Accuracy in this bin
            accuracy_in_bin = (pred_classes[bin_indices] == target_classes[bin_indices]).mean()
            # Average confidence in this bin
            avg_confidence_in_bin = pred_conf[bin_indices].mean()
            # Add to AECE
            prop_in_bin = len(bin_indices) / n_samples
            aece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return reduce_metrics(torch.tensor([aece], dtype=torch.float32).unsqueeze(1), reduction=reduction)


def threshold_adaptive_calibration_error(
    pred: ArrayLike,
    target: ArrayLike,
    n_bins: int = 10,
    threshold: float = 0.5,
    reduction: ReductionType = "mean-case",
) -> Tensor:
    """
    Compute Threshold-Adaptive Calibration Error (TACE).

    TACE focuses on calibration error near the decision threshold, which is
    most relevant for binary classification decisions. It uses bins that are
    denser near the threshold and sparser away from it.

    Parameters
    ----------
    pred : ArrayLike
        Prediction probabilities
    target : ArrayLike
        Ground truth labels
    n_bins : int
        Number of bins
    threshold : float
        Decision threshold (bins are centered around this)
    reduction : ReductionType
        Reduction strategy

    Returns
    -------
    Tensor
        TACE scores
    """
    pred = as_tensor(pred).cpu().numpy()
    target = as_tensor(target).cpu().numpy()

    # Handle multi-class
    if pred.ndim > 1 and pred.shape[1] > 1:
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.softmax(torch.from_numpy(pred), dim=1).numpy()
        pred_conf = np.max(pred, axis=1)
        pred_classes = np.argmax(pred, axis=1)

        if target.ndim > 1 and target.shape[1] > 1:
            target_classes = np.argmax(target, axis=1)
        else:
            target_classes = target.astype(int)
    else:
        if pred.ndim > 1:
            pred = pred.squeeze(1) if pred.shape[1] == 1 else pred[:, 1]
        pred_conf = pred
        pred_classes = (pred > threshold).astype(int)

        if target.ndim > 1:
            target_classes = np.argmax(target, axis=1) if target.shape[1] > 1 else target.squeeze(1)
        else:
            target_classes = (target > 0.5).astype(int) if target.dtype == float else target.astype(int)

    # Create threshold-adaptive bins: denser near threshold, sparser away
    # Use a sigmoid-like spacing to concentrate bins near threshold
    half_bins = n_bins // 2

    # Bins below threshold (0 to threshold)
    lower_bins = threshold * (1 - np.exp(-np.linspace(0, 3, half_bins + 1)))
    lower_bins = np.sort(lower_bins)

    # Bins above threshold (threshold to 1)
    upper_bins = threshold + (1 - threshold) * (1 - np.exp(-np.linspace(0, 3, half_bins + 1)[::-1]))
    upper_bins = np.sort(upper_bins)

    # Combine bins
    bin_boundaries = np.unique(np.concatenate([lower_bins, upper_bins]))
    bin_boundaries = np.clip(bin_boundaries, 0, 1)
    bin_boundaries = np.sort(bin_boundaries)

    # Compute TACE
    tace = 0.0
    total_weight = 0.0

    for i in range(len(bin_boundaries) - 1):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (pred_conf > bin_lower) & (pred_conf <= bin_upper)
        n_in_bin = np.sum(in_bin)

        if n_in_bin > 0:
            # Weight bins near threshold more heavily
            bin_center = (bin_lower + bin_upper) / 2
            distance_to_threshold = abs(bin_center - threshold)
            weight = np.exp(-2 * distance_to_threshold)  # Exponential decay

            accuracy_in_bin = (pred_classes[in_bin] == target_classes[in_bin]).mean()
            avg_confidence_in_bin = pred_conf[in_bin].mean()

            tace += weight * np.abs(avg_confidence_in_bin - accuracy_in_bin) * n_in_bin
            total_weight += weight * n_in_bin

    if total_weight > 0:
        tace = tace / total_weight
    else:
        tace = 0.0

    return reduce_metrics(torch.tensor([tace], dtype=torch.float32).unsqueeze(1), reduction=reduction)


def brier_score_classification(
    pred: ArrayLike,
    target: ArrayLike,
    reduction: ReductionType = "mean-case",
) -> Tensor:
    """
    Compute Brier score for classification.

    Brier = mean((p - y)^2) where p is probability and y is binary label.

    Parameters
    ----------
    pred : ArrayLike
        Prediction probabilities
    target : ArrayLike
        Ground truth labels (binary or one-hot)
    reduction : ReductionType
        Reduction strategy

    Returns
    -------
    Tensor
        Brier scores
    """
    pred = as_tensor(pred).float()
    target = as_tensor(target).float()

    # Handle multi-class: convert to one-hot if needed
    if target.dim() == 1 or (target.dim() > 1 and target.shape[1] == 1):
        # Binary or class indices
        if target.dim() > 1:
            target = target.squeeze(1)
        target_one_hot = F.one_hot(target.long(), num_classes=pred.shape[1] if pred.dim() > 1 and pred.shape[1] > 1 else 2)
    else:
        target_one_hot = target

    # Ensure pred is probabilities
    if pred.dim() > 1 and pred.shape[1] > 1:
        if pred.min() < 0 or pred.max() > 1:
            pred = F.softmax(pred, dim=1)
    else:
        pred = torch.clamp(pred, 0, 1)
        if target_one_hot.shape[-1] == 2:
            # Binary: use positive class probability
            pred = torch.stack([1 - pred, pred], dim=-1)

    # Compute Brier score
    squared_error = (pred - target_one_hot) ** 2
    brier = squared_error.mean(dim=tuple(range(1, squared_error.dim())))

    return reduce_metrics(brier.unsqueeze(1), reduction=reduction)


def youden_threshold(
    pred: ArrayLike,
    target: ArrayLike,
) -> float:
    """
    Compute optimal threshold using Youden's J statistic.

    J = sensitivity + specificity - 1

    Parameters
    ----------
    pred : ArrayLike
        Prediction scores
    target : ArrayLike
        Ground truth labels

    Returns
    -------
    float
        Optimal threshold
    """
    pred = as_tensor(pred).cpu().numpy()
    target = as_tensor(target).cpu().numpy()

    # Binary classification
    if pred.ndim > 1:
        pred = pred.squeeze(1) if pred.shape[1] == 1 else pred[:, 1]
    if target.ndim > 1:
        target = target.squeeze(1) if target.shape[1] == 1 else np.argmax(target, axis=1)

    target = (target > 0.5).astype(int) if target.dtype == float else target.astype(int)

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(target, pred)
    # Youden's J = TPR - FPR = sensitivity - (1 - specificity) = sensitivity + specificity - 1
    youden_scores = tpr - fpr
    optimal_idx = np.argmax(youden_scores)

    return float(thresholds[optimal_idx])


def reliability_diagram(
    pred: ArrayLike,
    target: ArrayLike,
    n_bins: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Compute reliability diagram data.

    Parameters
    ----------
    pred : ArrayLike
        Prediction probabilities
    target : ArrayLike
        Ground truth labels
    n_bins : int
        Number of bins

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with 'bin_centers', 'accuracies', 'confidences', 'counts'
    """
    pred = as_tensor(pred).cpu().numpy()
    target = as_tensor(target).cpu().numpy()

    # Handle multi-class
    if pred.ndim > 1 and pred.shape[1] > 1:
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.softmax(torch.from_numpy(pred), dim=1).numpy()
        pred_conf = np.max(pred, axis=1)
        pred_classes = np.argmax(pred, axis=1)

        if target.ndim > 1 and target.shape[1] > 1:
            target_classes = np.argmax(target, axis=1)
        else:
            target_classes = target.astype(int)
    else:
        if pred.ndim > 1:
            pred = pred.squeeze(1) if pred.shape[1] == 1 else pred[:, 1]
        pred_conf = pred
        pred_classes = (pred > 0.5).astype(int)

        if target.ndim > 1:
            target_classes = np.argmax(target, axis=1) if target.shape[1] > 1 else target.squeeze(1)
        else:
            target_classes = (target > 0.5).astype(int) if target.dtype == float else target.astype(int)

    # Compute bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

    accuracies = []
    confidences = []
    counts = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (pred_conf > bin_lower) & (pred_conf <= bin_upper)

        if np.sum(in_bin) > 0:
            accuracy = (pred_classes[in_bin] == target_classes[in_bin]).mean()
            confidence = pred_conf[in_bin].mean()
            count = np.sum(in_bin)
        else:
            accuracy = 0.0
            confidence = bin_centers[i]
            count = 0

        accuracies.append(accuracy)
        confidences.append(confidence)
        counts.append(count)

    return {
        "bin_centers": bin_centers,
        "accuracies": np.array(accuracies),
        "confidences": np.array(confidences),
        "counts": np.array(counts),
    }


def decision_curve(
    pred: ArrayLike,
    target: ArrayLike,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute decision curve (net benefit) for different thresholds.

    Net Benefit = (TP - FP * (pt / (1 - pt))) / N
    where pt is the threshold probability

    Parameters
    ----------
    pred : ArrayLike
        Prediction probabilities
    target : ArrayLike
        Ground truth labels
    thresholds : np.ndarray, optional
        Probability thresholds to evaluate. If None, uses [0.01, 0.02, ..., 0.99]

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with 'thresholds', 'net_benefit', 'treat_all', 'treat_none'
    """
    pred = as_tensor(pred).cpu().numpy()
    target = as_tensor(target).cpu().numpy()

    # Binary classification
    if pred.ndim > 1:
        pred = pred.squeeze(1) if pred.shape[1] == 1 else pred[:, 1]
    if target.ndim > 1:
        target = target.squeeze(1) if target.shape[1] == 1 else np.argmax(target, axis=1)

    target = (target > 0.5).astype(int) if target.dtype == float else target.astype(int)

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    n_total = len(target)
    n_positive = np.sum(target)

    net_benefits = []
    for pt in thresholds:
        # Predictions at this threshold
        pred_binary = (pred >= pt).astype(int)

        tp = np.sum((pred_binary == 1) & (target == 1))
        fp = np.sum((pred_binary == 1) & (target == 0))

        # Net benefit
        if pt > 0 and pt < 1:
            net_benefit = (tp - fp * (pt / (1 - pt))) / n_total
        else:
            net_benefit = 0.0

        net_benefits.append(net_benefit)

    # Treat all: assume all positive
    treat_all_benefit = (n_positive - (n_total - n_positive) * (thresholds / (1 - thresholds))) / n_total
    treat_all_benefit = np.clip(treat_all_benefit, 0, None)

    # Treat none: net benefit is 0
    treat_none_benefit = np.zeros_like(thresholds)

    return {
        "thresholds": thresholds,
        "net_benefit": np.array(net_benefits),
        "treat_all": treat_all_benefit,
        "treat_none": treat_none_benefit,
    }


def group_by_patient(
    pred: ArrayLike,
    target: ArrayLike,
    patient_ids: ArrayLike,
    aggregation: Literal["mean", "max", "min"] = "mean",
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Group predictions and targets by patient ID (e.g., study-level from slices).

    Parameters
    ----------
    pred : ArrayLike
        Predictions, shape (N, ...) where N can be slices
    target : ArrayLike
        Ground truth, same shape
    patient_ids : ArrayLike
        Patient/study IDs, shape (N,)
    aggregation : str
        How to aggregate per-patient: "mean", "max", "min"

    Returns
    -------
    Tuple[ArrayLike, ArrayLike]
        Aggregated predictions and targets per patient
    """
    pred = as_tensor(pred)
    target = as_tensor(target)
    patient_ids = as_tensor(patient_ids).cpu().numpy()

    unique_patients = np.unique(patient_ids)
    aggregated_preds: List[torch.Tensor] = []
    aggregated_targets: List[torch.Tensor] = []

    # Helper: aggregate discrete (integer/bool) targets robustly
    def _agg_discrete_target(x: torch.Tensor, how: Literal["mean", "max", "min"]) -> torch.Tensor:
        """Aggregate discrete targets per patient.

        - For integer class indices (shape (K,) or (K,1)): 
          - mean -> majority vote (mode)
          - max/min -> max/min
        - For one-hot encoded discrete targets (shape (K, C), integer/bool):
          - mean -> mean in float (keeps (C,) with class prevalence)
          - max/min -> max/min (keeps (C,))

        Returns a tensor with the same trailing shape as a single target entry.
        """
        # One-hot / multi-label style targets (last dim > 1)
        if x.dim() > 1 and x.shape[-1] > 1:
            if how == "mean":
                return x.float().mean(dim=0)
            if how == "max":
                return x.max(dim=0).values
            if how == "min":
                return x.min(dim=0).values
            raise ValueError(f"Unknown aggregation: {how}")

        # Class-index / binary labels
        if x.dim() > 1 and x.shape[-1] == 1:
            x = x.squeeze(-1)

        if how == "mean":
            # Majority vote (mode). Works for binary and multi-class integer labels.
            return torch.mode(x.long(), dim=0).values
        if how == "max":
            return x.max(dim=0).values
        if how == "min":
            return x.min(dim=0).values
        raise ValueError(f"Unknown aggregation: {how}")

    for patient_id in unique_patients:
        mask = patient_ids == patient_id
        patient_pred = pred[mask]
        patient_target = target[mask]

        # Aggregate predictions
        if aggregation == "mean":
            agg_pred = patient_pred.mean(dim=0)
        elif aggregation == "max":
            agg_pred = patient_pred.max(dim=0).values
        elif aggregation == "min":
            agg_pred = patient_pred.min(dim=0).values
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        # Aggregate targets safely (avoid .mean() on integer labels)
        if patient_target.is_floating_point() or patient_target.is_complex():
            if aggregation == "mean":
                agg_target = patient_target.mean(dim=0)
            elif aggregation == "max":
                agg_target = patient_target.max(dim=0).values
            elif aggregation == "min":
                agg_target = patient_target.min(dim=0).values
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
        else:
            agg_target = _agg_discrete_target(patient_target, aggregation)

        aggregated_preds.append(agg_pred)
        aggregated_targets.append(agg_target)

    return torch.stack(aggregated_preds), torch.stack(aggregated_targets)


def compute_classification_metrics(
    pred: ArrayLike,
    target: ArrayLike,
    compute_ci: bool = False,
    confidence: float = 0.95,
    include_calibration: bool = False,
    reduction: ReductionType = "mean-case",
) -> Dict[str, Union[float, Tensor, Tuple[float, float, float]]]:
    """
    Compute comprehensive classification metrics.

    Parameters
    ----------
    pred : ArrayLike
        Predictions (logits or probabilities)
    target : ArrayLike
        Ground truth labels
    compute_ci : bool
        If True, compute confidence intervals
    confidence : float
        Confidence level for CI
    include_calibration : bool
        If True, include calibration metrics
    reduction : ReductionType
        Reduction strategy

    Returns
    -------
    Dict[str, Union[float, Tensor, Tuple]]
        Dictionary of metric names to values
    """
    results = {}

    # Classification metrics
    results["auroc"] = auroc(pred, target, compute_ci=compute_ci, confidence=confidence)
    results["auprc"] = auprc(pred, target, compute_ci=compute_ci, confidence=confidence)
    results["accuracy"] = accuracy(pred, target, reduction=reduction)
    results["balanced_accuracy"] = balanced_accuracy(pred, target, reduction=reduction)
    results["sensitivity"] = sensitivity(pred, target, reduction=reduction)
    results["specificity"] = specificity(pred, target, reduction=reduction)
    results["f1"] = f1_score_metric(pred, target, reduction=reduction)
    results["mcc"] = mcc(pred, target, reduction=reduction)
    results["cohen_kappa"] = cohen_kappa(pred, target, reduction=reduction)

    # Calibration metrics
    if include_calibration:
        results["ece"] = expected_calibration_error(pred, target, reduction=reduction)
        results["aece"] = adaptive_expected_calibration_error(pred, target, reduction=reduction)
        results["tace"] = threshold_adaptive_calibration_error(pred, target, reduction=reduction)
        results["brier"] = brier_score_classification(pred, target, reduction=reduction)

    return results

