"""Metrics for medical imaging evaluation."""

from medeval.metrics.classification import (
    accuracy,
    adaptive_expected_calibration_error,
    auprc,
    auroc,
    balanced_accuracy,
    brier_score_classification,
    cohen_kappa,
    compute_classification_metrics,
    decision_curve,
    expected_calibration_error,
    f1_score_metric,
    group_by_patient,
    mcc,
    reliability_diagram,
    sensitivity,
    specificity,
    threshold_adaptive_calibration_error,
    youden_threshold,
)

from medeval.metrics.segmentation import (
    average_symmetric_surface_distance,
    brier_score,
    compute_segmentation_metrics,
    dice_score,
    hausdorff_distance,
    hausdorff_distance_95,
    jaccard_index,
    precision_score,
    recall_score,
    soft_dice_score,
    surface_dice,
    threshold_sweep,
    volumetric_similarity,
)

__all__ = [
    # Segmentation metrics
    "dice_score",
    "jaccard_index",
    "precision_score",
    "recall_score",
    "volumetric_similarity",
    "hausdorff_distance",
    "hausdorff_distance_95",
    "average_symmetric_surface_distance",
    "surface_dice",
    "soft_dice_score",
    "brier_score",
    "threshold_sweep",
    "compute_segmentation_metrics",
    # Classification metrics
    "auroc",
    "auprc",
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "f1_score_metric",
    "mcc",
    "cohen_kappa",
    # Calibration metrics
    "expected_calibration_error",
    "adaptive_expected_calibration_error",
    "threshold_adaptive_calibration_error",
    "brier_score_classification",
    "youden_threshold",
    "reliability_diagram",
    "decision_curve",
    # Utility functions
    "group_by_patient",
    "compute_classification_metrics",
]

