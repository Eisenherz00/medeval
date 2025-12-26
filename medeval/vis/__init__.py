"""Visualization utilities for medical imaging evaluation.

This module provides plotting functions for:
- ROC/PR/FROC curves
- Calibration diagrams
- Segmentation overlays
- Error histograms and distributions
"""

from medeval.vis.curves import (
    plot_roc_curve,
    plot_pr_curve,
    plot_froc_curve,
    plot_multiple_roc_curves,
    plot_multiple_pr_curves,
    plot_roc_from_predictions,
    plot_pr_from_predictions,
)

from medeval.vis.calibration import (
    plot_reliability_diagram,
    plot_reliability_from_predictions,
    plot_calibration_comparison,
    plot_decision_curve,
    plot_confidence_histogram,
)

from medeval.vis.overlays import (
    plot_segmentation_overlay,
    plot_prediction_comparison,
    plot_3d_slices,
)

from medeval.vis.histograms import (
    plot_error_histogram,
    plot_metric_distribution,
    plot_per_class_metrics,
    plot_stratified_results,
)

__all__ = [
    # Curves
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_froc_curve",
    "plot_multiple_roc_curves",
    "plot_multiple_pr_curves",
    "plot_roc_from_predictions",
    "plot_pr_from_predictions",
    # Calibration
    "plot_reliability_diagram",
    "plot_reliability_from_predictions",
    "plot_calibration_comparison",
    "plot_decision_curve",
    "plot_confidence_histogram",
    # Overlays
    "plot_segmentation_overlay",
    "plot_prediction_comparison",
    "plot_3d_slices",
    # Histograms
    "plot_error_histogram",
    "plot_metric_distribution",
    "plot_per_class_metrics",
    "plot_stratified_results",
]
