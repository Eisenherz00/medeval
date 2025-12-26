"""Curve plotting utilities for ROC, PR, and FROC curves."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None  # type: ignore
    Figure = None  # type: ignore

try:
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    roc_curve = None  # type: ignore
    precision_recall_curve = None  # type: ignore
    auc = None  # type: ignore

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from medeval.core.typing import ArrayLike, as_tensor


import warnings


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


def _validate_curve_input(x: np.ndarray, y: np.ndarray, x_name: str, y_name: str) -> None:
    """
    Validate that inputs look like pre-computed curve data, not raw labels/scores.
    
    Raises warnings if inputs appear to be raw y_true/y_score instead of curve points.
    """
    # Check for multi-dimensional input
    if x.ndim > 1 or y.ndim > 1:
        warnings.warn(
            f"Input arrays have shape {x_name}={x.shape}, {y_name}={y.shape}. "
            f"Expected 1D arrays of curve points. "
            f"If you have raw labels and scores, use plot_roc_from_predictions() or "
            f"plot_pr_from_predictions() instead.",
            UserWarning,
            stacklevel=3
        )
    
    # Check if x looks like binary labels (only 0s and 1s)
    unique_x = np.unique(x)
    if len(unique_x) == 2 and set(unique_x).issubset({0, 1, 0.0, 1.0}):
        warnings.warn(
            f"{x_name} contains only values {unique_x}, which looks like binary labels. "
            f"This function expects pre-computed curve points (e.g., fpr from sklearn.metrics.roc_curve). "
            f"If you have raw labels and scores, use plot_roc_from_predictions() or "
            f"plot_pr_from_predictions() instead.",
            UserWarning,
            stacklevel=3
        )
    
    # Check if curve is not monotonic (suggests unsorted or wrong data)
    if len(x) > 2:
        x_diff = np.diff(x)
        # ROC: FPR should be monotonically non-decreasing
        # PR: Recall should be monotonically non-decreasing  
        if not (np.all(x_diff >= -1e-10) or np.all(x_diff <= 1e-10)):
            # x is neither monotonically increasing nor decreasing
            n_direction_changes = np.sum(np.diff(np.sign(x_diff)) != 0)
            if n_direction_changes > len(x) * 0.3:  # More than 30% direction changes
                warnings.warn(
                    f"{x_name} is highly non-monotonic ({n_direction_changes} direction changes), "
                    f"which suggests the data may not be a valid curve. "
                    f"A proper ROC/PR curve should have monotonically increasing x-values. "
                    f"If you have raw labels and scores, use plot_roc_from_predictions() or "
                    f"plot_pr_from_predictions() instead.",
                    UserWarning,
                    stacklevel=3
                )


def _validate_predictions_input(y_true: np.ndarray, y_score: np.ndarray) -> None:
    """
    Validate that y_true and y_score look like valid prediction data.
    
    Raises warnings for suspicious shapes that might cause issues.
    """
    # Warn about multi-dimensional input that will be flattened
    if y_true.ndim > 1 and y_true.shape[0] > 1 and y_true.shape[1] > 1:
        warnings.warn(
            f"y_true has shape {y_true.shape} and will be flattened to 1D. "
            f"If this is bootstrap/CV data with multiple runs, consider plotting "
            f"each run separately or using plot_multiple_roc_curves().",
            UserWarning,
            stacklevel=3
        )
    
    if y_score.ndim > 1 and y_score.shape[0] > 1 and y_score.shape[1] > 1:
        warnings.warn(
            f"y_score has shape {y_score.shape} and will be flattened to 1D. "
            f"If this is bootstrap/CV data with multiple runs, consider plotting "
            f"each run separately or using plot_multiple_roc_curves().",
            UserWarning,
            stacklevel=3
        )


def plot_roc_curve(
    fpr: ArrayLike,
    tpr: ArrayLike,
    auc: Optional[float] = None,
    label: Optional[str] = None,
    ax: Optional[Any] = None,
    color: Optional[str] = None,
    linestyle: str = "-",
    linewidth: float = 2.0,
    show_diagonal: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> Tuple[Any, Any]:
    """
    Plot ROC (Receiver Operating Characteristic) curve.

    Parameters
    ----------
    fpr : ArrayLike
        False positive rates
    tpr : ArrayLike
        True positive rates
    auc : float, optional
        Area under the ROC curve (for legend)
    label : str, optional
        Label for the curve
    ax : plt.Axes, optional
        Axes to plot on. If None, creates new figure.
    color : str, optional
        Line color
    linestyle : str
        Line style
    linewidth : float
        Line width
    show_diagonal : bool
        If True, show diagonal reference line
    title : str, optional
        Plot title
    figsize : Tuple[int, int]
        Figure size if creating new figure

    Returns
    -------
    Tuple[Figure, plt.Axes]
        Figure and axes objects
    """
    _check_matplotlib()

    fpr = np.asarray(as_tensor(fpr).cpu().numpy() if hasattr(fpr, 'cpu') else fpr)
    tpr = np.asarray(as_tensor(tpr).cpu().numpy() if hasattr(tpr, 'cpu') else tpr)

    # Validate inputs look like pre-computed curve data
    _validate_curve_input(fpr, tpr, "fpr", "tpr")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Build label
    if label is None:
        label = "ROC"
    if auc is not None:
        label = f"{label} (AUC = {auc:.3f})"

    ax.plot(fpr.ravel(), tpr.ravel(), color=color, linestyle=linestyle, linewidth=linewidth, label=label)

    if show_diagonal:
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.0, label="Random")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title or "ROC Curve", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    return fig, ax


def plot_pr_curve(
    recall: ArrayLike,
    precision: ArrayLike,
    auprc: Optional[float] = None,
    label: Optional[str] = None,
    ax: Optional[Any] = None,
    color: Optional[str] = None,
    linestyle: str = "-",
    linewidth: float = 2.0,
    show_baseline: bool = True,
    baseline_prevalence: Optional[float] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> Tuple[Any, Any]:
    """
    Plot Precision-Recall curve.

    Parameters
    ----------
    recall : ArrayLike
        Recall values
    precision : ArrayLike
        Precision values
    auprc : float, optional
        Area under the PR curve (for legend)
    label : str, optional
        Label for the curve
    ax : plt.Axes, optional
        Axes to plot on
    color : str, optional
        Line color
    linestyle : str
        Line style
    linewidth : float
        Line width
    show_baseline : bool
        If True, show baseline (random classifier)
    baseline_prevalence : float, optional
        Prevalence for baseline line (default 0.5)
    title : str, optional
        Plot title
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    Tuple[Figure, plt.Axes]
        Figure and axes objects
    """
    _check_matplotlib()

    recall = np.asarray(as_tensor(recall).cpu().numpy() if hasattr(recall, 'cpu') else recall)
    precision = np.asarray(as_tensor(precision).cpu().numpy() if hasattr(precision, 'cpu') else precision)

    # Validate inputs look like pre-computed curve data
    _validate_curve_input(recall, precision, "recall", "precision")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Build label
    if label is None:
        label = "PR"
    if auprc is not None:
        label = f"{label} (AUPRC = {auprc:.3f})"

    ax.plot(recall.ravel(), precision.ravel(), color=color, linestyle=linestyle, linewidth=linewidth, label=label)

    if show_baseline:
        baseline = baseline_prevalence or 0.5
        ax.axhline(y=baseline, color="gray", linestyle="--", linewidth=1.0, label=f"Baseline ({baseline:.2f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title or "Precision-Recall Curve", fontsize=14)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_froc_curve(
    sensitivity: ArrayLike,
    avg_fps_per_image: ArrayLike,
    label: Optional[str] = None,
    ax: Optional[Any] = None,
    color: Optional[str] = None,
    linestyle: str = "-",
    linewidth: float = 2.0,
    fps_range: Tuple[float, float] = (0.125, 8.0),
    operating_points: Optional[List[float]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> Tuple[Any, Any]:
    """
    Plot FROC (Free-Response ROC) curve.

    Parameters
    ----------
    sensitivity : ArrayLike
        Sensitivity (true positive rate) values
    avg_fps_per_image : ArrayLike
        Average false positives per image
    label : str, optional
        Label for the curve
    ax : plt.Axes, optional
        Axes to plot on
    color : str, optional
        Line color
    linestyle : str
        Line style
    linewidth : float
        Line width
    fps_range : Tuple[float, float]
        Range of FPs per image to display
    operating_points : List[float], optional
        FP rates at which to mark operating points
    title : str, optional
        Plot title
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    Tuple[Figure, plt.Axes]
        Figure and axes objects
    """
    _check_matplotlib()

    sensitivity = np.asarray(as_tensor(sensitivity).cpu().numpy() if hasattr(sensitivity, 'cpu') else sensitivity)
    avg_fps_per_image = np.asarray(as_tensor(avg_fps_per_image).cpu().numpy() if hasattr(avg_fps_per_image, 'cpu') else avg_fps_per_image)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.plot(avg_fps_per_image, sensitivity, color=color, linestyle=linestyle, linewidth=linewidth, label=label)

    # Mark operating points
    if operating_points is not None:
        for fp_rate in operating_points:
            idx = np.argmin(np.abs(avg_fps_per_image - fp_rate))
            if idx < len(sensitivity):
                sens_at_fp = sensitivity[idx]
                ax.scatter([fp_rate], [sens_at_fp], s=100, zorder=5, edgecolors="black", linewidths=1)
                ax.annotate(f"({fp_rate:.1f}, {sens_at_fp:.2f})", (fp_rate, sens_at_fp),
                           textcoords="offset points", xytext=(10, 10), fontsize=9)

    ax.set_xscale("log")
    ax.set_xlim(fps_range)
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Average False Positives per Image", fontsize=12)
    ax.set_ylabel("Sensitivity", fontsize=12)
    ax.set_title(title or "FROC Curve", fontsize=14)
    if label:
        ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")

    return fig, ax


def plot_multiple_roc_curves(
    curves: List[Dict[str, Union[ArrayLike, float, str]]],
    ax: Optional[Any] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
    colors: Optional[List[str]] = None,
) -> Tuple[Any, Any]:
    """
    Plot multiple ROC curves on the same axes.

    Parameters
    ----------
    curves : List[Dict]
        List of curve dictionaries with keys: 'fpr', 'tpr', 'auc' (optional), 'label'
    ax : plt.Axes, optional
        Axes to plot on
    title : str, optional
        Plot title
    figsize : Tuple[int, int]
        Figure size
    colors : List[str], optional
        List of colors for each curve

    Returns
    -------
    Tuple[Figure, plt.Axes]
        Figure and axes objects
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if colors is None:
        colors = plt.cm.tab10.colors

        for i, curve in enumerate(curves):
            color = colors[i % len(colors)]  # type: ignore
            plot_roc_curve(
                fpr=curve["fpr"],
                tpr=curve["tpr"],
                auc=float(curve["auc"]) if "auc" in curve and curve["auc"] is not None else None,
                label=str(curve.get("label", f"Model {i+1}")),
                ax=ax,
                color=color,
                show_diagonal=(i == 0),  # Only show diagonal once
        )

    ax.set_title(title or "ROC Curves Comparison", fontsize=14)

    return fig, ax


def plot_multiple_pr_curves(
    curves: List[Dict[str, Union[ArrayLike, float, str]]],
    ax: Optional[Any] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
    colors: Optional[List[str]] = None,
) -> Tuple[Any, Any]:
    """
    Plot multiple PR curves on the same axes.

    Parameters
    ----------
    curves : List[Dict]
        List of curve dictionaries with keys: 'recall', 'precision', 'auprc' (optional), 'label'
    ax : plt.Axes, optional
        Axes to plot on
    title : str, optional
        Plot title
    figsize : Tuple[int, int]
        Figure size
    colors : List[str], optional
        List of colors for each curve

    Returns
    -------
    Tuple[Figure, plt.Axes]
        Figure and axes objects
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if colors is None:
        colors = plt.cm.tab10.colors

        for i, curve in enumerate(curves):
            color = colors[i % len(colors)]  # type: ignore
            plot_pr_curve(
                recall=curve["recall"],
                precision=curve["precision"],
                auprc=float(curve["auprc"]) if "auprc" in curve and curve["auprc"] is not None else None,
                label=str(curve.get("label", f"Model {i+1}")),
                ax=ax,
                color=color,
                show_baseline=(i == 0),
            )

    ax.set_title(title or "Precision-Recall Curves Comparison", fontsize=14)

    return fig, ax


def _check_sklearn():
    """Check if sklearn is available."""
    if not HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required for computing curves from predictions. "
            "Install with: pip install scikit-learn"
        )


def plot_roc_from_predictions(
    y_true: ArrayLike,
    y_score: ArrayLike,
    label: Optional[str] = None,
    ax: Optional[Any] = None,
    color: Optional[str] = None,
    linestyle: str = "-",
    linewidth: float = 2.0,
    show_diagonal: bool = True,
    show_auc: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> Tuple[Any, Any]:
    """
    Plot ROC curve directly from labels and prediction scores.

    This is a convenience function that computes the ROC curve from raw
    predictions and then plots it. Use this instead of plot_roc_curve when
    you have y_true/y_score rather than pre-computed fpr/tpr.

    Parameters
    ----------
    y_true : ArrayLike
        True binary labels (0 or 1), shape (n_samples,)
    y_score : ArrayLike
        Predicted probabilities or scores, shape (n_samples,)
    label : str, optional
        Label for the curve
    ax : plt.Axes, optional
        Axes to plot on. If None, creates new figure.
    color : str, optional
        Line color
    linestyle : str
        Line style
    linewidth : float
        Line width
    show_diagonal : bool
        If True, show diagonal reference line
    show_auc : bool
        If True, compute and show AUC in legend
    title : str, optional
        Plot title
    figsize : Tuple[int, int]
        Figure size if creating new figure

    Returns
    -------
    Tuple[Figure, plt.Axes]
        Figure and axes objects

    Examples
    --------
    >>> import numpy as np
    >>> import medeval.vis as vis
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_score = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fig, ax = vis.plot_roc_from_predictions(y_true, y_score)
    """
    _check_matplotlib()
    _check_sklearn()

    # Convert to numpy arrays (before ravel to check original shape)
    y_true_arr = np.asarray(as_tensor(y_true).cpu().numpy() if hasattr(y_true, 'cpu') else y_true)
    y_score_arr = np.asarray(as_tensor(y_score).cpu().numpy() if hasattr(y_score, 'cpu') else y_score)

    # Validate inputs and warn about suspicious shapes
    _validate_predictions_input(y_true_arr, y_score_arr)

    # Flatten to 1D
    y_true = y_true_arr.ravel()
    y_score = y_score_arr.ravel()

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)

    # Compute AUC if requested
    auc_value = None
    if show_auc:
        auc_value = auc(fpr, tpr)

    return plot_roc_curve(
        fpr=fpr,
        tpr=tpr,
        auc=auc_value,
        label=label,
        ax=ax,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        show_diagonal=show_diagonal,
        title=title,
        figsize=figsize,
    )


def plot_pr_from_predictions(
    y_true: ArrayLike,
    y_score: ArrayLike,
    label: Optional[str] = None,
    ax: Optional[Any] = None,
    color: Optional[str] = None,
    linestyle: str = "-",
    linewidth: float = 2.0,
    show_baseline: bool = True,
    show_auprc: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> Tuple[Any, Any]:
    """
    Plot Precision-Recall curve directly from labels and prediction scores.

    This is a convenience function that computes the PR curve from raw
    predictions and then plots it. Use this instead of plot_pr_curve when
    you have y_true/y_score rather than pre-computed recall/precision.

    Parameters
    ----------
    y_true : ArrayLike
        True binary labels (0 or 1), shape (n_samples,)
    y_score : ArrayLike
        Predicted probabilities or scores, shape (n_samples,)
    label : str, optional
        Label for the curve
    ax : plt.Axes, optional
        Axes to plot on
    color : str, optional
        Line color
    linestyle : str
        Line style
    linewidth : float
        Line width
    show_baseline : bool
        If True, show baseline (random classifier)
    show_auprc : bool
        If True, compute and show AUPRC in legend
    title : str, optional
        Plot title
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    Tuple[Figure, plt.Axes]
        Figure and axes objects

    Examples
    --------
    >>> import numpy as np
    >>> import medeval.vis as vis
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_score = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fig, ax = vis.plot_pr_from_predictions(y_true, y_score)
    """
    _check_matplotlib()
    _check_sklearn()

    # Convert to numpy arrays (before ravel to check original shape)
    y_true_arr = np.asarray(as_tensor(y_true).cpu().numpy() if hasattr(y_true, 'cpu') else y_true)
    y_score_arr = np.asarray(as_tensor(y_score).cpu().numpy() if hasattr(y_score, 'cpu') else y_score)

    # Validate inputs and warn about suspicious shapes
    _validate_predictions_input(y_true_arr, y_score_arr)

    # Flatten to 1D
    y_true = y_true_arr.ravel()
    y_score = y_score_arr.ravel()

    # Compute PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    # Compute AUPRC if requested
    auprc_value = None
    if show_auprc:
        # AUPRC: integrate precision over recall (note: recall is decreasing)
        auprc_value = auc(recall, precision)

    # Compute baseline prevalence
    baseline_prevalence = np.mean(y_true) if show_baseline else None

    return plot_pr_curve(
        recall=recall,
        precision=precision,
        auprc=auprc_value,
        label=label,
        ax=ax,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        show_baseline=show_baseline,
        baseline_prevalence=baseline_prevalence,
        title=title,
        figsize=figsize,
    )

