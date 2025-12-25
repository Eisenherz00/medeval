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

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from medeval.core.typing import ArrayLike, as_tensor


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
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

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Build label
    if label is None:
        label = "ROC"
    if auc is not None:
        label = f"{label} (AUC = {auc:.3f})"

    ax.plot(fpr, tpr, color=color, linestyle=linestyle, linewidth=linewidth, label=label)

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

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Build label
    if label is None:
        label = "PR"
    if auprc is not None:
        label = f"{label} (AUPRC = {auprc:.3f})"

    ax.plot(recall, precision, color=color, linestyle=linestyle, linewidth=linewidth, label=label)

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

