"""Calibration visualization utilities."""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    Figure = None

try:
    from sklearn.calibration import calibration_curve
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    calibration_curve = None  # type: ignore

from medeval.core.typing import ArrayLike, as_tensor


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


def plot_reliability_diagram(
    bin_centers: ArrayLike,
    accuracies: ArrayLike,
    confidences: ArrayLike,
    counts: Optional[ArrayLike] = None,
    ece: Optional[float] = None,
    ax: Optional["plt.Axes"] = None,
    show_histogram: bool = True,
    show_gap: bool = True,
    bar_color: str = "#1f77b4",
    gap_color: str = "#ff7f0e",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> Tuple[Figure, "plt.Axes"]:
    """
    Plot reliability diagram (calibration plot).

    Parameters
    ----------
    bin_centers : ArrayLike
        Center values of confidence bins
    accuracies : ArrayLike
        Accuracy in each bin
    confidences : ArrayLike
        Average confidence in each bin
    counts : ArrayLike, optional
        Sample counts per bin (for histogram)
    ece : float, optional
        Expected Calibration Error (for annotation)
    ax : plt.Axes, optional
        Axes to plot on
    show_histogram : bool
        If True, show histogram of sample distribution
    show_gap : bool
        If True, show calibration gap
    bar_color : str
        Color for accuracy bars
    gap_color : str
        Color for gap region
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

    bin_centers = np.asarray(bin_centers)
    accuracies = np.asarray(accuracies)
    confidences = np.asarray(confidences)
    if counts is not None:
        counts = np.asarray(counts)

    n_bins = len(bin_centers)
    bin_width = 1.0 / n_bins

    if ax is None:
        if show_histogram and counts is not None:
            fig, (ax_main, ax_hist) = plt.subplots(
                2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]}, sharex=True
            )
        else:
            fig, ax_main = plt.subplots(figsize=figsize)
            ax_hist = None
    else:
        fig = ax.get_figure()
        ax_main = ax
        ax_hist = None

    # Plot perfect calibration line
    ax_main.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=2, label="Perfect calibration")

    # Plot accuracy bars
    ax_main.bar(
        bin_centers, accuracies, width=bin_width * 0.9, alpha=0.7,
        color=bar_color, edgecolor="black", linewidth=0.5, label="Accuracy"
    )

    # Plot gap (calibration error per bin)
    if show_gap:
        gap_plotted = False
        for i in range(n_bins):
            if accuracies[i] > 0 or confidences[i] > 0:
                gap_bottom = min(accuracies[i], confidences[i])
                gap_height = abs(accuracies[i] - confidences[i])
                # Only add label to first gap bar to avoid duplicate legend entries
                label = "Gap" if not gap_plotted else None
                ax_main.bar(
                    bin_centers[i], gap_height, bottom=gap_bottom,
                    width=bin_width * 0.9, alpha=0.3, color=gap_color,
                    edgecolor=gap_color, linewidth=1, label=label
                )
                gap_plotted = True

    ax_main.set_xlim([0, 1])
    ax_main.set_ylim([0, 1])
    ax_main.set_ylabel("Accuracy", fontsize=12)
    ax_main.set_xlabel("Confidence", fontsize=12)
    ax_main.legend(loc="upper left", fontsize=10)
    ax_main.grid(True, alpha=0.3)

    # Add ECE annotation
    if ece is not None:
        ax_main.text(
            0.95, 0.05, f"ECE = {ece:.3f}",
            transform=ax_main.transAxes, fontsize=12,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        )

    ax_main.set_title(title or "Reliability Diagram", fontsize=14)

    # Plot histogram
    if show_histogram and counts is not None and ax_hist is not None:
        ax_hist.bar(
            bin_centers, counts / counts.sum(), width=bin_width * 0.9,
            color="gray", alpha=0.5, edgecolor="black", linewidth=0.5
        )
        ax_hist.set_ylabel("Frequency", fontsize=10)
        ax_hist.set_xlabel("Confidence", fontsize=12)
        ax_hist.set_xlim([0, 1])
        ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig, ax_main


def plot_calibration_comparison(
    calibration_data: List[Dict[str, Union[ArrayLike, float, str]]],
    ax: Optional["plt.Axes"] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    colors: Optional[List[str]] = None,
) -> Tuple[Figure, "plt.Axes"]:
    """
    Plot multiple calibration curves for comparison.

    Parameters
    ----------
    calibration_data : List[Dict]
        List of dicts with keys: 'confidences', 'accuracies', 'label', 'ece' (optional)
    ax : plt.Axes, optional
        Axes to plot on
    title : str, optional
        Plot title
    figsize : Tuple[int, int]
        Figure size
    colors : List[str], optional
        Colors for each model

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

    # Plot perfect calibration
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=2, label="Perfect")

    for i, data in enumerate(calibration_data):
        confidences = np.asarray(data["confidences"])
        accuracies = np.asarray(data["accuracies"])
        label = data.get("label", f"Model {i+1}")
        ece = data.get("ece")

        if ece is not None:
            label = f"{label} (ECE={ece:.3f})"

        color = colors[i % len(colors)]
        ax.plot(confidences, accuracies, "o-", color=color, linewidth=2, markersize=6, label=label)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title(title or "Calibration Comparison", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    return fig, ax


def plot_decision_curve(
    thresholds: ArrayLike,
    net_benefit: ArrayLike,
    treat_all: Optional[ArrayLike] = None,
    treat_none: Optional[ArrayLike] = None,
    label: Optional[str] = None,
    ax: Optional["plt.Axes"] = None,
    color: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> Tuple[Figure, "plt.Axes"]:
    """
    Plot decision curve analysis.

    Parameters
    ----------
    thresholds : ArrayLike
        Probability thresholds
    net_benefit : ArrayLike
        Net benefit values
    treat_all : ArrayLike, optional
        Net benefit for treat-all strategy
    treat_none : ArrayLike, optional
        Net benefit for treat-none strategy
    label : str, optional
        Label for the model curve
    ax : plt.Axes, optional
        Axes to plot on
    color : str, optional
        Line color
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

    thresholds = np.asarray(thresholds)
    net_benefit = np.asarray(net_benefit)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot model
    ax.plot(thresholds, net_benefit, color=color or "#1f77b4", linewidth=2, label=label or "Model")

    # Plot treat all
    if treat_all is not None:
        ax.plot(thresholds, treat_all, color="gray", linestyle="--", linewidth=1.5, label="Treat All")

    # Plot treat none
    if treat_none is not None:
        ax.plot(thresholds, treat_none, color="black", linestyle="-", linewidth=1.5, label="Treat None")

    ax.set_xlim([0, 1])
    ax.set_xlabel("Threshold Probability", fontsize=12)
    ax.set_ylabel("Net Benefit", fontsize=12)
    ax.set_title(title or "Decision Curve Analysis", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_confidence_histogram(
    confidences: ArrayLike,
    labels: Optional[ArrayLike] = None,
    n_bins: int = 20,
    ax: Optional["plt.Axes"] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> Tuple[Figure, "plt.Axes"]:
    """
    Plot histogram of prediction confidences.

    Parameters
    ----------
    confidences : ArrayLike
        Prediction confidences
    labels : ArrayLike, optional
        True labels (0/1) for coloring
    n_bins : int
        Number of histogram bins
    ax : plt.Axes, optional
        Axes to plot on
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

    confidences = np.asarray(confidences).flatten()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    bins = np.linspace(0, 1, n_bins + 1)

    if labels is not None:
        labels = np.asarray(labels).flatten()
        ax.hist(confidences[labels == 0], bins=bins, alpha=0.6, label="Negative", color="#1f77b4")
        ax.hist(confidences[labels == 1], bins=bins, alpha=0.6, label="Positive", color="#ff7f0e")
        ax.legend(loc="upper center", fontsize=10)
    else:
        ax.hist(confidences, bins=bins, alpha=0.7, color="#1f77b4", edgecolor="black")

    ax.set_xlim([0, 1])
    ax.set_xlabel("Predicted Confidence", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title or "Confidence Distribution", fontsize=14)
    ax.grid(True, alpha=0.3)

    return fig, ax


def _check_sklearn():
    """Check if sklearn is available."""
    if not HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required for computing calibration curves. "
            "Install with: pip install scikit-learn"
        )


def plot_reliability_from_predictions(
    y_true: ArrayLike,
    y_score: ArrayLike,
    n_bins: int = 10,
    strategy: str = "uniform",
    ax: Optional["plt.Axes"] = None,
    show_histogram: bool = False,
    show_gap: bool = True,
    bar_color: str = "#1f77b4",
    gap_color: str = "#ff7f0e",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> Tuple[Figure, "plt.Axes"]:
    """
    Plot reliability diagram directly from labels and prediction scores.

    This is a convenience function that computes the calibration curve from raw
    predictions and then plots it. Use this instead of plot_reliability_diagram
    when you have y_true/y_score rather than pre-computed bin data.

    Parameters
    ----------
    y_true : ArrayLike
        True binary labels (0 or 1), shape (n_samples,)
    y_score : ArrayLike
        Predicted probabilities, shape (n_samples,)
    n_bins : int
        Number of bins for calibration curve (default 10)
    strategy : str
        Strategy for binning: 'uniform' (equal width) or 'quantile' (equal count)
    ax : plt.Axes, optional
        Axes to plot on. If None, creates new figure.
    show_histogram : bool
        If True, show histogram of sample distribution (requires creating new figure)
    show_gap : bool
        If True, show calibration gap between accuracy and confidence
    bar_color : str
        Color for accuracy bars
    gap_color : str
        Color for gap region
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
    >>> y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    >>> y_score = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.6, 0.4])
    >>> fig, ax = vis.plot_reliability_from_predictions(y_true, y_score, n_bins=5)

    Notes
    -----
    The reliability diagram shows:
    - Blue bars: Actual accuracy in each confidence bin
    - Orange gap: Calibration error (difference between confidence and accuracy)
    - Dashed line: Perfect calibration (y = x)

    A well-calibrated model will have accuracy bars close to the diagonal.
    """
    _check_matplotlib()
    _check_sklearn()

    # Convert to numpy arrays (before ravel to check original shape)
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)

    # Warn about suspicious multi-dimensional input
    if y_true_arr.ndim > 1 and y_true_arr.shape[0] > 1 and y_true_arr.shape[1] > 1:
        warnings.warn(
            f"y_true has shape {y_true_arr.shape} and will be flattened to 1D. "
            f"If this is bootstrap/CV data, calibration should be computed per-run.",
            UserWarning,
            stacklevel=2
        )
    if y_score_arr.ndim > 1 and y_score_arr.shape[0] > 1 and y_score_arr.shape[1] > 1:
        warnings.warn(
            f"y_score has shape {y_score_arr.shape} and will be flattened to 1D. "
            f"If this is bootstrap/CV data, calibration should be computed per-run.",
            UserWarning,
            stacklevel=2
        )

    # Flatten to 1D
    y_true = y_true_arr.ravel()
    y_score = y_score_arr.ravel()

    # Compute calibration curve using sklearn
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy=strategy)

    # Compute bin centers based on strategy
    if strategy == "uniform":
        bin_width = 1.0 / n_bins
        # prob_pred gives actual mean predicted probability per bin
        # Use prob_pred as bin centers (they represent where samples actually fall)
        bin_centers = prob_pred
    else:
        # For quantile strategy, use the predicted probabilities as centers
        bin_centers = prob_pred

    # Compute ECE (Expected Calibration Error)
    # We need bin counts to weight the gaps properly
    # Use histogram to get counts per bin
    if strategy == "uniform":
        bin_edges = np.linspace(0, 1, n_bins + 1)
        counts, _ = np.histogram(y_score, bins=bin_edges)
        # Filter to only bins that have samples (matching calibration_curve output)
        non_empty_mask = counts > 0
        counts = counts[non_empty_mask]
    else:
        # For quantile, bins have approximately equal counts
        counts = np.ones(len(prob_true)) * len(y_score) / len(prob_true)

    # Compute ECE
    total_samples = counts.sum()
    if total_samples > 0:
        ece = np.sum(counts * np.abs(prob_true - prob_pred)) / total_samples
    else:
        ece = 0.0

    return plot_reliability_diagram(
        bin_centers=bin_centers,
        accuracies=prob_true,
        confidences=prob_pred,
        counts=counts if show_histogram else None,
        ece=ece,
        ax=ax,
        show_histogram=show_histogram,
        show_gap=show_gap,
        bar_color=bar_color,
        gap_color=gap_color,
        title=title,
        figsize=figsize,
    )

