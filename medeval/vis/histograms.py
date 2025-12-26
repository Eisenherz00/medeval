"""Error histogram and distribution visualization."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    Figure = None

from medeval.core.typing import ArrayLike, as_tensor


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


def plot_error_histogram(
    errors: ArrayLike,
    bins: Union[int, str, ArrayLike] = 20,
    ax: Optional["plt.Axes"] = None,
    color: str = "#1f77b4",
    density: bool = False,
    show_stats: bool = True,
    show_percentiles: bool = True,
    percentiles: List[float] = [50, 95],
    xlabel: str = "Error",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    legend_loc: str = "auto",
) -> Tuple[Figure, "plt.Axes"]:
    """
    Plot histogram of errors/metrics with statistics.

    Parameters
    ----------
    errors : ArrayLike
        Error or metric values (e.g., Dice scores)
    bins : int, str, or ArrayLike
        Number of bins, 'auto' for automatic selection, or bin edges
    ax : plt.Axes, optional
        Axes to plot on
    color : str
        Histogram color
    density : bool
        If True, plot probability density instead of counts
    show_stats : bool
        If True, show mean and std in a text box
    show_percentiles : bool
        If True, show percentile lines
    percentiles : List[float]
        Percentiles to show (default: [50, 95])
    xlabel : str
        X-axis label
    title : str, optional
        Plot title
    figsize : Tuple[int, int]
        Figure size
    legend_loc : str
        Legend location ('auto' places it opposite to the data mass)

    Returns
    -------
    Tuple[Figure, plt.Axes]
        Figure and axes objects

    Examples
    --------
    >>> import numpy as np
    >>> import medeval.vis as vis
    >>> dice_scores = np.random.beta(8, 2, 100)  # Typical left-skewed Dice distribution
    >>> fig, ax = vis.plot_error_histogram(dice_scores, xlabel='Dice Score')
    """
    _check_matplotlib()

    errors = np.asarray(as_tensor(errors).cpu().numpy() if hasattr(errors, 'cpu') else errors).flatten()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot histogram
    n, bin_edges, patches = ax.hist(
        errors, bins=bins, color=color, alpha=0.7, 
        edgecolor="black", linewidth=0.5, density=density
    )

    # Compute statistics
    mean_val = np.mean(errors)
    std_val = np.std(errors)
    median_val = np.median(errors)

    # Determine legend/stats position based on data distribution
    # If median > midpoint of range, data is right-skewed, put legend on left
    data_range = errors.max() - errors.min()
    midpoint = errors.min() + data_range / 2
    data_is_right_heavy = median_val > midpoint

    if legend_loc == "auto":
        legend_loc = "upper left" if data_is_right_heavy else "upper right"
    
    # Stats box position opposite to legend
    stats_x = 0.02 if not data_is_right_heavy else 0.98
    stats_ha = "left" if not data_is_right_heavy else "right"

    # Add mean line
    ax.axvline(mean_val, color="#d62728", linestyle="--", linewidth=2, label=f"Mean = {mean_val:.3f}")

    # Add percentile lines
    if show_percentiles:
        percentile_colors = ["#ff7f0e", "#2ca02c", "#9467bd"]  # orange, green, purple
        for i, p in enumerate(percentiles):
            pval = np.percentile(errors, p)
            ax.axvline(pval, color=percentile_colors[i % len(percentile_colors)], 
                      linestyle=":", linewidth=2, label=f"{p}th pctl = {pval:.3f}")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Density" if density else "Count", fontsize=12)
    ax.set_title(title or "Distribution", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add statistics text box (positioned to avoid overlap)
    if show_stats:
        stats_text = f"n = {len(errors)}\nMean: {mean_val:.3f}\nStd: {std_val:.3f}"
        ax.text(stats_x, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", horizontalalignment=stats_ha,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    ax.legend(loc=legend_loc, fontsize=9, framealpha=0.9)

    return fig, ax


def plot_metric_distribution(
    metrics: Dict[str, ArrayLike],
    kind: str = "box",
    ax: Optional["plt.Axes"] = None,
    colors: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> Tuple[Figure, "plt.Axes"]:
    """
    Plot distribution of multiple metrics.

    Parameters
    ----------
    metrics : Dict[str, ArrayLike]
        Dictionary of metric names to values
    kind : str
        Type of plot: "box", "violin", or "bar"
    ax : plt.Axes, optional
        Axes to plot on
    colors : List[str], optional
        Colors for each metric
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

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    metric_names = list(metrics.keys())
    metric_values = [np.asarray(as_tensor(v).cpu().numpy() if hasattr(v, 'cpu') else v).flatten() 
                     for v in metrics.values()]

    if colors is None:
        colors = plt.cm.tab10.colors

    if kind == "box":
        bp = ax.boxplot(metric_values, labels=metric_names, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    elif kind == "violin":
        vp = ax.violinplot(metric_values, positions=range(len(metric_names)), showmeans=True, showmedians=True)
        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(colors[i % len(colors)])
            body.set_alpha(0.7)
        ax.set_xticks(range(len(metric_names)))
        ax.set_xticklabels(metric_names)
    elif kind == "bar":
        means = [np.mean(v) for v in metric_values]
        stds = [np.std(v) for v in metric_values]
        x = range(len(metric_names))
        ax.bar(x, means, yerr=stds, color=colors[:len(metric_names)], alpha=0.7, capsize=5, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
    else:
        raise ValueError(f"Unknown plot kind: {kind}. Use 'box', 'violin', or 'bar'.")

    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(title or "Metric Distribution", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    # Rotate x labels if many metrics
    if len(metric_names) > 5:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    return fig, ax


def plot_per_class_metrics(
    metrics: Dict[str, ArrayLike],
    class_names: Optional[List[str]] = None,
    ax: Optional["plt.Axes"] = None,
    show_values: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> Tuple[Figure, "plt.Axes"]:
    """
    Plot per-class metrics as grouped bar chart.

    Parameters
    ----------
    metrics : Dict[str, ArrayLike]
        Dictionary of metric names to per-class values
    class_names : List[str], optional
        Names for each class
    ax : plt.Axes, optional
        Axes to plot on
    show_values : bool
        If True, show value labels on bars
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

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    metric_names = list(metrics.keys())
    metric_values = [np.asarray(as_tensor(v).cpu().numpy() if hasattr(v, 'cpu') else v).flatten() 
                     for v in metrics.values()]

    n_classes = len(metric_values[0])
    n_metrics = len(metric_names)

    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    x = np.arange(n_classes)
    width = 0.8 / n_metrics

    colors = plt.cm.tab10.colors

    for i, (name, values) in enumerate(zip(metric_names, metric_values)):
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=name, color=colors[i % len(colors)], alpha=0.8)
        
        if show_values:
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                       f"{val:.2f}", ha="center", va="bottom", fontsize=8, rotation=90)

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(title or "Per-Class Metrics", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    return fig, ax


def plot_stratified_results(
    results: Dict[str, Dict[str, float]],
    metric_name: str = "dice",
    strata_names: Optional[List[str]] = None,
    ax: Optional["plt.Axes"] = None,
    show_ci: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> Tuple[Figure, "plt.Axes"]:
    """
    Plot stratified results (e.g., by site or scanner).

    Parameters
    ----------
    results : Dict[str, Dict[str, float]]
        Results dict with structure {stratum: {metric: value or (value, lower, upper)}}
    metric_name : str
        Name of metric to plot
    strata_names : List[str], optional
        Display names for strata
    ax : plt.Axes, optional
        Axes to plot on
    show_ci : bool
        If True, show confidence intervals
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

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    strata = list(results.keys())
    if strata_names is None:
        strata_names = strata

    values = []
    errors_low = []
    errors_high = []

    for stratum in strata:
        val = results[stratum].get(metric_name, 0)
        if isinstance(val, tuple):
            values.append(val[0])
            errors_low.append(val[0] - val[1])
            errors_high.append(val[2] - val[0])
        else:
            values.append(val)
            errors_low.append(0)
            errors_high.append(0)

    x = np.arange(len(strata))

    if show_ci and any(e > 0 for e in errors_low + errors_high):
        ax.bar(x, values, yerr=[errors_low, errors_high], capsize=5,
               color=plt.cm.tab10.colors[0], alpha=0.7, edgecolor="black")
    else:
        ax.bar(x, values, color=plt.cm.tab10.colors[0], alpha=0.7, edgecolor="black")

    ax.set_xlabel("Stratum", fontsize=12)
    ax.set_ylabel(metric_name.capitalize(), fontsize=12)
    ax.set_title(title or f"{metric_name.capitalize()} by Stratum", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(strata_names, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    return fig, ax

