"""Aggregation utilities: bootstrap, jackknife, stratified aggregation, confidence intervals."""

from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from scipy import stats

from medeval.core.typing import ArrayLike, Tensor, as_tensor

AggregationMethod = Literal["mean", "median", "std", "sem"]


def _compute_statistic(values: np.ndarray, method: AggregationMethod) -> float:
    """
    Compute statistic from values array.

    Parameters
    ----------
    values : np.ndarray
        Input values array
    method : AggregationMethod
        Statistic to compute: "mean", "median", "std", "sem"

    Returns
    -------
    float
        Computed statistic
    """
    if method == "mean":
        return float(np.mean(values))
    elif method == "median":
        return float(np.median(values))
    elif method == "std":
        return float(np.std(values, ddof=1))
    elif method == "sem":
        return float(np.std(values, ddof=1) / np.sqrt(len(values)))
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def _flatten_to_1d(values: ArrayLike) -> np.ndarray:
    """
    Flatten multi-dimensional array to 1D for aggregation.

    Parameters
    ----------
    values : ArrayLike
        Input values (tensor or array)

    Returns
    -------
    np.ndarray
        Flattened 1D array
    """
    values_tensor = as_tensor(values)
    if values_tensor.dim() > 1:
        # Flatten if truly 1D after flattening, else take mean over non-sample dims
        if values_tensor.numel() == values_tensor.shape[0]:
            values_tensor = values_tensor.flatten()
        else:
            values_tensor = values_tensor.mean(dim=tuple(range(1, values_tensor.dim())))
    return values_tensor.cpu().numpy()


def bootstrap_ci(
    values: ArrayLike,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    method: AggregationMethod = "mean",
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Parameters
    ----------
    values : ArrayLike
        Sample values, shape (N,) or (N, ...)
    confidence : float
        Confidence level (e.g., 0.95 for 95% CI)
    n_bootstrap : int
        Number of bootstrap samples
    method : AggregationMethod
        Statistic to compute: "mean", "median", "std", "sem"
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    Tuple[float, float, float]
        (statistic, lower_bound, upper_bound)
    """
    values_np = _flatten_to_1d(values)
    n = len(values_np)

    if seed is not None:
        np.random.seed(seed)

    # Bootstrap sampling
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        sample = values_np[indices]
        stat = _compute_statistic(sample, method)
        bootstrap_stats.append(stat)

    bootstrap_stats = np.array(bootstrap_stats)

    # Compute confidence interval
    alpha = 1.0 - confidence
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    lower_bound = np.percentile(bootstrap_stats, lower_percentile)
    upper_bound = np.percentile(bootstrap_stats, upper_percentile)

    # Compute actual statistic
    statistic = _compute_statistic(values_np, method)

    return float(statistic), float(lower_bound), float(upper_bound)


def jackknife_ci(
    values: ArrayLike,
    confidence: float = 0.95,
    method: AggregationMethod = "mean",
) -> Tuple[float, float, float]:
    """
    Compute jackknife confidence interval for a metric.

    Parameters
    ----------
    values : ArrayLike
        Sample values, shape (N,)
    confidence : float
        Confidence level (e.g., 0.95 for 95% CI)
    method : AggregationMethod
        Statistic to compute: "mean", "median", "std", "sem"

    Returns
    -------
    Tuple[float, float, float]
        (statistic, lower_bound, upper_bound)
    """
    values_np = _flatten_to_1d(values)
    n = len(values_np)

    # Compute full statistic
    full_stat = _compute_statistic(values_np, method)

    # Jackknife: leave-one-out estimates
    jackknife_stats = []
    for i in range(n):
        jackknife_sample = np.concatenate([values_np[:i], values_np[i + 1 :]])
        stat = _compute_statistic(jackknife_sample, method)
        jackknife_stats.append(stat)

    jackknife_stats = np.array(jackknife_stats)

    # Jackknife bias and variance
    jackknife_mean = np.mean(jackknife_stats)
    bias = (n - 1) * (jackknife_mean - full_stat)
    variance = (n - 1) / n * np.sum((jackknife_stats - jackknife_mean) ** 2)
    std_error = np.sqrt(variance)

    # Confidence interval using t-distribution approximation
    alpha = 1.0 - confidence
    t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)

    lower_bound = full_stat - t_critical * std_error
    upper_bound = full_stat + t_critical * std_error

    return float(full_stat), float(lower_bound), float(upper_bound)


def aggregate_metrics(
    metrics: Dict[str, ArrayLike],
    method: AggregationMethod = "mean",
    compute_ci: bool = True,
    ci_method: Literal["bootstrap", "jackknife"] = "bootstrap",
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: Optional[int] = None,
) -> Dict[str, Union[float, Tuple[float, float, float]]]:
    """
    Aggregate multiple metrics with optional confidence intervals.

    Parameters
    ----------
    metrics : Dict[str, ArrayLike]
        Dictionary of metric names to values (shape (N,) per metric)
    method : AggregationMethod
        Aggregation method: "mean", "median", "std", "sem"
    compute_ci : bool
        If True, compute confidence intervals
    ci_method : Literal["bootstrap", "jackknife"]
        Method for CI computation
    confidence : float
        Confidence level
    n_bootstrap : int
        Number of bootstrap samples (if ci_method="bootstrap")
    seed : int, optional
        Random seed

    Returns
    -------
    Dict[str, Union[float, Tuple[float, float, float]]]
        Aggregated metrics, optionally with (stat, lower, upper) tuples
    """
    results = {}

    for name, values in metrics.items():
        values_np = _flatten_to_1d(values)
        stat = _compute_statistic(values_np, method)

        if compute_ci:
            if ci_method == "bootstrap":
                _, lower, upper = bootstrap_ci(values, confidence, n_bootstrap, method, seed)
            else:
                _, lower, upper = jackknife_ci(values, confidence, method)
            results[name] = (stat, lower, upper)
        else:
            results[name] = stat

    return results


def stratified_aggregate(
    metrics: Dict[str, ArrayLike],
    strata: ArrayLike,
    method: AggregationMethod = "mean",
    compute_ci: bool = True,
    ci_method: Literal["bootstrap", "jackknife"] = "bootstrap",
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: Optional[int] = None,
) -> Dict[str, Dict[Union[str, int], Union[float, Tuple[float, float, float]]]]:
    """
    Aggregate metrics stratified by site/scanner/group.

    Parameters
    ----------
    metrics : Dict[str, ArrayLike]
        Dictionary of metric names to values
    strata : ArrayLike
        Stratum labels for each sample, shape (N,)
    method : AggregationMethod
        Aggregation method
    compute_ci : bool
        If True, compute confidence intervals
    ci_method : Literal["bootstrap", "jackknife"]
        CI computation method
    confidence : float
        Confidence level
    n_bootstrap : int
        Number of bootstrap samples
    seed : int, optional
        Random seed

    Returns
    -------
    Dict[str, Dict[Union[str, int], Union[float, Tuple[float, float, float]]]]
        Metrics stratified by stratum, then by metric name
    """
    strata_tensor = as_tensor(strata)
    if strata_tensor.dim() > 1:
        strata_tensor = strata_tensor.flatten()
    strata_np = strata_tensor.cpu().numpy()

    unique_strata = np.unique(strata_np)
    results = {}

    for metric_name, values in metrics.items():
        values_np = _flatten_to_1d(values)

        if len(values_np) != len(strata_np):
            raise ValueError(
                f"Metric {metric_name} length {len(values_np)} != strata length {len(strata_np)}"
            )

        results[metric_name] = {}

        for stratum in unique_strata:
            mask = strata_np == stratum
            stratum_values = values_np[mask]
            stat = _compute_statistic(stratum_values, method)

            if compute_ci:
                if ci_method == "bootstrap":
                    _, lower, upper = bootstrap_ci(stratum_values, confidence, n_bootstrap, method, seed)
                else:
                    _, lower, upper = jackknife_ci(stratum_values, confidence, method)
                results[metric_name][str(int(stratum))] = (stat, lower, upper)
            else:
                results[metric_name][str(int(stratum))] = stat

    return results

