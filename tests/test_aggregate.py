"""Tests for aggregation and confidence intervals."""

import numpy as np
import pytest
import torch

from medeval.core.aggregate import (
    aggregate_metrics,
    bootstrap_ci,
    jackknife_ci,
    stratified_aggregate,
)


def test_bootstrap_ci():
    """Test bootstrap confidence interval computation."""
    # Create sample data
    values = torch.randn(100) + 5.0  # Mean around 5.0

    stat, lower, upper = bootstrap_ci(values, confidence=0.95, n_bootstrap=100, seed=42)

    # Check that CI bounds are reasonable
    assert lower < stat < upper
    assert stat > 0  # Should be around 5.0

    # Test with different methods
    stat_median, lower_median, upper_median = bootstrap_ci(
        values, confidence=0.95, n_bootstrap=100, method="median", seed=42
    )
    assert lower_median < stat_median < upper_median


def test_jackknife_ci():
    """Test jackknife confidence interval computation."""
    values = torch.randn(50) + 3.0

    stat, lower, upper = jackknife_ci(values, confidence=0.95, method="mean")

    assert lower < stat < upper
    assert stat > 0


def test_aggregate_metrics():
    """Test metric aggregation."""
    metrics = {
        "dice": torch.rand(100) * 0.5 + 0.5,  # Between 0.5 and 1.0
        "hausdorff": torch.rand(100) * 10.0,  # Between 0 and 10
    }

    results = aggregate_metrics(metrics, method="mean", compute_ci=True, n_bootstrap=100, seed=42)

    assert "dice" in results
    assert "hausdorff" in results

    # Check that results are tuples (stat, lower, upper) when CI is computed
    assert isinstance(results["dice"], tuple)
    assert len(results["dice"]) == 3

    # Test without CI
    results_no_ci = aggregate_metrics(metrics, method="mean", compute_ci=False)
    assert isinstance(results_no_ci["dice"], float)


def test_stratified_aggregate():
    """Test stratified aggregation."""
    metrics = {
        "dice": torch.rand(100) * 0.5 + 0.5,
    }
    strata = torch.tensor([0] * 50 + [1] * 50)  # Two strata

    results = stratified_aggregate(
        metrics, strata, method="mean", compute_ci=True, n_bootstrap=50, seed=42
    )

    assert "dice" in results
    assert "0" in results["dice"]
    assert "1" in results["dice"]

    # Check that each stratum has a result
    assert isinstance(results["dice"]["0"], tuple)
    assert isinstance(results["dice"]["1"], tuple)


def test_aggregate_metrics_different_methods():
    """Test different aggregation methods."""
    metrics = {"test": torch.rand(100)}

    for method in ["mean", "median", "std", "sem"]:
        results = aggregate_metrics(metrics, method=method, compute_ci=False)
        assert "test" in results
        assert isinstance(results["test"], float)

