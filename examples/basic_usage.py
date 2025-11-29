"""Basic usage examples for MedEval core functionality."""

import torch

from medeval.core import (
    aggregate_metrics,
    as_tensor,
    bootstrap_ci,
    reduce_metrics,
)


def example_typing():
    """Example of using typing utilities."""
    print("=== Typing Example ===")

    # Convert numpy array to tensor
    import numpy as np

    arr = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    tensor = as_tensor(arr, device="cpu")
    print(f"Converted array to tensor: {tensor}")
    print(f"Device: {tensor.device}, Dtype: {tensor.dtype}")


def example_reduction():
    """Example of metric reduction."""
    print("\n=== Reduction Example ===")

    # Simulate metrics for 5 cases, 3 classes
    metrics = torch.rand(5, 3)  # (B, C)

    print(f"Original metrics shape: {metrics.shape}")

    # Reduce per case
    reduced = reduce_metrics(metrics, reduction="mean-case")
    print(f"Mean-case reduction: {reduced.item():.4f}")

    # Reduce per class
    reduced_per_class = reduce_metrics(metrics, reduction="mean-class", per_class=True)
    print(f"Mean-class reduction (per class): {reduced_per_class}")


def example_aggregation():
    """Example of metric aggregation with confidence intervals."""
    print("\n=== Aggregation Example ===")

    # Simulate dice scores for 100 cases
    dice_scores = torch.rand(100) * 0.3 + 0.7  # Between 0.7 and 1.0

    metrics = {"dice": dice_scores}

    # Aggregate with bootstrap CI
    results = aggregate_metrics(
        metrics, method="mean", compute_ci=True, ci_method="bootstrap", n_bootstrap=1000, seed=42
    )

    stat, lower, upper = results["dice"]
    print(f"Dice Score: {stat:.4f} (95% CI: [{lower:.4f}, {upper:.4f}])")


def example_bootstrap():
    """Example of bootstrap confidence interval."""
    print("\n=== Bootstrap CI Example ===")

    # Sample data
    values = torch.randn(50) + 5.0

    stat, lower, upper = bootstrap_ci(values, confidence=0.95, n_bootstrap=1000, seed=42)
    print(f"Mean: {stat:.4f} (95% CI: [{lower:.4f}, {upper:.4f}])")


if __name__ == "__main__":
    example_typing()
    example_reduction()
    example_aggregation()
    example_bootstrap()

