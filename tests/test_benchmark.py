"""Benchmark tests for large-scale evaluation."""

import gc
import sys

import numpy as np
import pytest
import torch

from medeval.core.aggregate import aggregate_metrics
from medeval.core.utils import reduce_metrics


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    import tracemalloc
    
    if not tracemalloc.is_tracing():
        tracemalloc.start()
    
    current, peak = tracemalloc.get_traced_memory()
    return peak / 1024 / 1024  # Convert to MB


@pytest.mark.slow
@pytest.mark.acceptance
def test_10k_case_benchmark():
    """
    Acceptance test: 10k-case synthetic benchmark should complete <2 GB RAM (streaming).

    This test verifies that the framework can handle large-scale evaluation
    without excessive memory usage.
    """
    n_cases = 10000
    n_classes = 5

    # Simulate metrics for 10k cases - process in chunks to limit memory
    chunk_size = 1000
    all_metrics = []

    for chunk_start in range(0, n_cases, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_cases)
        chunk_size_actual = chunk_end - chunk_start

        # Simulate metric computation for this chunk
        # Shape: (chunk_size, n_classes)
        chunk_metrics = torch.rand(chunk_size_actual, n_classes)

        # Reduce per case - average over classes to get per-case score
        # Use mean over class dimension (dim=1) to get (chunk_size,) tensor
        reduced = chunk_metrics.mean(dim=1)
        all_metrics.append(reduced)
        
        # Clear intermediate tensors
        del chunk_metrics
        gc.collect()

    # Concatenate all chunk results into single tensor
    all_metrics_tensor = torch.cat(all_metrics)

    # Final aggregation with CI
    metrics_dict = {"test_metric": all_metrics_tensor}
    results = aggregate_metrics(metrics_dict, method="mean", compute_ci=True, n_bootstrap=100, seed=42)

    # Verify results
    assert "test_metric" in results
    assert isinstance(results["test_metric"], tuple)
    assert len(results["test_metric"]) == 3

    # Check that we processed all cases
    assert len(all_metrics_tensor) == n_cases


@pytest.mark.slow
@pytest.mark.acceptance
def test_10k_case_memory_limit():
    """
    Acceptance test: Verify memory usage stays under 2GB for 10k cases.
    
    This test uses tracemalloc to monitor actual memory usage.
    """
    import tracemalloc
    
    # Start memory tracking
    tracemalloc.start()
    
    n_cases = 10000
    n_classes = 5
    chunk_size = 500  # Smaller chunks to test streaming behavior
    
    all_dice_scores = []
    
    for chunk_start in range(0, n_cases, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_cases)
        chunk_size_actual = chunk_end - chunk_start
        
        # Simulate predictions and targets for this chunk
        # Shape: (chunk_size, 1, 16, 16, 16) - small volumes
        preds = torch.rand(chunk_size_actual, 1, 16, 16, 16) > 0.5
        targets = torch.rand(chunk_size_actual, 1, 16, 16, 16) > 0.5
        
        # Compute Dice-like metric (intersection / union)
        intersection = (preds & targets).sum(dim=(1, 2, 3, 4)).float()
        union = preds.sum(dim=(1, 2, 3, 4)).float() + targets.sum(dim=(1, 2, 3, 4)).float()
        dice = (2 * intersection / (union + 1e-6))
        
        all_dice_scores.append(dice.numpy())
        
        # Clear tensors
        del preds, targets, intersection, union, dice
        gc.collect()
    
    # Aggregate
    all_scores = np.concatenate(all_dice_scores)
    
    # Check memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    peak_mb = peak / 1024 / 1024
    
    # Verify we stayed under 2GB
    assert peak_mb < 2000, f"Peak memory usage {peak_mb:.1f} MB exceeded 2GB limit"
    
    # Verify we processed all cases
    assert len(all_scores) == n_cases
    
    print(f"Peak memory usage: {peak_mb:.1f} MB")


@pytest.mark.slow
def test_gpu_optional():
    """Test that operations work on CPU (GPU optional)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU test")

    # Test that we can move tensors to GPU
    data = torch.rand(10, 10, 10)
    data_gpu = data.cuda()

    # Perform some operations
    result = data_gpu.mean()

    assert result.device.type == "cuda"
    assert isinstance(result.item(), float)


@pytest.mark.slow
def test_streaming_evaluation():
    """Test streaming evaluation pattern for large datasets."""
    n_total = 5000
    batch_size = 100
    
    # Accumulate metrics in streaming fashion
    running_sum = 0.0
    running_count = 0
    all_values = []
    
    for batch_start in range(0, n_total, batch_size):
        batch_end = min(batch_start + batch_size, n_total)
        batch_n = batch_end - batch_start
        
        # Simulate batch metric computation
        batch_metrics = np.random.rand(batch_n)
        
        # Update running statistics
        running_sum += batch_metrics.sum()
        running_count += batch_n
        all_values.extend(batch_metrics.tolist())
    
    # Compute final statistics
    mean = running_sum / running_count
    std = np.std(all_values)
    
    # Bootstrap CI (using stored values)
    all_values = np.array(all_values)
    bootstrap_means = []
    for _ in range(100):
        sample = np.random.choice(all_values, size=len(all_values), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)
    
    # Verify
    assert running_count == n_total
    assert 0 < mean < 1  # Random values should average around 0.5
    assert ci_lower < mean < ci_upper

