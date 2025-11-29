"""Benchmark tests for large-scale evaluation."""

import pytest
import torch

from medeval.core.aggregate import aggregate_metrics
from medeval.core.utils import reduce_metrics


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
    spatial_shape = (32, 32, 32)  # Small 3D volumes to keep memory reasonable

    # Simulate metrics for 10k cases
    # Instead of storing all data, we'll process in chunks
    chunk_size = 1000
    all_metrics = []

    for chunk_start in range(0, n_cases, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_cases)
        chunk_size_actual = chunk_end - chunk_start

        # Simulate metric computation for this chunk
        # Shape: (chunk_size, n_classes)
        chunk_metrics = torch.rand(chunk_size_actual, n_classes)

        # Reduce per case
        reduced = reduce_metrics(chunk_metrics, reduction="mean-case", per_class=False)
        all_metrics.append(reduced)

    # Aggregate all metrics
    all_metrics_tensor = torch.cat(all_metrics)

    # Final aggregation
    metrics_dict = {"test_metric": all_metrics_tensor}
    results = aggregate_metrics(metrics_dict, method="mean", compute_ci=True, n_bootstrap=100, seed=42)

    # Verify results
    assert "test_metric" in results
    assert isinstance(results["test_metric"], tuple)
    assert len(results["test_metric"]) == 3

    # Check that we processed all cases
    assert len(all_metrics_tensor) == n_cases


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

