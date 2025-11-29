"""Tests for core utilities."""

import numpy as np
import pytest
import torch

from medeval.core.utils import (
    apply_spacing,
    compute_one_hot,
    compute_weights,
    label_mapping,
    reduce_metrics,
    sample_with_spacing,
)


def test_apply_spacing():
    """Test spacing application to coordinates."""
    coords = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=torch.float32)
    spacing = (2.0, 1.5, 0.5)

    physical_coords = apply_spacing(coords, spacing)

    expected = torch.tensor([[0, 0, 0], [2.0, 1.5, 0.5], [4.0, 3.0, 1.0]], dtype=torch.float32)
    assert torch.allclose(physical_coords, expected)


def test_sample_with_spacing():
    """Test resampling with spacing."""
    # Create 3D image with channel: (B, C, Z, Y, X)
    image = torch.rand(1, 1, 10, 20, 30)  # (B, C, Z, Y, X)
    spacing = (2.0, 1.0, 1.0)  # Anisotropic
    target_spacing = (1.0, 1.0, 1.0)  # Isotropic

    resampled, actual_spacing = sample_with_spacing(image, spacing, target_spacing)

    # Check that spacing was applied
    assert actual_spacing == target_spacing
    # Z dimension should be doubled (2.0 -> 1.0 means 2x more slices)
    assert resampled.shape[2] >= image.shape[2]  # At least as many slices


def test_sample_with_spacing_no_channel():
    """Test resampling with spacing for (B, Z, Y, X) format."""
    # Create 3D image without channel: (B, Z, Y, X)
    image = torch.rand(1, 10, 20, 30)  # (B, Z, Y, X)
    spacing = (2.0, 1.0, 1.0)  # Anisotropic
    target_spacing = (1.0, 1.0, 1.0)  # Isotropic

    resampled, actual_spacing = sample_with_spacing(image, spacing, target_spacing)

    # Check that spacing was applied
    assert actual_spacing == target_spacing
    # Should maintain 4D shape (B, Z, Y, X)
    assert resampled.dim() == 4
    # Z dimension should be doubled
    assert resampled.shape[1] >= image.shape[1]  # At least as many slices


def test_label_mapping():
    """Test label value mapping."""
    labels = torch.tensor([0, 1, 2, 3, 1, 0])
    mapping = {0: 10, 1: 20, 2: 30}

    mapped = label_mapping(labels, mapping)

    expected = torch.tensor([10, 20, 30, 3, 20, 10])
    assert torch.equal(mapped, expected)


def test_label_mapping_with_ignore():
    """Test label mapping with ignore index."""
    labels = torch.tensor([0, 1, 2, 255, 1, 0])
    mapping = {0: 10, 1: 20}
    ignore_index = 255

    mapped = label_mapping(labels, mapping, ignore_index=ignore_index)

    expected = torch.tensor([10, 20, 2, 255, 20, 10])
    assert torch.equal(mapped, expected)


def test_compute_one_hot():
    """Test one-hot encoding."""
    labels = torch.tensor([[0, 1, 2], [1, 0, 2]])
    num_classes = 3

    one_hot = compute_one_hot(labels, num_classes=num_classes)

    assert one_hot.shape == (*labels.shape, num_classes)
    # Check that each position has exactly one class active
    assert torch.all(one_hot.sum(dim=-1) == 1)


def test_reduce_metrics():
    """Test metric reduction strategies."""
    # Create metrics with shape (B, C) = (3, 2)
    metrics = torch.tensor([[0.5, 0.7], [0.6, 0.8], [0.4, 0.9]])

    # Test mean-case
    reduced = reduce_metrics(metrics, reduction="mean-case")
    assert reduced.numel() == 1
    assert torch.allclose(reduced, torch.tensor(metrics.mean()))

    # Test mean-class
    reduced = reduce_metrics(metrics, reduction="mean-class", per_class=True)
    assert reduced.shape == (2,)  # Per class
    assert torch.allclose(reduced, metrics.mean(dim=0))

    # Test global
    reduced = reduce_metrics(metrics, reduction="global")
    assert reduced.numel() == 1
    assert torch.allclose(reduced, torch.tensor(metrics.mean()))


def test_compute_weights():
    """Test sample weight computation."""
    labels = torch.tensor([0, 0, 1, 1, 1, 2])

    # Uniform weights
    weights = compute_weights(labels, method="uniform")
    assert torch.allclose(weights, torch.ones(6))

    # Inverse frequency
    weights = compute_weights(labels, method="inverse_freq")
    assert weights.shape == (6,)
    assert torch.all(weights > 0)


def test_reduce_metrics_per_class():
    """Test per-class metric reduction."""
    metrics = torch.tensor([[0.5, 0.7], [0.6, 0.8], [0.4, 0.9]])

    reduced = reduce_metrics(metrics, reduction="mean-case", per_class=True)
    assert reduced.shape == (2,)
    expected = metrics.mean(dim=0)
    assert torch.allclose(reduced, expected)

