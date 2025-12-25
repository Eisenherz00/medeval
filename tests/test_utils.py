"""Tests for core utilities."""

import numpy as np
import pytest
import torch

from medeval.core.utils import (
    apply_spacing,
    compute_one_hot,
    compute_weights,
    get_spatial_dims_from_spacing,
    label_mapping,
    normalize_input_shapes,
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


class TestNormalizeInputShapes:
    """Tests for normalize_input_shapes function."""

    # ===== Shape normalization tests =====

    def test_2d_unbatched(self):
        """Test 2D unbatched input: (H, W) -> (1, 1, H, W)."""
        pred = torch.rand(64, 64)
        target = torch.rand(64, 64)
        p, t, spatial_dims, sp = normalize_input_shapes(pred, target)
        assert p.shape == (1, 1, 64, 64)
        assert t.shape == (1, 1, 64, 64)
        assert spatial_dims == 2
        assert sp is None

    def test_2d_batched_no_channel(self):
        """Test 2D batched without channel: (B, H, W) -> (B, 1, H, W)."""
        pred = torch.rand(4, 64, 64)
        target = torch.rand(4, 64, 64)
        p, t, spatial_dims, sp = normalize_input_shapes(pred, target)
        assert p.shape == (4, 1, 64, 64)
        assert t.shape == (4, 1, 64, 64)
        assert spatial_dims == 2

    def test_2d_batched_with_channel(self):
        """Test 2D batched with channel: (B, C, H, W) -> unchanged."""
        pred = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)
        p, t, spatial_dims, sp = normalize_input_shapes(pred, target)
        assert p.shape == (2, 3, 64, 64)
        assert t.shape == (2, 3, 64, 64)
        assert spatial_dims == 2

    def test_3d_unbatched(self):
        """Test 3D unbatched input: (Z, Y, X) -> (1, 1, Z, Y, X)."""
        pred = torch.rand(32, 64, 64)  # Z=32 > 4, so treated as 3D
        target = torch.rand(32, 64, 64)
        p, t, spatial_dims, sp = normalize_input_shapes(pred, target)
        assert p.shape == (1, 1, 32, 64, 64)
        assert t.shape == (1, 1, 32, 64, 64)
        assert spatial_dims == 3

    def test_3d_batched_no_channel(self):
        """Test 3D batched without channel: (B, Z, Y, X) -> (B, 1, Z, Y, X)."""
        pred = torch.rand(2, 32, 64, 64)  # axis-1=32 > 4, so treated as 3D
        target = torch.rand(2, 32, 64, 64)
        p, t, spatial_dims, sp = normalize_input_shapes(pred, target)
        assert p.shape == (2, 1, 32, 64, 64)
        assert t.shape == (2, 1, 32, 64, 64)
        assert spatial_dims == 3

    def test_3d_batched_with_channel(self):
        """Test 3D batched with channel: (B, C, Z, Y, X) -> unchanged."""
        pred = torch.rand(2, 1, 32, 64, 64)
        target = torch.rand(2, 1, 32, 64, 64)
        p, t, spatial_dims, sp = normalize_input_shapes(pred, target)
        assert p.shape == (2, 1, 32, 64, 64)
        assert t.shape == (2, 1, 32, 64, 64)
        assert spatial_dims == 3

    # ===== Spacing validation tests =====

    def test_spacing_valid_2d(self):
        """Test valid 2D spacing."""
        pred = torch.rand(64, 64)
        target = torch.rand(64, 64)
        spacing = (0.5, 0.5)  # (dy, dx)
        p, t, spatial_dims, sp = normalize_input_shapes(pred, target, spacing=spacing)
        assert spatial_dims == 2
        assert sp == spacing

    def test_spacing_valid_3d(self):
        """Test valid 3D spacing."""
        pred = torch.rand(32, 64, 64)
        target = torch.rand(32, 64, 64)
        spacing = (2.0, 0.5, 0.5)  # (dz, dy, dx)
        p, t, spatial_dims, sp = normalize_input_shapes(pred, target, spacing=spacing)
        assert spatial_dims == 3
        assert sp == spacing

    def test_spacing_mismatch_raises(self):
        """Test that mismatched spacing raises ValueError."""
        pred = torch.rand(64, 64)  # 2D
        target = torch.rand(64, 64)
        spacing = (1.0, 1.0, 1.0)  # 3D spacing for 2D image
        with pytest.raises(ValueError, match="Spacing dimension"):
            normalize_input_shapes(pred, target, spacing=spacing)

    def test_require_spacing_none_raises(self):
        """Test that require_spacing=True with spacing=None raises ValueError."""
        pred = torch.rand(64, 64)
        target = torch.rand(64, 64)
        with pytest.raises(ValueError, match="Spacing is required"):
            normalize_input_shapes(pred, target, spacing=None, require_spacing=True)

    def test_require_spacing_false_allows_none(self):
        """Test that require_spacing=False allows spacing=None."""
        pred = torch.rand(64, 64)
        target = torch.rand(64, 64)
        p, t, spatial_dims, sp = normalize_input_shapes(
            pred, target, spacing=None, require_spacing=False
        )
        assert sp is None  # No error, spacing remains None

    # ===== dtype/device contract tests =====

    def test_pred_dtype_float32(self):
        """Test that pred is always converted to float32."""
        pred = torch.randint(0, 2, (64, 64), dtype=torch.int64)
        target = torch.randint(0, 2, (64, 64), dtype=torch.int64)
        p, t, _, _ = normalize_input_shapes(pred, target)
        assert p.dtype == torch.float32

    def test_target_dtype_preserved(self):
        """Test that target keeps its original dtype (for label maps)."""
        pred = torch.rand(64, 64)
        target = torch.randint(0, 5, (64, 64), dtype=torch.int64)
        p, t, _, _ = normalize_input_shapes(pred, target)
        assert t.dtype == torch.int64  # Original dtype preserved

    def test_device_alignment(self):
        """Test that target is moved to pred's device."""
        pred = torch.rand(64, 64)
        target = torch.rand(64, 64)
        # Both on CPU by default, test passes if no error
        p, t, _, _ = normalize_input_shapes(pred, target)
        assert p.device == t.device


class TestGetSpatialDimsFromSpacing:
    """Tests for get_spatial_dims_from_spacing helper."""

    def test_none_spacing(self):
        """Test that None spacing returns None."""
        assert get_spatial_dims_from_spacing(None) is None

    def test_2d_spacing(self):
        """Test 2D spacing returns 2."""
        assert get_spatial_dims_from_spacing((1.0, 1.0)) == 2

    def test_3d_spacing(self):
        """Test 3D spacing returns 3."""
        assert get_spatial_dims_from_spacing((1.0, 1.0, 1.0)) == 3

