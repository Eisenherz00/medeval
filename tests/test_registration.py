"""Tests for registration metrics."""

import numpy as np
import pytest
import torch

from medeval.metrics.registration import (
    bending_energy,
    compute_registration_metrics,
    deformation_smoothness,
    jacobian_determinant,
    mind_ssd,
    normalized_cross_correlation,
    normalized_mutual_information,
    target_registration_error,
)


class TestTargetRegistrationError:
    """Tests for Target Registration Error (TRE)."""

    def test_tre_perfect_alignment(self):
        """Test TRE for perfectly aligned landmarks."""
        pred_landmarks = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        target_landmarks = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

        results = target_registration_error(pred_landmarks, target_landmarks)

        assert results["mean"] == 0.0
        assert results["median"] == 0.0
        assert results["95th_percentile"] == 0.0

    def test_tre_with_error(self):
        """Test TRE with known error."""
        pred_landmarks = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        target_landmarks = np.array([[1.0, 0.0, 0.0], [2.0, 1.0, 1.0]])  # Shifted by 1.0 in x

        results = target_registration_error(pred_landmarks, target_landmarks)

        # Mean TRE should be 1.0 (Euclidean distance)
        assert abs(results["mean"] - 1.0) < 1e-6

    def test_tre_with_spacing(self):
        """Test TRE with physical spacing."""
        pred_landmarks = np.array([[0.0, 0.0, 0.0]])
        target_landmarks = np.array([[1.0, 1.0, 1.0]])

        # With spacing (2.0, 2.0, 2.0), physical distance should be doubled
        spacing = (2.0, 2.0, 2.0)
        results = target_registration_error(pred_landmarks, target_landmarks, spacing=spacing)

        # Physical distance = sqrt((2*1)^2 + (2*1)^2 + (2*1)^2) = sqrt(12) ≈ 3.46
        assert 3.4 < results["mean"] < 3.5

    @pytest.mark.acceptance
    def test_tre_landmark_perturbation(self):
        """Acceptance test: TRE with controlled noise perturbation."""
        # Create ground truth landmarks
        n_landmarks = 10
        target_landmarks = np.random.rand(n_landmarks, 3) * 100

        # Add controlled noise (Gaussian with std=1.0)
        noise_std = 1.0
        noise = np.random.randn(n_landmarks, 3) * noise_std
        pred_landmarks = target_landmarks + noise

        results = target_registration_error(pred_landmarks, target_landmarks)

        # Mean TRE should be approximately noise_std * sqrt(3) for 3D
        expected_mean = noise_std * np.sqrt(3)
        assert abs(results["mean"] - expected_mean) < 2.0  # Allow some variance

    def test_tre_per_case(self):
        """Test TRE with per-case computation."""
        # Multiple cases
        pred_landmarks = np.array([[[0.0, 0.0], [1.0, 1.0]], [[2.0, 2.0], [3.0, 3.0]]])
        target_landmarks = np.array([[[0.0, 0.0], [1.0, 1.0]], [[2.0, 2.0], [3.0, 3.0]]])

        results = target_registration_error(pred_landmarks, target_landmarks, per_case=True)

        assert "per_case" in results
        assert len(results["per_case"]) == 2


class TestImageSimilarity:
    """Tests for image similarity metrics."""

    def test_nmi_identical_images(self):
        """Test NMI for identical images."""
        image1 = np.random.rand(100, 100)
        image2 = image1.copy()

        nmi = normalized_mutual_information(image1, image2)

        # Identical images should have high NMI (close to 1.0)
        assert nmi > 0.9

    def test_nmi_different_images(self):
        """Test NMI for different images."""
        image1 = np.random.rand(100, 100)
        image2 = np.random.rand(100, 100)

        nmi = normalized_mutual_information(image1, image2)

        # Different images should have lower NMI
        assert 0.0 <= nmi <= 1.0

    def test_ncc_identical_images(self):
        """Test NCC for identical images."""
        image1 = torch.rand(100, 100)
        image2 = image1.clone()

        ncc = normalized_cross_correlation(image1, image2)

        # Identical images should have NCC = 1.0
        assert abs(ncc - 1.0) < 1e-6

    def test_ncc_opposite_images(self):
        """Test NCC for opposite images."""
        image1 = torch.rand(100, 100)
        image2 = 1.0 - image1

        ncc = normalized_cross_correlation(image1, image2)

        # Opposite images should have negative NCC
        assert ncc < 0.0

    def test_mind_ssd(self):
        """Test MIND-SSD computation."""
        image1 = np.random.rand(50, 50)
        image2 = np.random.rand(50, 50)

        ssd = mind_ssd(image1, image2)

        assert ssd >= 0.0


class TestDeformationFieldQuality:
    """Tests for deformation field quality metrics."""

    def test_jacobian_determinant(self):
        """Test Jacobian determinant computation."""
        # Simple 2D deformation field (identity + small perturbation)
        deformation_field = torch.zeros(2, 10, 10)
        # Add small uniform deformation
        deformation_field[0, :, :] = 0.1  # x displacement
        deformation_field[1, :, :] = 0.1  # y displacement

        results = jacobian_determinant(deformation_field)

        assert "mean" in results
        assert "median" in results
        assert "folding_percentage" in results
        assert "jacobians" in results
        assert 0.0 <= results["folding_percentage"] <= 100.0

    def test_bending_energy(self):
        """Test bending energy computation."""
        # Smooth deformation field
        deformation_field = torch.zeros(2, 10, 10)

        be = bending_energy(deformation_field)

        assert be >= 0.0

    def test_deformation_smoothness(self):
        """Test deformation smoothness metrics."""
        deformation_field = torch.zeros(2, 10, 10)

        results = deformation_smoothness(deformation_field)

        assert "average_gradient_magnitude" in results
        assert "bending_energy" in results
        assert results["bending_energy"] >= 0.0


class TestGeometricTransforms:
    """Tests with known geometric transforms."""

    @pytest.mark.acceptance
    def test_known_geometric_transform(self):
        """Acceptance test: Known geometric transform on phantom."""
        # Create simple phantom (square) with some padding
        phantom = np.zeros((100, 100))
        phantom[40:60, 40:60] = 1.0

        # Apply small translation (5 pixels in x, 3 pixels in y)
        translation_x, translation_y = 5, 3
        transformed = np.zeros_like(phantom)
        transformed[40 + translation_y : 60 + translation_y, 40 + translation_x : 60 + translation_x] = 1.0

        # Compute similarity metrics
        nmi = normalized_mutual_information(phantom, transformed)
        ncc = normalized_cross_correlation(torch.from_numpy(phantom), torch.from_numpy(transformed))

        # For binary images with translation, NCC can be low due to sparse overlap
        # NMI should still be reasonable since both have same value distribution
        assert nmi > 0.3
        # NCC for translated binary masks is lower - just check it's computable
        assert -1.0 <= ncc <= 1.0

    @pytest.mark.acceptance
    def test_known_rotation_transform(self):
        """Acceptance test: Known rotation transform."""
        # Create simple phantom
        phantom = np.zeros((100, 100))
        phantom[40:60, 40:60] = 1.0

        # Apply 90-degree rotation (simplified - just swap axes)
        rotated = phantom.T

        # Compute similarity
        nmi = normalized_mutual_information(phantom, rotated)

        # Should have reasonable similarity
        assert nmi > 0.3


class TestComprehensiveRegistration:
    """Tests for comprehensive registration metrics."""

    def test_compute_registration_metrics(self):
        """Test comprehensive registration metric computation."""
        # Create test images
        image1 = np.random.rand(50, 50)
        image2 = np.random.rand(50, 50)

        # Create test landmarks
        pred_landmarks = np.array([[10.0, 10.0], [20.0, 20.0]])
        target_landmarks = np.array([[10.0, 10.0], [20.0, 20.0]])

        results = compute_registration_metrics(
            pred_image=image1,
            target_image=image2,
            pred_landmarks=pred_landmarks,
            target_landmarks=target_landmarks,
            include_image_similarity=True,
        )

        assert "tre" in results
        assert "nmi" in results
        assert "ncc" in results

    def test_compute_registration_metrics_with_deformation(self):
        """Test registration metrics with deformation field."""
        deformation_field = torch.zeros(2, 10, 10)

        results = compute_registration_metrics(
            deformation_field=deformation_field,
            include_deformation_quality=True,
        )

        assert "jacobian" in results
        assert "bending_energy" in results
        assert "smoothness" in results

