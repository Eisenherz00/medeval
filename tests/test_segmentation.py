"""Tests for segmentation metrics."""

import numpy as np
import pytest
import torch

from medeval.metrics.segmentation import (
    average_symmetric_surface_distance,
    brier_score,
    compute_segmentation_metrics,
    dice_score,
    hausdorff_distance,
    hausdorff_distance_95,
    jaccard_index,
    precision_score,
    recall_score,
    soft_dice_score,
    surface_dice,
    threshold_sweep,
    volumetric_similarity,
)


class TestOverlapMetrics:
    """Tests for overlap metrics (Dice, Jaccard, Precision, Recall, VS)."""

    def test_dice_perfect_match(self):
        """Test Dice score for perfect match."""
        pred = torch.ones(1, 10, 10)
        target = torch.ones(1, 10, 10)

        dice = dice_score(pred, target, reduction="none")
        assert torch.allclose(dice, torch.tensor(1.0))

    def test_dice_no_overlap(self):
        """Test Dice score for no overlap."""
        pred = torch.zeros(1, 10, 10)
        target = torch.ones(1, 10, 10)

        dice = dice_score(pred, target, reduction="none")
        assert torch.allclose(dice, torch.tensor(0.0))

    def test_dice_partial_overlap(self):
        """Test Dice score for partial overlap."""
        pred = torch.zeros(1, 10, 10)
        pred[0, 0:5, 0:5] = 1.0
        target = torch.zeros(1, 10, 10)
        target[0, 0:7, 0:7] = 1.0

        dice = dice_score(pred, target, reduction="none")
        # Intersection: 5x5 = 25, Union: 5x5 + 7x7 - 25 = 25 + 49 - 25 = 49
        # Dice = 2 * 25 / (25 + 49) = 50 / 74 ≈ 0.676
        assert 0.6 < dice.item() < 0.7

    def test_dice_empty_prediction(self):
        """Test Dice score with empty prediction (TP=0 edge case)."""
        pred = torch.zeros(1, 10, 10)
        target = torch.ones(1, 10, 10)

        dice = dice_score(pred, target, reduction="none")
        assert torch.allclose(dice, torch.tensor(0.0))

    def test_dice_empty_target(self):
        """Test Dice score with empty target."""
        pred = torch.ones(1, 10, 10)
        target = torch.zeros(1, 10, 10)

        dice = dice_score(pred, target, reduction="none")
        assert torch.allclose(dice, torch.tensor(0.0))

    def test_jaccard_index(self):
        """Test Jaccard index (IoU)."""
        pred = torch.zeros(1, 10, 10)
        pred[0, 0:5, 0:5] = 1.0
        target = torch.zeros(1, 10, 10)
        target[0, 0:7, 0:7] = 1.0

        jaccard = jaccard_index(pred, target, reduction="none")
        # Intersection: 25, Union: 49
        # IoU = 25 / 49 ≈ 0.510
        assert 0.5 < jaccard.item() < 0.52

    def test_precision_recall(self):
        """Test precision and recall."""
        pred = torch.zeros(1, 10, 10)
        pred[0, 0:5, 0:5] = 1.0  # 25 voxels
        target = torch.zeros(1, 10, 10)
        target[0, 0:7, 0:7] = 1.0  # 49 voxels

        precision = precision_score(pred, target, reduction="none")
        recall = recall_score(pred, target, reduction="none")

        # TP = 25, FP = 0, FN = 24
        # Precision = 25 / 25 = 1.0
        # Recall = 25 / 49 ≈ 0.510
        assert torch.allclose(precision, torch.tensor(1.0))
        assert 0.5 < recall.item() < 0.52

    def test_volumetric_similarity(self):
        """Test volumetric similarity."""
        pred = torch.zeros(1, 10, 10)
        pred[0, 0:5, 0:5] = 1.0  # Volume = 25
        target = torch.zeros(1, 10, 10)
        target[0, 0:7, 0:7] = 1.0  # Volume = 49

        vs = volumetric_similarity(pred, target, reduction="none")
        # VS = 1 - |25 - 49| / (25 + 49) = 1 - 24/74 ≈ 0.676
        assert 0.67 < vs.item() < 0.68

    def test_multi_class_dice(self):
        """Test Dice score for multi-class segmentation."""
        # 3 classes: background (0), class1 (1), class2 (2)
        pred = torch.zeros(1, 3, 10, 10, dtype=torch.float32)
        pred[0, 1, 0:5, 0:5] = 1.0  # Class 1 in top-left
        pred[0, 2, 5:10, 5:10] = 1.0  # Class 2 in bottom-right

        target = torch.zeros(1, 3, 10, 10, dtype=torch.float32)
        target[0, 1, 0:7, 0:7] = 1.0  # Class 1 larger
        target[0, 2, 3:10, 3:10] = 1.0  # Class 2 larger

        dice = dice_score(pred, target, reduction="mean-class")
        # Should compute per-class Dice and average
        assert dice.numel() == 1
        assert 0.0 < dice.item() < 1.0

    def test_ignore_index(self):
        """Test metrics with ignore_index."""
        pred = torch.zeros(1, 10, 10)
        pred[0, 0:5, 0:5] = 1.0
        target = torch.zeros(1, 10, 10)
        target[0, 0:7, 0:7] = 1.0
        target[0, 8:10, 8:10] = 255  # Ignore region

        dice = dice_score(pred, target, ignore_index=255, reduction="none")
        # Should ignore the 255 region
        assert 0.0 < dice.item() < 1.0

    def test_reduction_strategies(self):
        """Test different reduction strategies."""
        pred = torch.rand(3, 1, 10, 10) > 0.5
        target = torch.rand(3, 1, 10, 10) > 0.5

        dice_none = dice_score(pred, target, reduction="none")
        dice_mean_case = dice_score(pred, target, reduction="mean-case")
        dice_global = dice_score(pred, target, reduction="global")

        assert dice_none.shape[0] == 3  # Per-case
        assert dice_mean_case.numel() == 1
        assert dice_global.numel() == 1


class TestSurfaceMetrics:
    """Tests for surface-based metrics."""

    @pytest.mark.acceptance
    def test_hausdorff_distance_toy_mask(self):
        """Acceptance test: HD on toy mask with known distance."""
        # Create two squares with known distance
        pred = np.zeros((20, 20), dtype=np.float32)
        pred[5:10, 5:10] = 1.0  # Square at indices [5,9] in both dims

        target = np.zeros((20, 20), dtype=np.float32)
        target[15:20, 15:20] = 1.0  # Square at indices [15,19] in both dims

        # Hausdorff distance is MAX of min distances from surface points
        # Furthest pred point (5,5) to nearest target (15,15) = sqrt((15-5)^2 + (15-5)^2) = sqrt(200) ≈ 14.14
        pred_tensor = torch.from_numpy(pred).unsqueeze(0)
        target_tensor = torch.from_numpy(target).unsqueeze(0)

        hd = hausdorff_distance(pred_tensor, target_tensor, reduction="none")
        # Should be approximately sqrt(200) ≈ 14.14
        assert 14.0 < hd.item() < 14.5

    @pytest.mark.acceptance
    def test_hausdorff_distance_spacing_aware(self):
        """Acceptance test: HD with spacing (physical units)."""
        # Create two squares
        pred = np.zeros((20, 20), dtype=np.float32)
        pred[5:10, 5:10] = 1.0
        target = np.zeros((20, 20), dtype=np.float32)
        target[15:20, 15:20] = 1.0

        # With spacing (2.0, 2.0), physical distance is computed with scaled coords
        # Physical HD = sqrt((2*10)^2 + (2*10)^2) = sqrt(800) ≈ 28.28
        spacing = (2.0, 2.0)
        pred_tensor = torch.from_numpy(pred).unsqueeze(0)
        target_tensor = torch.from_numpy(target).unsqueeze(0)

        hd = hausdorff_distance(pred_tensor, target_tensor, spacing=spacing, reduction="none")
        # Physical distance ≈ sqrt(800) ≈ 28.28
        assert 28.0 < hd.item() < 29.0

    def test_hausdorff_distance_95(self):
        """Test HD95 (95th percentile Hausdorff distance)."""
        pred = torch.zeros(1, 20, 20)
        pred[0, 5:10, 5:10] = 1.0
        target = torch.zeros(1, 20, 20)
        target[0, 15:20, 15:20] = 1.0

        hd95 = hausdorff_distance_95(pred, target, reduction="none")
        # HD95 should be <= HD
        hd = hausdorff_distance(pred, target, reduction="none")
        assert hd95.item() <= hd.item()

    def test_assd(self):
        """Test Average Symmetric Surface Distance."""
        pred = torch.zeros(1, 20, 20)
        pred[0, 5:10, 5:10] = 1.0
        target = torch.zeros(1, 20, 20)
        target[0, 15:20, 15:20] = 1.0

        assd = average_symmetric_surface_distance(pred, target, reduction="none")
        # ASSD should be finite and positive
        assert np.isfinite(assd.item())
        assert assd.item() > 0

    def test_surface_dice(self):
        """Test Surface Dice at tolerance."""
        pred = torch.zeros(1, 20, 20)
        pred[0, 5:10, 5:10] = 1.0
        target = torch.zeros(1, 20, 20)
        target[0, 6:11, 6:11] = 1.0  # Slightly offset

        # With tolerance 2.0, should have high surface dice
        surface_dice_score = surface_dice(pred, target, tolerance=2.0, reduction="none")
        assert 0.0 < surface_dice_score.item() <= 1.0

    def test_surface_metrics_empty_sets(self):
        """Test surface metrics with empty predictions/targets.

        Empty-set policy (surface metrics):
        - both empty => distance 0.0 (perfect match)
        - one empty  => NaN (undefined; caller/aggregator should be NaN-aware)
        """
        # one empty
        pred = torch.zeros(1, 10, 10)
        target = torch.ones(1, 10, 10)

        hd = hausdorff_distance(pred, target, reduction="none")
        hd95 = hausdorff_distance_95(pred, target, reduction="none")
        assd = average_symmetric_surface_distance(pred, target, reduction="none")

        assert torch.isnan(hd) or torch.isinf(hd)
        assert torch.isnan(hd95) or torch.isinf(hd95)
        assert torch.isnan(assd) or torch.isinf(assd)

        # both empty
        pred2 = torch.zeros(1, 10, 10)
        target2 = torch.zeros(1, 10, 10)

        hd2 = hausdorff_distance(pred2, target2, reduction="none")
        hd95_2 = hausdorff_distance_95(pred2, target2, reduction="none")
        assd2 = average_symmetric_surface_distance(pred2, target2, reduction="none")
        sd2 = surface_dice(pred2, target2, tolerance=1.0, reduction="none")

        assert torch.allclose(hd2, torch.tensor(0.0))
        assert torch.allclose(hd95_2, torch.tensor(0.0))
        assert torch.allclose(assd2, torch.tensor(0.0))
        assert torch.allclose(sd2, torch.tensor(1.0))
    def test_surface_metrics_multiclass_ignore_index(self):
        """Test that ignore_index skips only the specified class channel (not all classes)."""
        # 3-class one-hot-ish tensor (B, C, H, W)
        pred = torch.zeros(1, 3, 20, 20)
        target = torch.zeros(1, 3, 20, 20)

        # class 1: slightly shifted squares
        pred[0, 1, 5:10, 5:10] = 1.0
        target[0, 1, 6:11, 6:11] = 1.0

        # class 2: totally different squares (would increase distances if included)
        pred[0, 2, 1:3, 1:3] = 1.0
        target[0, 2, 15:18, 15:18] = 1.0

        # If we ignore class 2, metrics should be driven mostly by class 1 (finite).
        spacing = (1.0, 1.0)
        hd = hausdorff_distance(pred, target, spacing=spacing, ignore_index=2, reduction="none")
        assd = average_symmetric_surface_distance(pred, target, spacing=spacing, ignore_index=2, reduction="none")
        sd = surface_dice(pred, target, tolerance=2.0, spacing=spacing, ignore_index=2, reduction="none")

        assert torch.isfinite(hd).item()
        assert torch.isfinite(assd).item()
        assert 0.0 <= sd.item() <= 1.0

    def test_surface_metrics_3d(self):
        """Test surface metrics on 3D volumes."""
        pred = torch.zeros(1, 10, 10, 10)
        pred[0, 2:5, 2:5, 2:5] = 1.0
        target = torch.zeros(1, 10, 10, 10)
        target[0, 6:9, 6:9, 6:9] = 1.0

        spacing = (1.0, 1.0, 1.0)
        hd = hausdorff_distance(pred, target, spacing=spacing, reduction="none")
        assert np.isfinite(hd.item())
        assert hd.item() > 0


class TestCalibrationMetrics:
    """Tests for calibration metrics."""

    def test_soft_dice(self):
        """Test soft Dice score for probabilistic predictions."""
        pred = torch.rand(1, 10, 10)  # Probabilistic
        target = (torch.rand(1, 10, 10) > 0.5).float()  # Binary

        soft_dice = soft_dice_score(pred, target, reduction="none")
        assert 0.0 <= soft_dice.item() <= 1.0

    def test_brier_score(self):
        """Test Brier score (per-voxel MSE)."""
        pred = torch.rand(1, 10, 10)  # Probabilistic
        target = (torch.rand(1, 10, 10) > 0.5).float()  # Binary

        brier = brier_score(pred, target, reduction="none")
        # Brier score should be in [0, 1]
        assert 0.0 <= brier.item() <= 1.0

    def test_brier_perfect_calibration(self):
        """Test Brier score with perfect calibration."""
        pred = torch.tensor([[[0.0, 0.5, 1.0]]])
        target = torch.tensor([[[0.0, 0.5, 1.0]]])

        brier = brier_score(pred, target, reduction="none")
        assert torch.allclose(brier, torch.tensor(0.0))

    def test_threshold_sweep(self):
        """Test threshold sweeping."""
        pred = torch.rand(1, 10, 10)
        target = (torch.rand(1, 10, 10) > 0.5).float()

        results = threshold_sweep(pred, target, thresholds=[0.3, 0.5, 0.7], metric_fn="dice")
        assert "thresholds" in results
        assert "scores" in results
        assert len(results["thresholds"]) == 3
        assert results["scores"].shape[0] == 3


class TestEdgeCases:
    """Tests for edge cases."""

    def test_small_objects(self):
        """Test metrics on small objects (1-2 voxels)."""
        pred = torch.zeros(1, 10, 10)
        pred[0, 5, 5] = 1.0  # Single voxel
        target = torch.zeros(1, 10, 10)
        target[0, 5, 5] = 1.0

        dice = dice_score(pred, target, reduction="none")
        assert torch.allclose(dice, torch.tensor(1.0))

    def test_class_imbalance(self):
        """Test metrics with severe class imbalance."""
        pred = torch.zeros(1, 100, 100)
        pred[0, 45:55, 45:55] = 1.0  # Small object
        target = torch.zeros(1, 100, 100)
        target[0, 45:55, 45:55] = 1.0

        dice = dice_score(pred, target, reduction="none")
        # Should handle small objects correctly
        assert torch.allclose(dice, torch.tensor(1.0))

    def test_all_background(self):
        """Test metrics when both pred and target are all background."""
        pred = torch.zeros(1, 10, 10)
        target = torch.zeros(1, 10, 10)

        dice = dice_score(pred, target, reduction="none")
        # Empty sets: both empty = perfect match = 1.0
        assert torch.allclose(dice, torch.tensor(1.0))

    def test_2d_3d_support(self):
        """Test that metrics work for both 2D and 3D."""
        # 2D
        pred_2d = torch.zeros(1, 10, 10)
        target_2d = torch.zeros(1, 10, 10)
        pred_2d[0, 0:5, 0:5] = 1.0
        target_2d[0, 0:5, 0:5] = 1.0

        dice_2d = dice_score(pred_2d, target_2d, reduction="none")
        assert torch.allclose(dice_2d, torch.tensor(1.0))

        # 3D
        pred_3d = torch.zeros(1, 10, 10, 10)
        target_3d = torch.zeros(1, 10, 10, 10)
        pred_3d[0, 0:5, 0:5, 0:5] = 1.0
        target_3d[0, 0:5, 0:5, 0:5] = 1.0

        dice_3d = dice_score(pred_3d, target_3d, reduction="none")
        assert torch.allclose(dice_3d, torch.tensor(1.0))


class TestComprehensiveMetrics:
    """Tests for compute_segmentation_metrics function."""

    def test_compute_all_metrics(self):
        """Test comprehensive metric computation."""
        pred = torch.rand(1, 10, 10) > 0.5
        target = torch.rand(1, 10, 10) > 0.5

        results = compute_segmentation_metrics(
            pred, target, spacing=(1.0, 1.0), include_surface=True, include_calibration=True
        )

        assert "dice" in results
        assert "jaccard" in results
        assert "precision" in results
        assert "recall" in results
        assert "volumetric_similarity" in results
        assert "hausdorff" in results
        assert "hausdorff_95" in results
        assert "assd" in results
        assert "surface_dice" in results
        assert "soft_dice" in results
        assert "brier" in results

        # Also exercise 3D surface path on a tiny volume
        pred3 = (torch.rand(1, 8, 8, 8) > 0.5).float()
        tgt3 = (torch.rand(1, 8, 8, 8) > 0.5).float()
        results3 = compute_segmentation_metrics(
            pred3, tgt3, spacing=(1.0, 1.0, 1.0), include_surface=True, include_calibration=False
        )
        assert "hausdorff" in results3
        assert "assd" in results3
        assert "surface_dice" in results3

    def test_compute_metrics_no_surface(self):
        """Test metric computation without surface metrics."""
        pred = torch.rand(1, 10, 10) > 0.5
        target = torch.rand(1, 10, 10) > 0.5

        results = compute_segmentation_metrics(
            pred, target, include_surface=False, include_calibration=False
        )

        assert "dice" in results
        assert "hausdorff" not in results
        assert "soft_dice" not in results

    def test_per_class_metrics(self):
        """Test per-class metric computation."""
        pred = torch.zeros(1, 3, 10, 10, dtype=torch.float32)
        pred[0, 1, 0:5, 0:5] = 1.0
        target = torch.zeros(1, 3, 10, 10, dtype=torch.float32)
        target[0, 1, 0:5, 0:5] = 1.0

        results = compute_segmentation_metrics(
            pred, target, reduction="mean-case", per_class=True
        )

        # Should return per-class metrics
        assert "dice" in results
        assert results["dice"].numel() >= 1

