"""Tests for detection metrics."""

import numpy as np
import pytest
import torch

from medeval.metrics.detection import (
    average_recall,
    box_iou_2d,
    box_iou_3d,
    froc,
    hungarian_matching,
    instance_segmentation_matching,
    mean_average_precision,
)


class TestBoxIoU:
    """Tests for bounding box IoU computation."""

    def test_box_iou_2d_perfect_match(self):
        """Test 2D box IoU for perfect match."""
        boxes1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        boxes2 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])

        iou = box_iou_2d(boxes1, boxes2)
        assert torch.allclose(iou, torch.tensor([[1.0]]))

    def test_box_iou_2d_no_overlap(self):
        """Test 2D box IoU for no overlap."""
        boxes1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        boxes2 = torch.tensor([[20.0, 20.0, 30.0, 30.0]])

        iou = box_iou_2d(boxes1, boxes2)
        assert torch.allclose(iou, torch.tensor([[0.0]]))

    def test_box_iou_2d_partial_overlap(self):
        """Test 2D box IoU for partial overlap."""
        boxes1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        boxes2 = torch.tensor([[5.0, 5.0, 15.0, 15.0]])

        iou = box_iou_2d(boxes1, boxes2)
        # Intersection: 5x5 = 25, Union: 100 + 100 - 25 = 175
        # IoU = 25/175 ≈ 0.143
        assert 0.14 < iou.item() < 0.15

    def test_box_iou_2d_xywh_format(self):
        """Test 2D box IoU with xywh format."""
        boxes1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])  # xywh
        boxes2 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])  # xywh

        iou = box_iou_2d(boxes1, boxes2, format="xywh")
        assert torch.allclose(iou, torch.tensor([[1.0]]))

    def test_box_iou_3d(self):
        """Test 3D box IoU."""
        boxes1 = torch.tensor([[0.0, 0.0, 0.0, 10.0, 10.0, 10.0]])  # xyzxyz
        boxes2 = torch.tensor([[0.0, 0.0, 0.0, 10.0, 10.0, 10.0]])

        iou = box_iou_3d(boxes1, boxes2)
        assert torch.allclose(iou, torch.tensor([[1.0]]))

    def test_box_iou_3d_no_overlap(self):
        """Test 3D box IoU for no overlap."""
        boxes1 = torch.tensor([[0.0, 0.0, 0.0, 10.0, 10.0, 10.0]])
        boxes2 = torch.tensor([[20.0, 20.0, 20.0, 30.0, 30.0, 30.0]])

        iou = box_iou_3d(boxes1, boxes2)
        assert torch.allclose(iou, torch.tensor([[0.0]]))


class TestMeanAveragePrecision:
    """Tests for mean Average Precision."""

    def test_map_simple(self):
        """Test mAP computation on simple case."""
        # Single image, single class
        pred_boxes = [torch.tensor([[10.0, 10.0, 20.0, 20.0]])]
        pred_scores = [torch.tensor([0.9])]
        pred_labels = [torch.tensor([0])]
        target_boxes = [torch.tensor([[10.0, 10.0, 20.0, 20.0]])]
        target_labels = [torch.tensor([0])]

        results = mean_average_precision(
            pred_boxes, pred_scores, pred_labels, target_boxes, target_labels, iou_thresholds=np.array([0.5])
        )

        assert "mAP@0.50" in results
        assert results["mAP@0.50"] > 0.9  # Should be high for perfect match

    def test_map_multiple_images(self):
        """Test mAP with multiple images."""
        pred_boxes = [
            torch.tensor([[10.0, 10.0, 20.0, 20.0]]),
            torch.tensor([[30.0, 30.0, 40.0, 40.0]]),
        ]
        pred_scores = [torch.tensor([0.9]), torch.tensor([0.8])]
        pred_labels = [torch.tensor([0]), torch.tensor([0])]
        target_boxes = [
            torch.tensor([[10.0, 10.0, 20.0, 20.0]]),
            torch.tensor([[30.0, 30.0, 40.0, 40.0]]),
        ]
        target_labels = [torch.tensor([0]), torch.tensor([0])]

        results = mean_average_precision(
            pred_boxes, pred_scores, pred_labels, target_boxes, target_labels, iou_thresholds=np.array([0.5])
        )

        assert "mAP@0.50" in results

    @pytest.mark.acceptance
    def test_map_coco_style(self):
        """Acceptance test: COCO-style mAP computation."""
        # Simulate COCO-style detection results
        # Multiple images, multiple classes, multiple detections per image
        pred_boxes = [
            torch.tensor([[10.0, 10.0, 20.0, 20.0], [30.0, 30.0, 40.0, 40.0]]),
            torch.tensor([[50.0, 50.0, 60.0, 60.0]]),
        ]
        pred_scores = [torch.tensor([0.9, 0.7]), torch.tensor([0.8])]
        pred_labels = [torch.tensor([0, 1]), torch.tensor([0])]
        target_boxes = [
            torch.tensor([[10.0, 10.0, 20.0, 20.0], [30.0, 30.0, 40.0, 40.0]]),
            torch.tensor([[50.0, 50.0, 60.0, 60.0]]),
        ]
        target_labels = [torch.tensor([0, 1]), torch.tensor([0])]

        results = mean_average_precision(
            pred_boxes,
            pred_scores,
            pred_labels,
            target_boxes,
            target_labels,
            iou_thresholds=np.arange(0.50, 1.0, 0.05),
        )

        assert "mAP@[.50:.95]" in results
        assert 0.0 <= results["mAP@[.50:.95]"] <= 1.0

    def test_map_class_aware(self):
        """Test class-aware mAP."""
        pred_boxes = [torch.tensor([[10.0, 10.0, 20.0, 20.0]])]
        pred_scores = [torch.tensor([0.9])]
        pred_labels = [torch.tensor([0])]
        target_boxes = [torch.tensor([[10.0, 10.0, 20.0, 20.0]])]
        target_labels = [torch.tensor([0])]

        results = mean_average_precision(
            pred_boxes,
            pred_scores,
            pred_labels,
            target_boxes,
            target_labels,
            iou_thresholds=np.array([0.5]),
            class_aware=True,
        )

        assert "mAP@0.50" in results


class TestFROC:
    """Tests for FROC (Free-Response ROC)."""

    def test_froc_simple(self):
        """Test FROC computation."""
        pred_boxes = [torch.tensor([[10.0, 10.0, 20.0, 20.0]])]
        pred_scores = [torch.tensor([0.9])]
        pred_labels = [torch.tensor([0])]
        target_boxes = [torch.tensor([[10.0, 10.0, 20.0, 20.0]])]
        target_labels = [torch.tensor([0])]

        results = froc(pred_boxes, pred_scores, pred_labels, target_boxes, target_labels)

        assert "sensitivity" in results
        assert "avg_fps_per_image" in results
        assert "thresholds" in results
        assert len(results["sensitivity"]) == len(results["avg_fps_per_image"])


class TestAverageRecall:
    """Tests for Average Recall."""

    def test_average_recall(self):
        """Test Average Recall computation."""
        pred_boxes = [torch.tensor([[10.0, 10.0, 20.0, 20.0]])]
        pred_scores = [torch.tensor([0.9])]
        target_boxes = [torch.tensor([[10.0, 10.0, 20.0, 20.0]])]

        ar = average_recall(pred_boxes, pred_scores, target_boxes)

        assert 0.0 <= ar <= 1.0


class TestHungarianMatching:
    """Tests for Hungarian algorithm matching."""

    def test_hungarian_matching(self):
        """Test Hungarian matching."""
        # Cost matrix: 2x2
        cost_matrix = torch.tensor([[1.0, 2.0], [3.0, 1.0]])

        row_indices, col_indices = hungarian_matching(cost_matrix)

        assert len(row_indices) == len(col_indices)
        assert len(row_indices) == 2

    def test_hungarian_maximize(self):
        """Test Hungarian matching with maximization."""
        # Similarity matrix (to maximize)
        similarity_matrix = torch.tensor([[0.9, 0.3], [0.2, 0.8]])

        row_indices, col_indices = hungarian_matching(similarity_matrix, maximize=True)

        assert len(row_indices) == len(col_indices)


class TestInstanceSegmentationMatching:
    """Tests for instance segmentation matching."""

    def test_instance_matching(self):
        """Test instance segmentation matching."""
        # Simple 2D masks
        pred_masks = [torch.zeros(2, 10, 10)]
        pred_masks[0][0, 2:5, 2:5] = 1.0  # First instance
        pred_masks[0][1, 7:9, 7:9] = 1.0  # Second instance

        pred_scores = [torch.tensor([0.9, 0.8])]

        target_masks = [torch.zeros(2, 10, 10)]
        target_masks[0][0, 2:5, 2:5] = 1.0  # First target
        target_masks[0][1, 7:9, 7:9] = 1.0  # Second target

        results = instance_segmentation_matching(pred_masks, pred_scores, target_masks)

        assert "matches" in results
        assert "precision" in results
        assert "recall" in results
        assert results["precision"] > 0.5
        assert results["recall"] > 0.5

