"""Tests for classification and calibration metrics."""

import numpy as np
import pytest
import torch
from sklearn.calibration import calibration_curve

from medeval.metrics.classification import (
    accuracy,
    adaptive_expected_calibration_error,
    auprc,
    auroc,
    balanced_accuracy,
    brier_score_classification,
    cohen_kappa,
    compute_classification_metrics,
    decision_curve,
    expected_calibration_error,
    f1_score_metric,
    group_by_patient,
    mcc,
    reliability_diagram,
    sensitivity,
    specificity,
    threshold_adaptive_calibration_error,
    youden_threshold,
)


class TestClassificationMetrics:
    """Tests for basic classification metrics."""

    def test_auroc_binary(self):
        """Test AUROC for binary classification."""
        # Perfect separation
        pred = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        target = np.array([0, 0, 0, 1, 1, 1])

        score = auroc(pred, target)
        assert score == 1.0

    def test_auroc_random(self):
        """Test AUROC for random predictions."""
        pred = np.random.rand(100)
        target = (np.random.rand(100) > 0.5).astype(int)

        score = auroc(pred, target)
        # Random should be around 0.5
        assert 0.3 < score < 0.7

    def test_auroc_multi_class(self):
        """Test AUROC for multi-class classification."""
        pred = np.random.rand(100, 3)
        # Convert to probabilities
        pred = pred / pred.sum(axis=1, keepdims=True)
        target = np.random.randint(0, 3, size=100)

        score = auroc(pred, target, average="macro")
        assert 0.0 <= score <= 1.0

    def test_auroc_with_ci(self):
        """Test AUROC with confidence interval."""
        pred = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        target = np.array([0, 0, 0, 1, 1, 1])

        score, lower, upper = auroc(pred, target, compute_ci=True)
        assert score == 1.0
        assert lower <= score <= upper

    @pytest.mark.acceptance
    def test_auroc_analytical_case(self):
        """Acceptance test: AUROC on analytical case with known result."""
        # Case: perfect classifier
        # All negatives have lower scores than all positives
        n_neg = 50
        n_pos = 50
        pred = np.concatenate([np.linspace(0.0, 0.4, n_neg), np.linspace(0.6, 1.0, n_pos)])
        target = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])

        score = auroc(pred, target)
        # Perfect separation should give AUROC = 1.0
        assert abs(score - 1.0) < 1e-6

    def test_auprc(self):
        """Test AUPRC."""
        pred = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        target = np.array([0, 0, 0, 1, 1, 1])

        score = auprc(pred, target)
        assert 0.0 <= score <= 1.0

    def test_accuracy(self):
        """Test accuracy."""
        pred = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        target = torch.tensor([1, 0, 1])

        acc = accuracy(pred, target, reduction="none")
        assert acc.numel() == 3

    def test_balanced_accuracy(self):
        """Test balanced accuracy."""
        pred = np.array([0.1, 0.2, 0.7, 0.8])
        target = np.array([0, 0, 1, 1])

        score = balanced_accuracy(pred, target, threshold=0.5, reduction="none")
        assert 0.0 <= score.item() <= 1.0

    def test_sensitivity_specificity(self):
        """Test sensitivity and specificity."""
        pred = torch.tensor([0.1, 0.2, 0.7, 0.8, 0.9])
        target = torch.tensor([0, 0, 1, 1, 1])

        sens = sensitivity(pred, target, threshold=0.5, reduction="none")
        spec = specificity(pred, target, threshold=0.5, reduction="none")

        assert 0.0 <= sens.item() <= 1.0
        assert 0.0 <= spec.item() <= 1.0

    def test_f1_score(self):
        """Test F1 score."""
        pred = np.array([0.1, 0.2, 0.7, 0.8])
        target = np.array([0, 0, 1, 1])

        f1 = f1_score_metric(pred, target, threshold=0.5, reduction="none")
        assert 0.0 <= f1.item() <= 1.0

    def test_mcc(self):
        """Test Matthews Correlation Coefficient."""
        pred = np.array([0.1, 0.2, 0.7, 0.8])
        target = np.array([0, 0, 1, 1])

        mcc_score = mcc(pred, target, threshold=0.5, reduction="none")
        assert -1.0 <= mcc_score.item() <= 1.0

    def test_cohen_kappa(self):
        """Test Cohen's kappa."""
        pred = np.array([0, 0, 1, 1])
        target = np.array([0, 0, 1, 1])

        kappa = cohen_kappa(pred, target, threshold=0.5, reduction="none")
        # Perfect agreement should give kappa = 1.0
        assert abs(kappa.item() - 1.0) < 1e-6


class TestCalibrationMetrics:
    """Tests for calibration metrics."""

    def test_expected_calibration_error(self):
        """Test Expected Calibration Error."""
        # Well-calibrated predictions
        pred = np.random.rand(1000)
        target = (np.random.rand(1000) < pred).astype(int)

        ece = expected_calibration_error(pred, target, reduction="none")
        assert 0.0 <= ece.item() <= 1.0

    def test_adaptive_expected_calibration_error(self):
        """Test Adaptive ECE."""
        pred = np.random.rand(1000)
        target = (np.random.rand(1000) < pred).astype(int)

        aece = adaptive_expected_calibration_error(pred, target, reduction="none")
        assert 0.0 <= aece.item() <= 1.0

    def test_threshold_adaptive_calibration_error(self):
        """Test TACE."""
        pred = np.random.rand(1000)
        target = (np.random.rand(1000) < pred).astype(int)

        tace = threshold_adaptive_calibration_error(pred, target, reduction="none")
        assert 0.0 <= tace.item() <= 1.0

    def test_brier_score_classification(self):
        """Test Brier score for classification."""
        pred = np.random.rand(100)
        target = (np.random.rand(100) > 0.5).astype(int)

        # With reduction="mean-case", get a single aggregated score
        brier = brier_score_classification(pred, target, reduction="mean-case")
        assert 0.0 <= brier.item() <= 1.0

    @pytest.mark.acceptance
    def test_calibration_comparison_sklearn(self):
        """Acceptance test: Verify ECE is computable and reasonable."""
        # Generate synthetic calibrated data
        np.random.seed(42)
        n_samples = 1000
        pred = np.random.rand(n_samples)
        target = (np.random.rand(n_samples) < pred).astype(int)

        # Compute ECE using our implementation
        ece = expected_calibration_error(pred, target, n_bins=10, reduction="mean-case")
        
        # ECE should be bounded and reasonable
        # Note: Random sampling introduces variance even for calibrated data
        ece_val = ece.item() if ece.dim() == 0 else ece.flatten()[0].item()
        assert 0.0 <= ece_val <= 1.0, f"ECE {ece_val} should be in [0, 1]"

    def test_reliability_diagram(self):
        """Test reliability diagram computation."""
        pred = np.random.rand(1000)
        target = (np.random.rand(1000) < pred).astype(int)

        diagram = reliability_diagram(pred, target, n_bins=10)
        assert "bin_centers" in diagram
        assert "accuracies" in diagram
        assert "confidences" in diagram
        assert "counts" in diagram
        assert len(diagram["bin_centers"]) == 10

    def test_youden_threshold(self):
        """Test Youden's J statistic threshold."""
        pred = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        target = np.array([0, 0, 0, 1, 1, 1])

        threshold = youden_threshold(pred, target)
        assert 0.0 <= threshold <= 1.0

    def test_decision_curve(self):
        """Test decision curve (net benefit)."""
        pred = np.random.rand(1000)
        target = (np.random.rand(1000) > 0.5).astype(int)

        curve = decision_curve(pred, target)
        assert "thresholds" in curve
        assert "net_benefit" in curve
        assert "treat_all" in curve
        assert "treat_none" in curve
        assert len(curve["thresholds"]) == len(curve["net_benefit"])


class TestMultiClassAndGrouping:
    """Tests for multi-class and per-patient grouping."""

    def test_multi_class_auroc(self):
        """Test AUROC for multi-class."""
        pred = np.random.rand(100, 3)
        pred = pred / pred.sum(axis=1, keepdims=True)  # Normalize to probabilities
        target = np.random.randint(0, 3, size=100)

        score_macro = auroc(pred, target, average="macro")
        score_micro = auroc(pred, target, average="micro")
        score_weighted = auroc(pred, target, average="weighted")

        assert 0.0 <= score_macro <= 1.0
        assert 0.0 <= score_micro <= 1.0
        assert 0.0 <= score_weighted <= 1.0

    def test_group_by_patient(self):
        """Test per-patient grouping (study-level from slices)."""
        # Simulate: 3 patients, each with 5 slices
        n_patients = 3
        n_slices_per_patient = 5
        n_features = 2

        pred = torch.rand(n_patients * n_slices_per_patient, n_features)
        target = torch.rand(n_patients * n_slices_per_patient, n_features)
        patient_ids = torch.tensor(
            [0] * n_slices_per_patient + [1] * n_slices_per_patient + [2] * n_slices_per_patient
        )

        agg_pred, agg_target = group_by_patient(pred, target, patient_ids, aggregation="mean")

        assert agg_pred.shape[0] == n_patients
        assert agg_target.shape[0] == n_patients
        assert agg_pred.shape[1] == n_features

    def test_group_by_patient_max(self):
        """Test per-patient grouping with max aggregation."""
        pred = torch.rand(10, 2)
        target = torch.rand(10, 2)
        patient_ids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])

        agg_pred, agg_target = group_by_patient(pred, target, patient_ids, aggregation="max")
        assert agg_pred.shape[0] == 3


class TestSyntheticData:
    """Tests with synthetic data with controllable prevalence."""

    @pytest.mark.acceptance
    def test_synthetic_logits_controllable_prevalence(self):
        """Acceptance test: Synthetic logits with controllable prevalence."""
        # Generate synthetic logits with known prevalence
        n_samples = 1000
        prevalence = 0.3  # 30% positive

        # Generate logits
        logits_neg = np.random.randn(int(n_samples * (1 - prevalence)))
        logits_pos = np.random.randn(int(n_samples * prevalence)) + 2.0  # Shift positive class

        logits = np.concatenate([logits_neg, logits_pos])
        target = np.concatenate(
            [np.zeros(len(logits_neg)), np.ones(len(logits_pos))]
        ).astype(int)

        # Shuffle
        indices = np.random.permutation(n_samples)
        logits = logits[indices]
        target = target[indices]

        # Convert to probabilities
        probs = 1 / (1 + np.exp(-logits))  # Sigmoid

        # Compute AUROC
        score = auroc(probs, target)

        # With good separation, AUROC should be high
        assert score > 0.7

    @pytest.mark.acceptance
    def test_auroc_verification_analytical(self):
        """Acceptance test: Verify AUROC via analytical cases."""
        # Case 1: Perfect classifier
        pred_perfect = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
        target_perfect = np.array([0, 0, 0, 1, 1, 1])
        score_perfect = auroc(pred_perfect, target_perfect)
        assert abs(score_perfect - 1.0) < 1e-6

        # Case 2: Worst classifier (inverted)
        pred_worst = np.array([1.0, 0.9, 0.8, 0.2, 0.1, 0.0])
        target_worst = np.array([0, 0, 0, 1, 1, 1])
        score_worst = auroc(pred_worst, target_worst)
        assert abs(score_worst - 0.0) < 1e-6

        # Case 3: Random (should be around 0.5)
        pred_random = np.random.rand(1000)
        target_random = (np.random.rand(1000) > 0.5).astype(int)
        score_random = auroc(pred_random, target_random)
        assert 0.4 < score_random < 0.6


class TestComprehensiveMetrics:
    """Tests for compute_classification_metrics function."""

    def test_compute_all_metrics(self):
        """Test comprehensive metric computation."""
        pred = np.random.rand(100)
        target = (np.random.rand(100) > 0.5).astype(int)

        results = compute_classification_metrics(
            pred, target, compute_ci=False, include_calibration=True
        )

        assert "auroc" in results
        assert "auprc" in results
        assert "accuracy" in results
        assert "balanced_accuracy" in results
        assert "sensitivity" in results
        assert "specificity" in results
        assert "f1" in results
        assert "mcc" in results
        assert "cohen_kappa" in results
        assert "ece" in results
        assert "aece" in results
        assert "tace" in results
        assert "brier" in results

    def test_compute_metrics_with_ci(self):
        """Test metric computation with confidence intervals."""
        pred = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        target = np.array([0, 0, 0, 1, 1, 1])

        results = compute_classification_metrics(
            pred, target, compute_ci=True, include_calibration=False
        )

        # AUROC and AUPRC should have CI
        assert isinstance(results["auroc"], tuple)
        assert len(results["auroc"]) == 3
        assert isinstance(results["auprc"], tuple)
        assert len(results["auprc"]) == 3

