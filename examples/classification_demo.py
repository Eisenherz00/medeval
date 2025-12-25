#!/usr/bin/env python
"""
Classification Metrics Demo
===========================

This script demonstrates how to use medeval for evaluating classification models.
It uses synthetic data with controllable prevalence and separation.

Usage:
    python classification_demo.py
"""

import numpy as np
import torch

# MedEval imports
from medeval.metrics.classification import (
    auroc,
    auprc,
    accuracy,
    balanced_accuracy,
    sensitivity,
    specificity,
    f1_score_metric,
    mcc,
    cohen_kappa,
    expected_calibration_error,
    reliability_diagram,
    youden_threshold,
    decision_curve,
    compute_classification_metrics,
)
from medeval.core.aggregate import aggregate_metrics, bootstrap_ci

# Optional visualization
try:
    from medeval.vis import (
        plot_roc_curve,
        plot_pr_curve,
        plot_reliability_diagram,
        plot_decision_curve,
        plot_confidence_histogram,
        plot_multiple_roc_curves,
    )
    from sklearn.metrics import roc_curve, precision_recall_curve
    HAS_VIS = True
except ImportError:
    HAS_VIS = False
    print("Note: matplotlib not installed, skipping visualizations")


def create_synthetic_classification_data(
    n_samples: int = 1000,
    prevalence: float = 0.3,
    separation: float = 2.0,
    calibration_error: float = 0.0,
    seed: int = 42,
) -> tuple:
    """
    Create synthetic classification data with controllable properties.
    
    Parameters
    ----------
    n_samples : int
        Total number of samples
    prevalence : float
        Proportion of positive class (0-1)
    separation : float
        Separation between class distributions (higher = easier task)
    calibration_error : float
        Amount of miscalibration to introduce
    seed : int
        Random seed
        
    Returns
    -------
    tuple
        (probabilities, labels)
    """
    np.random.seed(seed)
    
    n_pos = int(n_samples * prevalence)
    n_neg = n_samples - n_pos
    
    # Generate logits from Gaussian distributions
    logits_neg = np.random.randn(n_neg)
    logits_pos = np.random.randn(n_pos) + separation
    
    # Combine and create labels
    logits = np.concatenate([logits_neg, logits_pos])
    labels = np.concatenate([np.zeros(n_neg), np.ones(n_pos)]).astype(int)
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    logits = logits[indices]
    labels = labels[indices]
    
    # Convert to probabilities (sigmoid)
    probs = 1 / (1 + np.exp(-logits))
    
    # Introduce calibration error
    if calibration_error > 0:
        probs = probs ** (1 + calibration_error)  # Power transform
        probs = np.clip(probs, 0, 1)
    
    return probs, labels


def main():
    print("=" * 60)
    print("MedEval Classification Metrics Demo")
    print("=" * 60)
    
    # Create well-calibrated data
    print("\n1. Creating synthetic classification data...")
    n_samples = 1000
    prevalence = 0.3
    
    probs_good, labels = create_synthetic_classification_data(
        n_samples=n_samples,
        prevalence=prevalence,
        separation=2.0,  # Good separation
        calibration_error=0.0,
        seed=42,
    )
    
    probs_poor, _ = create_synthetic_classification_data(
        n_samples=n_samples,
        prevalence=prevalence,
        separation=1.0,  # Poor separation
        calibration_error=0.3,  # Miscalibrated
        seed=42,
    )
    
    print(f"   Created {n_samples} samples with {prevalence:.0%} prevalence")
    
    # Compute AUROC
    print("\n2. Computing classification metrics...")
    
    # Good model
    auroc_good = auroc(probs_good, labels)
    auprc_good = auprc(probs_good, labels)
    
    # Poor model
    auroc_poor = auroc(probs_poor, labels)
    auprc_poor = auprc(probs_poor, labels)
    
    print(f"   Good Model: AUROC={auroc_good:.4f}, AUPRC={auprc_good:.4f}")
    print(f"   Poor Model: AUROC={auroc_poor:.4f}, AUPRC={auprc_poor:.4f}")
    
    # Compute with confidence intervals
    print("\n3. Computing metrics with confidence intervals...")
    auroc_ci = auroc(probs_good, labels, compute_ci=True, confidence=0.95)
    print(f"   AUROC (Good Model): {auroc_ci[0]:.4f} [{auroc_ci[1]:.4f}, {auroc_ci[2]:.4f}]")
    
    # Calibration metrics
    print("\n4. Computing calibration metrics...")
    ece_good = expected_calibration_error(probs_good, labels, n_bins=10)
    ece_poor = expected_calibration_error(probs_poor, labels, n_bins=10)
    
    print(f"   ECE (Good Model): {ece_good.item():.4f}")
    print(f"   ECE (Poor Model): {ece_poor.item():.4f}")
    
    # Find optimal threshold
    print("\n5. Finding optimal threshold...")
    optimal_thresh = youden_threshold(probs_good, labels)
    print(f"   Optimal threshold (Youden): {optimal_thresh:.4f}")
    
    # Compute metrics at optimal threshold
    preds_binary = (probs_good > optimal_thresh).astype(int)
    sens = sensitivity(torch.tensor(probs_good), torch.tensor(labels), threshold=optimal_thresh)
    spec = specificity(torch.tensor(probs_good), torch.tensor(labels), threshold=optimal_thresh)
    
    print(f"   At threshold {optimal_thresh:.2f}:")
    print(f"     Sensitivity: {sens.item():.4f}")
    print(f"     Specificity: {spec.item():.4f}")
    
    # Comprehensive metrics
    print("\n6. Computing comprehensive metrics...")
    all_metrics = compute_classification_metrics(
        probs_good, labels,
        compute_ci=False,
        include_calibration=True,
    )
    
    print("   Available metrics:", list(all_metrics.keys()))
    
    # Decision curve
    print("\n7. Computing decision curve...")
    dc_results = decision_curve(probs_good, labels)
    
    # Find threshold range where model has positive net benefit
    net_benefit = dc_results["net_benefit"]
    treat_all = dc_results["treat_all"]
    thresholds = dc_results["thresholds"]
    
    model_better = net_benefit > treat_all
    if np.any(model_better):
        useful_range = (thresholds[model_better].min(), thresholds[model_better].max())
        print(f"   Model useful at thresholds: {useful_range[0]:.2f} to {useful_range[1]:.2f}")
    
    # Visualization
    if HAS_VIS:
        print("\n8. Creating visualizations...")
        
        # ROC curves comparison
        fpr_good, tpr_good, _ = roc_curve(labels, probs_good)
        fpr_poor, tpr_poor, _ = roc_curve(labels, probs_poor)
        
        fig, ax = plot_multiple_roc_curves([
            {"fpr": fpr_good, "tpr": tpr_good, "auc": auroc_good, "label": "Good Model"},
            {"fpr": fpr_poor, "tpr": tpr_poor, "auc": auroc_poor, "label": "Poor Model"},
        ], title="ROC Curves Comparison")
        fig.savefig("roc_comparison.png", dpi=150, bbox_inches="tight")
        print("   Saved: roc_comparison.png")
        
        # Reliability diagram
        rel_diagram = reliability_diagram(probs_poor, labels, n_bins=10)
        fig, ax = plot_reliability_diagram(
            bin_centers=rel_diagram["bin_centers"],
            accuracies=rel_diagram["accuracies"],
            confidences=rel_diagram["confidences"],
            counts=rel_diagram["counts"],
            ece=ece_poor.item(),
            title="Reliability Diagram (Poor Model)",
        )
        fig.savefig("reliability_diagram.png", dpi=150, bbox_inches="tight")
        print("   Saved: reliability_diagram.png")
        
        # Decision curve
        fig, ax = plot_decision_curve(
            thresholds=dc_results["thresholds"],
            net_benefit=dc_results["net_benefit"],
            treat_all=dc_results["treat_all"],
            treat_none=dc_results["treat_none"],
            label="Model",
            title="Decision Curve Analysis",
        )
        fig.savefig("decision_curve.png", dpi=150, bbox_inches="tight")
        print("   Saved: decision_curve.png")
        
        # Confidence histogram
        fig, ax = plot_confidence_histogram(
            probs_good,
            labels=labels,
            title="Confidence Distribution by Class",
        )
        fig.savefig("confidence_histogram.png", dpi=150, bbox_inches="tight")
        print("   Saved: confidence_histogram.png")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

