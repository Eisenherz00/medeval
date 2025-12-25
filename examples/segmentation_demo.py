#!/usr/bin/env python
"""
Segmentation Metrics Demo
=========================

This script demonstrates how to use medeval for evaluating segmentation models.
It uses synthetic data to showcase the available metrics and visualization tools.

Usage:
    python segmentation_demo.py
"""

import numpy as np
import torch

# MedEval imports
from medeval.metrics.segmentation import (
    dice_score,
    jaccard_index,
    hausdorff_distance,
    hausdorff_distance_95,
    average_symmetric_surface_distance,
    surface_dice,
    compute_segmentation_metrics,
)
from medeval.core.aggregate import aggregate_metrics, bootstrap_ci
from medeval.core.containers import MedicalPrediction

# Optional visualization (requires matplotlib)
try:
    from medeval.vis import (
        plot_segmentation_overlay,
        plot_prediction_comparison,
        plot_metric_distribution,
        plot_error_histogram,
    )
    HAS_VIS = True
except ImportError:
    HAS_VIS = False
    print("Note: matplotlib not installed, skipping visualizations")


def create_synthetic_3d_data(
    shape: tuple = (64, 64, 64),
    n_samples: int = 10,
    noise_level: float = 0.1,
    seed: int = 42,
) -> tuple:
    """
    Create synthetic 3D segmentation data.
    
    Returns predictions and ground truth masks with controlled overlap.
    """
    np.random.seed(seed)
    
    predictions = []
    targets = []
    
    for i in range(n_samples):
        # Create ground truth: sphere in center
        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
        center = np.array(shape) // 2
        radius = min(shape) // 4 + np.random.randint(-5, 5)
        
        # Ground truth sphere
        dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
        target = (dist <= radius).astype(np.float32)
        
        # Prediction: sphere with random offset and size variation
        offset = np.random.randint(-3, 4, size=3)
        pred_center = center + offset
        pred_radius = radius + np.random.randint(-3, 4)
        
        pred_dist = np.sqrt(
            (z - pred_center[0])**2 + 
            (y - pred_center[1])**2 + 
            (x - pred_center[2])**2
        )
        pred = (pred_dist <= pred_radius).astype(np.float32)
        
        # Add some noise
        noise = np.random.rand(*shape) < noise_level
        pred = np.logical_xor(pred, noise).astype(np.float32)
        
        predictions.append(pred)
        targets.append(target)
    
    return np.stack(predictions), np.stack(targets)


def main():
    print("=" * 60)
    print("MedEval Segmentation Metrics Demo")
    print("=" * 60)
    
    # Create synthetic data
    print("\n1. Creating synthetic 3D data...")
    n_samples = 20
    shape = (32, 32, 32)
    spacing = (2.0, 1.0, 1.0)  # Anisotropic spacing
    
    preds, targets = create_synthetic_3d_data(
        shape=shape, 
        n_samples=n_samples,
        noise_level=0.05,
    )
    print(f"   Created {n_samples} samples of shape {shape}")
    print(f"   Spacing: {spacing} mm")
    
    # Convert to tensors
    pred_tensor = torch.from_numpy(preds)
    target_tensor = torch.from_numpy(targets)
    
    # Compute individual metrics
    print("\n2. Computing metrics...")
    
    dice_scores = dice_score(pred_tensor, target_tensor, reduction="none")
    jaccard_scores = jaccard_index(pred_tensor, target_tensor, reduction="none")
    
    print(f"   Dice scores: mean={dice_scores.mean():.4f}, std={dice_scores.std():.4f}")
    print(f"   Jaccard scores: mean={jaccard_scores.mean():.4f}, std={jaccard_scores.std():.4f}")
    
    # Compute surface metrics with spacing
    print("\n3. Computing surface metrics (spacing-aware)...")
    hd_scores = hausdorff_distance(pred_tensor, target_tensor, spacing=spacing, reduction="none")
    hd95_scores = hausdorff_distance_95(pred_tensor, target_tensor, spacing=spacing, reduction="none")
    assd_scores = average_symmetric_surface_distance(pred_tensor, target_tensor, spacing=spacing, reduction="none")
    
    # Filter out inf values for statistics
    hd_finite = hd_scores[~torch.isinf(hd_scores)]
    hd95_finite = hd95_scores[~torch.isinf(hd95_scores)]
    assd_finite = assd_scores[~torch.isinf(assd_scores)]
    
    if len(hd_finite) > 0:
        print(f"   HD: mean={hd_finite.mean():.2f} mm, median={hd_finite.median():.2f} mm")
        print(f"   HD95: mean={hd95_finite.mean():.2f} mm, median={hd95_finite.median():.2f} mm")
        print(f"   ASSD: mean={assd_finite.mean():.2f} mm, median={assd_finite.median():.2f} mm")
    
    # Comprehensive metrics
    print("\n4. Computing comprehensive metrics...")
    all_metrics = compute_segmentation_metrics(
        pred_tensor, target_tensor,
        spacing=spacing,
        include_surface=True,
        include_calibration=True,
        reduction="none",
    )
    
    print("   Available metrics:", list(all_metrics.keys()))
    
    # Aggregate with confidence intervals
    print("\n5. Aggregating metrics with bootstrap CI...")
    metrics_dict = {
        "dice": dice_scores.numpy(),
        "jaccard": jaccard_scores.numpy(),
    }
    
    aggregated = aggregate_metrics(
        metrics_dict,
        method="mean",
        compute_ci=True,
        ci_method="bootstrap",
        confidence=0.95,
        n_bootstrap=1000,
        seed=42,
    )
    
    print("\n   Aggregated Results (mean [95% CI]):")
    for name, (mean, lower, upper) in aggregated.items():
        print(f"   {name}: {mean:.4f} [{lower:.4f}, {upper:.4f}]")
    
    # Using MedicalPrediction container
    print("\n6. Using MedicalPrediction container...")
    pred_container = MedicalPrediction(
        data=pred_tensor[0],
        spacing=spacing,
        patient_id="patient_001",
        strata="site_A",
    )
    
    print(f"   Patient ID: {pred_container.patient_id}")
    print(f"   Spacing: {pred_container.spacing}")
    print(f"   Physical shape: {pred_container.get_physical_shape()}")
    
    # Visualization
    if HAS_VIS:
        print("\n7. Creating visualizations...")
        
        # Plot metric distribution
        fig, ax = plot_metric_distribution(
            {"Dice": dice_scores.numpy(), "Jaccard": jaccard_scores.numpy()},
            kind="box",
            title="Segmentation Metric Distribution",
        )
        fig.savefig("metric_distribution.png", dpi=150, bbox_inches="tight")
        print("   Saved: metric_distribution.png")
        
        # Plot prediction comparison for one sample
        fig, axes = plot_prediction_comparison(
            image=targets[0],  # Use target as "image" for visualization
            ground_truth=targets[0],
            prediction=preds[0],
            slice_idx=shape[0] // 2,
            title="Segmentation Comparison (Middle Slice)",
        )
        fig.savefig("segmentation_comparison.png", dpi=150, bbox_inches="tight")
        print("   Saved: segmentation_comparison.png")
        
        # Plot error histogram
        errors = 1.0 - dice_scores.numpy()  # Dice error = 1 - Dice
        fig, ax = plot_error_histogram(
            errors,
            xlabel="Dice Error (1 - Dice)",
            title="Segmentation Error Distribution",
        )
        fig.savefig("error_histogram.png", dpi=150, bbox_inches="tight")
        print("   Saved: error_histogram.png")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

