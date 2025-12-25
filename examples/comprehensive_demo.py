#!/usr/bin/env python
"""
MedEval Comprehensive Demo
==========================

This script demonstrates the full capabilities of the MedEval library for
evaluating medical imaging tasks across all four domains:
- Segmentation (2D/3D with surface metrics)
- Classification (with calibration and CI)
- Detection (mAP, FROC)
- Registration (TRE, image similarity)

Usage:
    python comprehensive_demo.py

This example shows how medeval can be used in a real medical imaging pipeline.
"""

import numpy as np
import torch

print("=" * 70)
print("MedEval - Medical Imaging Evaluation Library")
print("Comprehensive Demo")
print("=" * 70)


# =============================================================================
# 1. SEGMENTATION METRICS
# =============================================================================
print("\n" + "=" * 70)
print("1. SEGMENTATION METRICS")
print("=" * 70)

from medeval.metrics.segmentation import (
    dice_score,
    jaccard_index,
    hausdorff_distance,
    hausdorff_distance_95,
    average_symmetric_surface_distance,
    surface_dice,
    compute_segmentation_metrics,
)
from medeval.core.aggregate import aggregate_metrics

# Create synthetic 3D segmentation data
print("\n[Segmentation] Creating synthetic 3D tumor segmentation data...")
np.random.seed(42)
n_samples = 10
shape = (32, 32, 32)
spacing = (2.0, 1.0, 1.0)  # Anisotropic spacing (mm)

def create_sphere_mask(shape, center, radius):
    """Create a spherical binary mask."""
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
    return (dist <= radius).astype(np.float32)

predictions = []
targets = []
for i in range(n_samples):
    center = np.array(shape) // 2
    radius = 8 + np.random.randint(-2, 3)
    
    # Ground truth
    target = create_sphere_mask(shape, center, radius)
    
    # Prediction with slight offset (simulating model imperfection)
    pred_center = center + np.random.randint(-2, 3, size=3)
    pred_radius = radius + np.random.randint(-1, 2)
    pred = create_sphere_mask(shape, pred_center, pred_radius)
    
    predictions.append(pred)
    targets.append(target)

pred_tensor = torch.from_numpy(np.stack(predictions))
target_tensor = torch.from_numpy(np.stack(targets))

print(f"   Data shape: {pred_tensor.shape}")
print(f"   Voxel spacing: {spacing} mm (anisotropic)")

# Compute comprehensive metrics
print("\n[Segmentation] Computing metrics...")
results = compute_segmentation_metrics(
    pred_tensor, target_tensor,
    spacing=spacing,
    include_surface=True,
    include_calibration=False,
    reduction="none"
)

# Display results
print("\n   Results (per-sample):")
print(f"   Dice:      {results['dice'].mean():.4f} ± {results['dice'].std():.4f}")
print(f"   Jaccard:   {results['jaccard'].mean():.4f} ± {results['jaccard'].std():.4f}")
print(f"   Precision: {results['precision'].mean():.4f} ± {results['precision'].std():.4f}")
print(f"   Recall:    {results['recall'].mean():.4f} ± {results['recall'].std():.4f}")

# Filter valid surface metrics (exclude NaN)
hd = results['hausdorff']
hd_valid = hd[~torch.isnan(hd)]
if len(hd_valid) > 0:
    print(f"\n   Surface Metrics (with {spacing} mm spacing):")
    print(f"   HD:        {hd_valid.mean():.2f} ± {hd_valid.std():.2f} mm")
    hd95 = results['hausdorff_95']
    hd95_valid = hd95[~torch.isnan(hd95)]
    print(f"   HD95:      {hd95_valid.mean():.2f} ± {hd95_valid.std():.2f} mm")
    assd = results['assd']
    assd_valid = assd[~torch.isnan(assd)]
    print(f"   ASSD:      {assd_valid.mean():.2f} ± {assd_valid.std():.2f} mm")

# Aggregate with bootstrap confidence intervals
print("\n[Segmentation] Computing 95% CI (bootstrap)...")
metrics_dict = {"dice": results['dice'].numpy(), "jaccard": results['jaccard'].numpy()}
aggregated = aggregate_metrics(metrics_dict, method="mean", compute_ci=True, n_bootstrap=1000, seed=42)

print("\n   Aggregated Results with 95% CI:")
for name, (mean, lower, upper) in aggregated.items():
    print(f"   {name}: {mean:.4f} [{lower:.4f}, {upper:.4f}]")


# =============================================================================
# 2. CLASSIFICATION METRICS  
# =============================================================================
print("\n" + "=" * 70)
print("2. CLASSIFICATION METRICS")
print("=" * 70)

from medeval.metrics.classification import (
    auroc,
    auprc,
    accuracy,
    sensitivity,
    specificity,
    expected_calibration_error,
    youden_threshold,
    compute_classification_metrics,
)

# Create synthetic classification data (e.g., malignancy prediction)
print("\n[Classification] Creating synthetic malignancy classification data...")
np.random.seed(42)
n_samples = 500

# Simulate a classifier with decent performance
# Positive samples have higher predicted probabilities on average
labels = np.concatenate([np.zeros(300), np.ones(200)])  # 40% prevalence
probs = np.concatenate([
    np.clip(np.random.beta(2, 5, 300), 0, 1),  # Negatives: lower probs
    np.clip(np.random.beta(5, 2, 200), 0, 1),  # Positives: higher probs
])

# Shuffle
indices = np.random.permutation(n_samples)
probs = probs[indices]
labels = labels[indices]

print(f"   Samples: {n_samples}, Prevalence: {labels.mean():.1%}")

# Compute AUROC with DeLong CI
print("\n[Classification] Computing AUROC with DeLong CI...")
score, ci_lower, ci_upper = auroc(probs, labels, compute_ci=True)
print(f"   AUROC: {score:.4f} [95% CI: {ci_lower:.4f}, {ci_upper:.4f}]")

# Compute AUPRC
auprc_score = auprc(probs, labels)
print(f"   AUPRC: {auprc_score:.4f}")

# Find optimal threshold using Youden's J
optimal_thresh = youden_threshold(probs, labels)
print(f"   Optimal threshold (Youden): {optimal_thresh:.3f}")

# Compute metrics at optimal threshold
sens = sensitivity(probs, labels, threshold=optimal_thresh)
spec = specificity(probs, labels, threshold=optimal_thresh)
print(f"   Sensitivity @ {optimal_thresh:.2f}: {sens.item():.4f}")
print(f"   Specificity @ {optimal_thresh:.2f}: {spec.item():.4f}")

# Calibration
print("\n[Classification] Computing calibration metrics...")
ece = expected_calibration_error(probs, labels, n_bins=10)
print(f"   ECE (10 bins): {ece.item():.4f}")

# Comprehensive metrics
print("\n[Classification] All metrics summary:")
all_metrics = compute_classification_metrics(probs, labels, compute_ci=False, include_calibration=True)
for name, value in all_metrics.items():
    if isinstance(value, torch.Tensor):
        print(f"   {name}: {value.item():.4f}")
    else:
        print(f"   {name}: {value:.4f}")


# =============================================================================
# 3. DETECTION METRICS
# =============================================================================
print("\n" + "=" * 70)
print("3. DETECTION METRICS")
print("=" * 70)

from medeval.metrics.detection import (
    box_iou_2d,
    mean_average_precision,
    froc,
    hungarian_matching,
)

# Create synthetic detection data (e.g., lesion detection)
print("\n[Detection] Creating synthetic lesion detection data...")
np.random.seed(42)

# Format: [x1, y1, x2, y2] bounding boxes
# Ground truth: 3 lesions in an image
gt_boxes = np.array([
    [10, 10, 30, 30],   # Lesion 1
    [50, 50, 80, 80],   # Lesion 2
    [120, 100, 150, 130]  # Lesion 3
], dtype=np.float32)

# Predictions: 4 detections (3 true positives + 1 false positive)
pred_boxes = np.array([
    [12, 8, 32, 28],    # TP for lesion 1 (slight offset)
    [48, 52, 78, 82],   # TP for lesion 2
    [118, 102, 148, 132],  # TP for lesion 3
    [200, 200, 220, 220]   # FP (no matching GT)
], dtype=np.float32)

pred_scores = np.array([0.95, 0.88, 0.75, 0.60])

print(f"   Ground truth boxes: {len(gt_boxes)}")
print(f"   Predicted boxes: {len(pred_boxes)}")

# Compute IoU matrix
print("\n[Detection] Computing IoU matrix...")
iou_matrix = box_iou_2d(torch.from_numpy(pred_boxes), torch.from_numpy(gt_boxes))
print("   IoU matrix (predictions vs ground truth):")
print(iou_matrix.numpy().round(3))

# Hungarian matching (maximize IoU)
print("\n[Detection] Hungarian matching...")
row_ind, col_ind = hungarian_matching(iou_matrix, maximize=True)
print(f"   Matched pairs: {len(row_ind)}")
for r, c in zip(row_ind, col_ind):
    iou = iou_matrix[r, c].item()
    if iou > 0.3:  # Only show matches above threshold
        print(f"   Pred {r} -> GT {c} (IoU: {iou:.3f})")

# mAP calculation
print("\n[Detection] Computing mAP...")
# Single class (class 0 = lesion)
pred_labels = np.zeros(len(pred_boxes), dtype=np.int64)
gt_labels = np.zeros(len(gt_boxes), dtype=np.int64)

map_result = mean_average_precision(
    pred_boxes=[pred_boxes],
    pred_scores=[pred_scores],
    pred_labels=[pred_labels],
    target_boxes=[gt_boxes],
    target_labels=[gt_labels],
    iou_thresholds=np.array([0.5]),
    class_aware=False,
)
print(f"   mAP@0.5: {map_result['mAP@0.50']:.4f}")


# =============================================================================
# 4. REGISTRATION METRICS
# =============================================================================
print("\n" + "=" * 70)
print("4. REGISTRATION METRICS")
print("=" * 70)

from medeval.metrics.registration import (
    target_registration_error,
    normalized_mutual_information,
    normalized_cross_correlation,
    jacobian_determinant,
    compute_registration_metrics,
)

# Create synthetic registration data
print("\n[Registration] Creating synthetic landmark registration data...")
np.random.seed(42)
n_landmarks = 20

# Ground truth landmarks
target_landmarks = np.random.rand(n_landmarks, 3) * 100  # Random 3D points

# Predicted landmarks with registration error (simulating imperfect registration)
registration_noise = np.random.randn(n_landmarks, 3) * 1.5  # ~1.5mm noise
pred_landmarks = target_landmarks + registration_noise

print(f"   Landmarks: {n_landmarks}")
print(f"   Registration noise: ~1.5mm Gaussian")

# Compute TRE
print("\n[Registration] Computing Target Registration Error (TRE)...")
spacing = (1.0, 1.0, 1.0)  # Isotropic 1mm spacing
tre_results = target_registration_error(pred_landmarks, target_landmarks, spacing=spacing)

print(f"   Mean TRE: {tre_results['mean']:.2f} mm")
print(f"   Median TRE: {tre_results['median']:.2f} mm")
print(f"   95th percentile TRE: {tre_results['95th_percentile']:.2f} mm")

# Image similarity metrics
print("\n[Registration] Computing image similarity metrics...")
# Create synthetic images (a simple gradient pattern)
image_size = (64, 64)
x, y = np.meshgrid(np.linspace(0, 1, image_size[1]), np.linspace(0, 1, image_size[0]))
fixed_image = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
# Moving image with slight transformation
moving_image = np.sin(2 * np.pi * (x + 0.05)) * np.cos(2 * np.pi * (y - 0.03)) + np.random.randn(*image_size) * 0.1

nmi = normalized_mutual_information(fixed_image, moving_image)
ncc = normalized_cross_correlation(torch.from_numpy(fixed_image), torch.from_numpy(moving_image))

print(f"   NMI: {nmi:.4f} (1.0 = identical)")
print(f"   NCC: {ncc:.4f} (1.0 = identical)")

# Deformation field quality
print("\n[Registration] Computing deformation field quality...")
# Create a smooth 2D deformation field
deformation = torch.zeros(2, 32, 32)
# Add a small, smooth deformation
deformation[0] = torch.randn(32, 32) * 0.5  # x-displacement
deformation[1] = torch.randn(32, 32) * 0.5  # y-displacement

jac_results = jacobian_determinant(deformation)
print(f"   Mean Jacobian: {jac_results['mean']:.4f} (1.0 = volume preserving)")
print(f"   Folding percentage: {jac_results['folding_percentage']:.2f}% (0% = no folding)")


# =============================================================================
# 5. AGGREGATION WITH STRATIFICATION
# =============================================================================
print("\n" + "=" * 70)
print("5. AGGREGATION WITH STRATIFICATION")
print("=" * 70)

from medeval.core.aggregate import stratified_aggregate

# Simulate multi-site study
print("\n[Aggregation] Simulating multi-site study...")
np.random.seed(42)

# Site A: 30 samples, higher quality
site_a_dice = np.random.beta(8, 2, 30)  # Higher Dice
# Site B: 20 samples, slightly lower quality  
site_b_dice = np.random.beta(6, 3, 20)  # Lower Dice

all_dice = np.concatenate([site_a_dice, site_b_dice])
# Use numeric strata IDs (0=Site_A, 1=Site_B)
strata = np.array([0] * 30 + [1] * 20)
site_names = {0: "Site_A", 1: "Site_B"}

print(f"   Site A: {len(site_a_dice)} samples, mean Dice = {site_a_dice.mean():.4f}")
print(f"   Site B: {len(site_b_dice)} samples, mean Dice = {site_b_dice.mean():.4f}")

# Stratified aggregation
print("\n[Aggregation] Computing stratified metrics with 95% CI...")
stratified_results = stratified_aggregate(
    metrics={"dice": all_dice},
    strata=strata,
    method="mean",
    compute_ci=True,
    n_bootstrap=1000,
)

print("\n   Per-stratum results:")
# Structure is results[metric_name][stratum] = (mean, lower, upper)
dice_results = stratified_results["dice"]
for stratum_key, values in dice_results.items():
    stratum_int = int(stratum_key) if stratum_key.isdigit() else stratum_key
    name = site_names.get(stratum_int, stratum_key)
    mean, lower, upper = values
    print(f"   {name}: {mean:.4f} [95% CI: {lower:.4f}, {upper:.4f}]")

# Compute overall (pooled) separately
print("\n[Aggregation] Computing overall pooled metrics...")
from medeval.core.aggregate import bootstrap_ci
overall_ci = bootstrap_ci(all_dice, confidence=0.95, n_bootstrap=1000, method="mean")
print(f"   Overall: {overall_ci[0]:.4f} [95% CI: {overall_ci[1]:.4f}, {overall_ci[2]:.4f}]")


# =============================================================================
# 6. PRACTICAL WORKFLOW EXAMPLE
# =============================================================================
print("\n" + "=" * 70)
print("6. PRACTICAL WORKFLOW: Tumor Segmentation Study")
print("=" * 70)

print("""
Example workflow for a real tumor segmentation study:

1. Load data:
   >>> from medeval.core.io import load_nifti, get_nifti_spacing
   >>> pred = load_nifti("prediction.nii.gz", as_torch=True)
   >>> target = load_nifti("ground_truth.nii.gz", as_torch=True)
   >>> spacing = get_nifti_spacing("ground_truth.nii.gz")

2. Compute metrics:
   >>> from medeval.metrics.segmentation import compute_segmentation_metrics
   >>> results = compute_segmentation_metrics(
   ...     pred, target,
   ...     spacing=spacing,
   ...     include_surface=True,
   ... )

3. Aggregate across patients:
   >>> from medeval.core.aggregate import aggregate_metrics
   >>> aggregated = aggregate_metrics(
   ...     results_dict,
   ...     method="mean",
   ...     compute_ci=True,
   ...     ci_method="bootstrap",
   ... )

4. Generate report:
   >>> print(f"Dice: {aggregated['dice'][0]:.4f} [{aggregated['dice'][1]:.4f}, {aggregated['dice'][2]:.4f}]")
   >>> print(f"HD95: {aggregated['hausdorff_95'][0]:.2f} mm")
""")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
MedEval provides:

✓ Segmentation: Dice, Jaccard, HD, HD95, ASSD, Surface Dice
  - Proper spacing handling for anisotropic voxels
  - Multi-class and ignore-index support

✓ Classification: AUROC, AUPRC, ECE, Brier score
  - DeLong CI for AUROC
  - Calibration analysis
  - Youden threshold optimization

✓ Detection: IoU, mAP, FROC, Hungarian matching
  - COCO-style evaluation
  - 2D and 3D box support

✓ Registration: TRE, NMI, NCC, Jacobian analysis
  - Landmark-based and image-based metrics
  - Deformation field quality

✓ Aggregation: Bootstrap CI, jackknife, stratification
  - Per-site analysis
  - Confidence intervals

All metrics support:
  - PyTorch tensors and NumPy arrays
  - CPU and GPU computation
  - 2D and 3D data
  - Batch processing
""")

print("=" * 70)
print("Demo completed!")
print("=" * 70)

