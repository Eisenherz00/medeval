#!/usr/bin/env python
"""Demonstrate medeval Python API for segmentation and classification metrics."""

import json
import sys
from pathlib import Path

import numpy as np

# Ensure repo root is importable
DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from medeval.core.utils import normalize_input_shapes
from medeval.core.typing import as_tensor
from medeval.metrics.segmentation import (
    dice_score,
    jaccard_index,
    hausdorff_distance_95,
    average_symmetric_surface_distance,
    compute_segmentation_metrics,
)
from medeval.metrics.classification import (
    auroc,
    auprc,
    accuracy,
    expected_calibration_error,
    group_by_patient,
    compute_classification_metrics,
)

DATA_DIR = DEMO_DIR / "data"
SEG_DIR = DATA_DIR / "seg"
CLS_DIR = DATA_DIR / "cls"
OUT_DIR = DEMO_DIR / "out_python_api"


def load_segmentation_data():
    """Load segmentation pred/target and spacing."""
    meta_path = SEG_DIR / "case001_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    spacing = tuple(meta["spacing"])  # (dz, dy, dx)

    # Try NIfTI first, then .npy
    nii_pred = SEG_DIR / "case001_pred.nii.gz"
    nii_tgt = SEG_DIR / "case001_tgt.nii.gz"

    if nii_pred.exists() and nii_tgt.exists():
        try:
            import nibabel as nib
            pred = nib.load(str(nii_pred)).get_fdata().astype(np.float32)
            target = nib.load(str(nii_tgt)).get_fdata().astype(np.float32)
            print(f"[run_python_api] Loaded NIfTI files")
        except ImportError:
            pred = np.load(SEG_DIR / "case001_pred.npy")
            target = np.load(SEG_DIR / "case001_tgt.npy")
            print(f"[run_python_api] Loaded .npy files (nibabel not available)")
    else:
        pred = np.load(SEG_DIR / "case001_pred.npy")
        target = np.load(SEG_DIR / "case001_tgt.npy")
        print(f"[run_python_api] Loaded .npy files")

    print(f"  Pred shape: {pred.shape}, Target shape: {target.shape}")
    print(f"  Spacing (dz, dy, dx): {spacing}")

    return pred, target, spacing


def load_classification_data():
    """Load classification probs, labels, and group_ids."""
    probs = np.load(CLS_DIR / "cls_probs.npy")
    labels = np.load(CLS_DIR / "cls_labels.npy")
    group_ids = np.load(CLS_DIR / "cls_group_ids.npy")

    print(f"[run_python_api] Loaded classification data")
    print(f"  N={len(probs)}, prevalence={labels.mean():.2%}")

    return probs, labels, group_ids


def evaluate_segmentation(pred, target, spacing):
    """Compute segmentation metrics."""
    print("\n=== Segmentation Metrics ===")

    # Convert to tensors
    pred_t = as_tensor(pred)
    target_t = as_tensor(target)

    # Compute individual metrics
    dice = dice_score(pred_t, target_t, reduction="mean-case")
    iou = jaccard_index(pred_t, target_t, reduction="mean-case")

    # Surface metrics require spacing
    hd95 = hausdorff_distance_95(pred_t, target_t, spacing=spacing, reduction="mean-case")
    assd = average_symmetric_surface_distance(pred_t, target_t, spacing=spacing, reduction="mean-case")

    results = {
        "dice": float(dice.item()),
        "jaccard": float(iou.item()),
        "hausdorff_95": float(hd95.item()),
        "assd": float(assd.item()),
    }

    print(f"  Dice Score:    {results['dice']:.4f}")
    print(f"  Jaccard (IoU): {results['jaccard']:.4f}")
    print(f"  HD95:          {results['hausdorff_95']:.4f} mm")
    print(f"  ASSD:          {results['assd']:.4f} mm")

    # Also compute all metrics at once
    all_metrics = compute_segmentation_metrics(
        pred_t, target_t, spacing=spacing, include_surface=True, include_calibration=False
    )
    print(f"\n  All metrics via compute_segmentation_metrics():")
    for k, v in all_metrics.items():
        val = float(v.item()) if hasattr(v, 'item') else float(v)
        print(f"    {k}: {val:.4f}")

    # Smoke test: normalize_input_shapes
    pred_norm, target_norm, spatial_dims, sp = normalize_input_shapes(pred_t, target_t, spacing)
    print(f"\n  [smoke test] normalize_input_shapes: shape={tuple(pred_norm.shape)}, spatial_dims={spatial_dims}")

    return results


def evaluate_classification(probs, labels, group_ids):
    """Compute classification metrics."""
    print("\n=== Classification Metrics ===")

    # Convert to tensors
    probs_t = as_tensor(probs)
    labels_t = as_tensor(labels)
    group_ids_t = as_tensor(group_ids)

    # Sample-level metrics
    auc = auroc(probs_t, labels_t)
    ap = auprc(probs_t, labels_t)
    acc = accuracy(probs_t, labels_t)
    ece = expected_calibration_error(probs_t, labels_t)

    results = {
        "auroc": float(auc),
        "auprc": float(ap),
        "accuracy": float(acc.item()) if hasattr(acc, 'item') else float(acc),
        "ece": float(ece.item()) if hasattr(ece, 'item') else float(ece),
    }

    print(f"  AUROC:    {results['auroc']:.4f}")
    print(f"  AUPRC:    {results['auprc']:.4f}")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  ECE:      {results['ece']:.4f}")

    # Per-patient aggregation
    print(f"\n  Per-patient aggregation (group_by_patient):")
    patient_probs, patient_labels = group_by_patient(probs_t, labels_t, group_ids_t, aggregation="mean")
    patient_auc = auroc(patient_probs, patient_labels)
    print(f"    Patient-level AUROC: {patient_auc:.4f} (N={len(patient_probs)} patients)")

    results["patient_auroc"] = float(patient_auc)

    # All metrics at once
    all_metrics = compute_classification_metrics(
        probs_t, labels_t, include_calibration=True
    )
    print(f"\n  All metrics via compute_classification_metrics():")
    for k, v in all_metrics.items():
        if isinstance(v, tuple):
            val = v[0]  # value without CI
        elif hasattr(v, 'item'):
            val = float(v.item())
        else:
            val = float(v)
        print(f"    {k}: {val:.4f}")

    return results


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    pred, target, spacing = load_segmentation_data()
    probs, labels, group_ids = load_classification_data()

    # Compute metrics
    seg_results = evaluate_segmentation(pred, target, spacing)
    cls_results = evaluate_classification(probs, labels, group_ids)

    # Combine and save
    all_results = {
        "segmentation": seg_results,
        "classification": cls_results,
    }

    results_path = OUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[run_python_api] Results saved to: {results_path}")
    print("\n=== Summary ===")
    print(f"Segmentation Dice: {seg_results['dice']:.4f}")
    print(f"Classification AUROC: {cls_results['auroc']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

