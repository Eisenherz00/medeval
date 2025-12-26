#!/usr/bin/env python
"""Generate synthetic segmentation and classification data for demo."""

import csv
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
from medeval.core.io import save_nifti

DATA_DIR = DEMO_DIR / "data"
SEG_DIR = DATA_DIR / "seg"
CLS_DIR = DATA_DIR / "cls"


def create_sphere_mask(shape, center, radius):
    """Create a binary sphere mask in a 3D volume."""
    Z, Y, X = shape
    zz, yy, xx = np.ogrid[:Z, :Y, :X]
    cz, cy, cx = center
    dist_sq = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
    return (dist_sq <= radius ** 2).astype(np.float32)


def generate_segmentation_data(
    n_patients: int = 50,
    cases_per_patient: int = 2,
    strata: tuple[str, ...] = ("stratum_A", "stratum_B"),
    seed: int = 123,
):
    """Generate multiple 3D segmentation cases across patients and strata."""
    print("[gen_data] Creating synthetic 3D segmentation data for multiple cases...")

    np.random.seed(seed)

    shape = (32, 96, 96)
    spacing = (3.0, 1.0, 1.0)  # (dz, dy, dx) - anisotropic

    total_cases = n_patients * cases_per_patient
    rows = []

    try:
        import nibabel as _  # noqa: F401
        use_nifti = True
    except ImportError:
        use_nifti = False
        print("  nibabel not available, falling back to .npy format")

    for p in range(n_patients):
        patient_id = f"patient_{p+1:03d}"
        stratum = strata[p % len(strata)]
        for k in range(cases_per_patient):
            case_idx = p * cases_per_patient + k
            case_id = f"case{case_idx:04d}"

            # Ground truth sphere at center
            center_gt = (16, 48, 48)
            radius_gt = 12
            target = create_sphere_mask(shape, center_gt, radius_gt)

            # Prediction sphere shifted by small random offsets
            shift_z = int(np.random.randint(-1, 2))
            shift_y = int(np.random.randint(-3, 4))
            shift_x = int(np.random.randint(-3, 4))
            center_pred = (
                center_gt[0] + shift_z,
                center_gt[1] + shift_y,
                center_gt[2] + shift_x,
            )
            radius_pred = 12
            pred = create_sphere_mask(shape, center_pred, radius_pred)

            # Add small false positive blob with 50% probability
            if np.random.rand() < 0.5:
                fp_radius = int(np.random.choice([3, 4]))
                # Random center within bounds so that sphere fits
                margin = fp_radius + 1
                fp_z = np.random.randint(margin, shape[0] - margin)
                fp_y = np.random.randint(margin, shape[1] - margin)
                fp_x = np.random.randint(margin, shape[2] - margin)
                fp_blob = create_sphere_mask(shape, (fp_z, fp_y, fp_x), fp_radius)
                pred = np.clip(pred + fp_blob, 0, 1)

            # Save files
            if use_nifti:
                pred_path = SEG_DIR / f"{case_id}_pred.nii.gz"
                target_path = SEG_DIR / f"{case_id}_tgt.nii.gz"
                # MedEval save_nifti expects data in (Z,Y,X) with spacing (dz,dy,dx)
                save_nifti(pred, str(pred_path), spacing=spacing)
                save_nifti(target, str(target_path), spacing=spacing)
            else:
                pred_path = SEG_DIR / f"{case_id}_pred.npy"
                target_path = SEG_DIR / f"{case_id}_tgt.npy"
                np.save(pred_path, pred)
                np.save(target_path, target)

            meta = {
                "case_id": case_id,
                "patient_id": patient_id,
                "strata": stratum,
                "spacing": list(spacing),
                "shape": list(shape),
                "description": "Synthetic binary segmentation: sphere GT vs shifted sphere + optional FP blob",
                "shift": [shift_z, shift_y, shift_x],
            }
            meta_path = SEG_DIR / f"{case_id}_meta.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            # Collect manifest row with relative paths (aligned to CLI defaults)
            rows.append(
                {
                    "case_id": case_id,
                    "patient_id": patient_id,
                    "strata": stratum,
                    "prediction": str(pred_path.relative_to(DEMO_DIR)),
                    "target": str(target_path.relative_to(DEMO_DIR)),
                    "spacing": ",".join(map(str, spacing)),
                    "meta_path": str(meta_path.relative_to(DEMO_DIR)),
                }
            )

            # Print info for first 3 cases only
            if case_idx < 3:
                print(f"  Generated {case_id}: patient={patient_id}, strata={stratum}, shift={meta['shift']}")

    print(f"  Generated total {total_cases} segmentation cases.")

    first_pred = pred
    first_target = target
    return first_pred, first_target, spacing, rows


def write_segmentation_manifest(rows, manifest_path: Path) -> None:
    """Write segmentation manifest CSV for CLI and example scripts."""
    fieldnames = ["case_id", "patient_id", "strata", "prediction", "target", "spacing", "meta_path"]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_classification_data():
    """Generate synthetic classification data with per-patient grouping."""
    print("[gen_data] Creating synthetic classification data...")

    np.random.seed(42)

    n_samples = 200
    n_patients = 50
    samples_per_patient = n_samples // n_patients

    # Binary labels with ~25% prevalence
    labels = np.zeros(n_samples, dtype=np.int64)
    positive_patients = np.random.choice(n_patients, size=int(n_patients * 0.25), replace=False)

    for i, pid in enumerate(range(n_patients)):
        start = i * samples_per_patient
        end = start + samples_per_patient
        if pid in positive_patients:
            labels[start:end] = 1

    # Generate probabilities with signal (correlated with labels)
    # For positive samples: higher probs, for negative: lower probs
    probs = np.random.beta(2, 5, n_samples).astype(np.float32)  # base distribution
    probs[labels == 1] = np.random.beta(5, 2, np.sum(labels == 1))  # shift positive

    # Add some noise to make it imperfect
    probs = np.clip(probs + np.random.normal(0, 0.1, n_samples), 0.01, 0.99).astype(np.float32)

    # Group IDs: integer patient IDs
    group_ids = np.repeat(np.arange(n_patients), samples_per_patient).astype(np.int64)

    # Save
    np.save(CLS_DIR / "cls_probs.npy", probs)
    np.save(CLS_DIR / "cls_labels.npy", labels)
    np.save(CLS_DIR / "cls_group_ids.npy", group_ids)

    print(f"  Saved: {CLS_DIR / 'cls_probs.npy'}")
    print(f"  Saved: {CLS_DIR / 'cls_labels.npy'}")
    print(f"  Saved: {CLS_DIR / 'cls_group_ids.npy'}")
    print(f"  N={n_samples}, prevalence={labels.mean():.2%}, n_patients={n_patients}")

    return probs, labels, group_ids


def smoke_test_shapes(pred, target, spacing):
    """Verify normalize_input_shapes works on our data."""
    print("[gen_data] Running smoke test on normalize_input_shapes...")

    pred_t = as_tensor(pred)
    target_t = as_tensor(target)

    pred_norm, target_norm, spatial_dims, spacing_out = normalize_input_shapes(
        pred_t, target_t, spacing=spacing
    )

    print(f"  Input shape: {tuple(pred_t.shape)}")
    print(f"  Normalized shape: {tuple(pred_norm.shape)}")
    print(f"  Spatial dims: {spatial_dims}")
    print(f"  Spacing: {spacing_out}")

    assert pred_norm.shape == target_norm.shape, "Shape mismatch after normalization"
    assert spatial_dims == 3, f"Expected 3D, got {spatial_dims}D"
    print("  Smoke test passed!")


def main():
    SEG_DIR.mkdir(parents=True, exist_ok=True)
    CLS_DIR.mkdir(parents=True, exist_ok=True)

    pred, target, spacing, rows = generate_segmentation_data(n_patients=50, cases_per_patient=2)
    manifest_path = DEMO_DIR / "manifest_seg.csv"
    write_segmentation_manifest(rows, manifest_path)
    print(f"[gen_data] Wrote segmentation manifest with {len(rows)} cases: {manifest_path}")

    generate_classification_data()

    # Smoke test
    smoke_test_shapes(pred, target, spacing)

    print("\n[gen_data] Done! Data generated in demo_external_user/data/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
