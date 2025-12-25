#!/usr/bin/env python
"""
Realistic external user workflow demonstrating medeval.

Run from repo root:
    python demo_external_user/run_example.py

Requires:
    python demo_external_user/gen_data.py  (to generate synthetic data first)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repo root is importable
DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

# medeval imports
from medeval.core.io import load_image
from medeval.core.typing import as_tensor
from medeval.metrics.segmentation import compute_segmentation_metrics
from medeval.metrics.classification import (
    auroc,
    auprc,
    accuracy,
    expected_calibration_error,
    reliability_diagram,
    group_by_patient,
)
from medeval.vis import plot_roc_curve, plot_reliability_diagram

# Paths
DATA_DIR = DEMO_DIR / "data"
SEG_DIR = DATA_DIR / "seg"
CLS_DIR = DATA_DIR / "cls"
OUT_DIR = DEMO_DIR / "out_example"

# sklearn for ROC curve data
from sklearn.metrics import roc_curve


def run_segmentation_workflow():
    """
    Segmentation workflow:
    - Read manifest with pandas
    - Load NIfTI files using medeval IO
    - Compute metrics per case
    - Return DataFrame of results
    """
    print("\n" + "=" * 60)
    print("SEGMENTATION WORKFLOW")
    print("=" * 60)

    # Read manifest using pandas
    manifest_path = DEMO_DIR / "manifest_seg.csv"
    df = pd.read_csv(manifest_path)
    print(f"Loaded manifest: {manifest_path}")
    print(f"  Cases: {len(df)}")

    # Support both legacy and current manifest column names
    pred_col = "pred_path" if "pred_path" in df.columns else "prediction"
    tgt_col = "target_path" if "target_path" in df.columns else "target"

    missing = [c for c in (pred_col, tgt_col) if c not in df.columns]
    if missing:
        raise KeyError(
            f"Manifest is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    results = []

    for idx, row in df.iterrows():
        case_id = row.get("case_id", f"case{idx:04d}")
        patient_id = row.get("patient_id", "")
        stratum = row.get("strata", "")

        pred_path = DEMO_DIR / str(row[pred_col])
        target_path = DEMO_DIR / str(row[tgt_col])

        if idx < 3:
            print(f"\nProcessing {case_id} (patient={patient_id}, strata={stratum})...")
        elif (idx + 1) % 20 == 0:
            print(f"Processed {idx+1}/{len(df)} cases...")

        # Load images using medeval IO
        pred = load_image(str(pred_path), as_torch=True)
        target = load_image(str(target_path), as_torch=True)

        # Demo convention: fixed anisotropic spacing (dz, dy, dx)
        spacing = (3.0, 1.0, 1.0)

        if idx < 3:
            print(f"  Pred shape: {tuple(pred.shape)}")
            print(f"  Spacing (dz, dy, dx): {spacing}")

        # Compute all segmentation metrics
        metrics = compute_segmentation_metrics(
            pred=pred,
            target=target,
            spacing=spacing,
            include_surface=True,
            include_calibration=False,
            reduction="mean-case",
        )

        # Collect results
        case_result = {
            "case_id": case_id,
            "patient_id": patient_id,
            "strata": stratum,
        }
        for k, v in metrics.items():
            val = float(v.item()) if hasattr(v, "item") else float(v)
            case_result[k] = val

        results.append(case_result)

        # Print key metrics
        if idx < 3:
            print(f"  Dice: {case_result['dice']:.4f}")
            print(f"  Jaccard: {case_result['jaccard']:.4f}")
            print(f"  HD95: {case_result.get('hausdorff_95', float('nan')):.4f}")
            print(f"  ASSD: {case_result.get('assd', float('nan')):.4f}")

    # Create DataFrame
    results_df = pd.DataFrame(results)
    return results_df


def run_classification_workflow():
    """
    Classification workflow:
    - Load numpy arrays
    - Compute metrics
    - Compute patient-level aggregation
    - Return metrics dict
    """
    print("\n" + "=" * 60)
    print("CLASSIFICATION WORKFLOW")
    print("=" * 60)

    # Load data
    probs = np.load(CLS_DIR / "cls_probs.npy")
    labels = np.load(CLS_DIR / "cls_labels.npy")
    group_ids = np.load(CLS_DIR / "cls_group_ids.npy")

    print(f"Loaded classification data:")
    print(f"  Samples: {len(probs)}")
    print(f"  Prevalence: {labels.mean():.2%}")
    print(f"  Patients: {len(np.unique(group_ids))}")

    # Convert to tensors
    probs_t = as_tensor(probs)
    labels_t = as_tensor(labels)
    group_ids_t = as_tensor(group_ids)

    # Sample-level metrics
    auc_val = auroc(probs_t, labels_t)
    ap_val = auprc(probs_t, labels_t)
    acc_val = accuracy(probs_t, labels_t)
    ece_val = expected_calibration_error(probs_t, labels_t)

    print(f"\nSample-level metrics:")
    print(f"  AUROC: {auc_val:.4f}")
    print(f"  AUPRC: {ap_val:.4f}")
    print(f"  Accuracy: {float(acc_val.item()) if hasattr(acc_val, 'item') else float(acc_val):.4f}")
    print(f"  ECE: {float(ece_val.item()) if hasattr(ece_val, 'item') else float(ece_val):.4f}")

    # Patient-level aggregation
    patient_probs, patient_labels = group_by_patient(
        probs_t, labels_t, group_ids_t, aggregation="mean"
    )
    patient_auc = auroc(patient_probs, patient_labels)

    print(f"\nPatient-level metrics:")
    print(f"  AUROC: {patient_auc:.4f} (N={len(patient_probs)} patients)")

    # Reliability diagram data (for visualization)
    rel_data = reliability_diagram(probs_t, labels_t, n_bins=10)

    # Handle return types (auroc etc. return float, but type hints may be ambiguous)
    auc_float = float(auc_val) if isinstance(auc_val, (int, float)) else float(auc_val[0])
    ap_float = float(ap_val) if isinstance(ap_val, (int, float)) else float(ap_val[0])
    acc_float = float(acc_val.item()) if hasattr(acc_val, "item") else float(acc_val)
    ece_float = float(ece_val.item()) if hasattr(ece_val, "item") else float(ece_val)
    patient_auc_float = float(patient_auc) if isinstance(patient_auc, (int, float)) else float(patient_auc[0])

    results = {
        "sample_level": {
            "auroc": auc_float,
            "auprc": ap_float,
            "accuracy": acc_float,
            "ece": ece_float,
            "n_samples": len(probs),
            "prevalence": float(labels.mean()),
        },
        "patient_level": {
            "auroc": patient_auc_float,
            "n_patients": int(len(patient_probs)),
        },
    }

    # Store data for plotting
    patient_probs_np = patient_probs.cpu().numpy() if hasattr(patient_probs, "cpu") else np.asarray(patient_probs)
    patient_labels_np = patient_labels.cpu().numpy() if hasattr(patient_labels, "cpu") else np.asarray(patient_labels)
    
    results["_plot_data"] = {
        "probs": probs,
        "labels": labels,
        "patient_probs": patient_probs_np,
        "patient_labels": patient_labels_np,
        "reliability": rel_data,
    }

    return results


def generate_visualizations(seg_results_df, cls_results):
    """Generate and save visualizations.

    Visualization is optional. If matplotlib is not installed, this step is
    skipped gracefully so the demo still completes successfully.
    """
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "\n[INFO] matplotlib is not installed. Skipping visualization step.\n"
            "To enable plots, install with:\n"
            "  pip install matplotlib\n"
        )
        return

    # 1. Segmentation distribution plots (many cases)
    print("\n1. Segmentation metric distributions...")

    dice_vals = seg_results_df["dice"].dropna().to_numpy() if "dice" in seg_results_df.columns else np.array([])
    hd95_vals = seg_results_df["hausdorff_95"].dropna().to_numpy() if "hausdorff_95" in seg_results_df.columns else np.array([])

    # 1a) Dice histogram
    if dice_vals.size > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(dice_vals, bins=20)
        ax.set_title("Dice distribution across cases")
        ax.set_xlabel("Dice")
        ax.set_ylabel("Number of cases")
        fig.savefig(OUT_DIR / "dice_hist.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {OUT_DIR / 'dice_hist.png'}")

    # 1b) Dice by strata (boxplot)
    if "strata" in seg_results_df.columns and dice_vals.size > 0:
        tmp = seg_results_df[["strata", "dice"]].dropna()
        # Keep stable order
        strata_names = [s for s in tmp["strata"].astype(str).unique().tolist()]
        data = [tmp.loc[tmp["strata"].astype(str) == s, "dice"].to_numpy() for s in strata_names]
        if len(data) >= 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.boxplot(data, labels=strata_names)
            ax.set_title("Dice by strata")
            ax.set_xlabel("Strata")
            ax.set_ylabel("Dice")
            fig.savefig(OUT_DIR / "dice_by_strata_box.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {OUT_DIR / 'dice_by_strata_box.png'}")

    # 1c) HD95 histogram (mm)
    if hd95_vals.size > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(hd95_vals, bins=20)
        ax.set_title("HD95 distribution across cases (mm)")
        ax.set_xlabel("HD95 (mm)")
        ax.set_ylabel("Number of cases")
        fig.savefig(OUT_DIR / "hd95_hist.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {OUT_DIR / 'hd95_hist.png'}")

    # 2. ROC curve for classification
    print("\n2. ROC curve...")

    plot_data = cls_results["_plot_data"]
    probs = plot_data["probs"]
    labels = plot_data["labels"]

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_val = cls_results["sample_level"]["auroc"]

    fig, ax = plot_roc_curve(
        fpr=fpr,
        tpr=tpr,
        auc=auc_val,
        label="Classifier",
        title="ROC Curve - Binary Classification",
        figsize=(8, 8),
    )
    fig.savefig(OUT_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'roc_curve.png'}")

    # 3. Reliability diagram (calibration)
    print("\n3. Reliability diagram...")

    rel_data = plot_data["reliability"]
    ece_val = cls_results["sample_level"]["ece"]

    fig, ax = plot_reliability_diagram(
        bin_centers=rel_data["bin_centers"],
        accuracies=rel_data["accuracies"],
        confidences=rel_data["confidences"],
        counts=rel_data["counts"],
        ece=ece_val,
        title="Reliability Diagram - Model Calibration",
        figsize=(8, 8),
    )
    fig.savefig(OUT_DIR / "reliability_diagram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'reliability_diagram.png'}")


def main():
    # Create output directory
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check that data exists
    if not (DEMO_DIR / "manifest_seg.csv").exists():
        print("ERROR: Demo manifest not found. Run gen_data.py first:")
        print("  python demo_external_user/gen_data.py")
        return 1

    try:
        # Run workflows
        seg_results_df = run_segmentation_workflow()
        cls_results = run_classification_workflow()

        # Save segmentation results to CSV
        seg_csv_path = OUT_DIR / "segmentation_results.csv"
        seg_results_df.to_csv(seg_csv_path, index=False)
        print(f"\nSaved: {seg_csv_path}")

        # Save a compact per-strata summary (looks more like a real evaluation report)
        if "strata" in seg_results_df.columns and seg_results_df["strata"].astype(str).str.len().gt(0).any():
            summary_cols = [c for c in ["dice", "jaccard", "precision", "recall", "hausdorff_95", "assd"] if c in seg_results_df.columns]
            seg_summary = (
                seg_results_df
                .groupby("strata")[summary_cols]
                .agg(["count", "mean", "std", "median"])
                .reset_index()
            )
            seg_summary_path = OUT_DIR / "segmentation_summary_by_strata.csv"
            seg_summary.to_csv(seg_summary_path, index=False)
            print(f"Saved: {seg_summary_path}")

            # Print a short console preview
            print("\nSegmentation summary by strata (mean ± std):")
            for _, r in seg_results_df.groupby("strata")["dice"].agg(["count", "mean", "std"]).reset_index().iterrows():
                s = r["strata"]
                n = int(r["count"])
                mu = float(r["mean"])
                sd = float(r["std"]) if not np.isnan(r["std"]) else 0.0
                print(f"  {s}: n={n}, dice={mu:.4f} ± {sd:.4f}")

        # Save classification results to JSON (without plot data)
        cls_json = {k: v for k, v in cls_results.items() if not k.startswith("_")}
        cls_json_path = OUT_DIR / "classification_results.json"
        with open(cls_json_path, "w") as f:
            json.dump(cls_json, f, indent=2)
        print(f"Saved: {cls_json_path}")

        # Generate visualizations
        generate_visualizations(seg_results_df, cls_results)

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Mean Dice:              {seg_results_df['dice'].mean():.4f}")
        print(f"AUROC (sample-level):   {cls_results['sample_level']['auroc']:.4f}")
        print(f"AUROC (patient-level):  {cls_results['patient_level']['auroc']:.4f}")
        print("=" * 60)

        # List output files
        print(f"\nOutput files in {OUT_DIR}/:")
        for f in sorted(OUT_DIR.iterdir()):
            print(f"  {f.name}")

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
