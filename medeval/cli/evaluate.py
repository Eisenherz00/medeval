"""Evaluation command implementation."""

import json
import logging
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

import medeval
from medeval.core.aggregate import aggregate_metrics, stratified_aggregate
from medeval.core.containers import EvaluationBatch, MedicalPrediction
from medeval.core.io import get_nifti_spacing, load_image, load_nifti
from medeval.core.typing import Tensor

logger = logging.getLogger(__name__)


def _get_version_fingerprint() -> Dict[str, str]:
    """Get version fingerprint for reproducibility.

    Returns
    -------
    Dict[str, str]
        Dictionary with version information
    """
    fingerprint = {
        "medeval_version": medeval.__version__,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
    }

    # Try to get git commit hash (optional, fail silently)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            fingerprint["git_commit"] = result.stdout.strip()[:12]  # Short hash
    except Exception:
        pass

    return fingerprint


def _load_prediction_target(
    row: pd.Series,
    pred_col: str = "prediction",
    target_col: str = "target",
    base_dir: Optional[Path] = None,
    spacing_col: Optional[str] = "spacing",
    patient_col: Optional[str] = "patient_id",
    strata_col: Optional[str] = "strata",
) -> Tuple[MedicalPrediction, MedicalPrediction]:
    """
    Load prediction and target from manifest row.

    Parameters
    ----------
    row : pd.Series
        Manifest row
    pred_col : str
        Column name for prediction path
    target_col : str
        Column name for target path
    spacing_col : str, optional
        Column name for spacing
    patient_col : str, optional
        Column name for patient ID
    strata_col : str, optional
        Column name for stratum

    Returns
    -------
    Tuple[MedicalPrediction, MedicalPrediction]
        Loaded prediction and target
    """
    pred_path_raw = str(row[pred_col])
    target_path_raw = str(row[target_col])

    # Resolve relative paths with respect to the manifest directory.
    # This makes CLI runs robust regardless of the current working directory.
    pred_path = pred_path_raw
    target_path = target_path_raw
    if base_dir is not None:
        try:
            p = Path(pred_path_raw)
            if not p.is_absolute():
                pred_path = str((base_dir / p).resolve())
        except Exception:
            pred_path = pred_path_raw
        try:
            t = Path(target_path_raw)
            if not t.is_absolute():
                target_path = str((base_dir / t).resolve())
        except Exception:
            target_path = target_path_raw

    # Load data
    pred_data = load_image(pred_path, as_torch=True)
    target_data = load_image(target_path, as_torch=True)

    # Detection special-case:
    # If the tensors look like Nx(6/8) box tables, do not add batch/channel dims.
    # (The detection metric APIs expect a flat table.)
    is_detection_table = (
        isinstance(pred_data, torch.Tensor)
        and isinstance(target_data, torch.Tensor)
        and pred_data.dim() == 2
        and target_data.dim() == 2
        and int(pred_data.shape[-1]) in (6, 8)
        and int(target_data.shape[-1]) in (6, 8)
    )

    if not is_detection_table:
        # Ensure batch and channel dimensions exist.
        # File loaders often return (H,W) or (Z,Y,X) without batch/channel.
        # Our metric APIs generally expect (B, ...) or (B, C, ...).
        if isinstance(pred_data, torch.Tensor) and pred_data.dim() in (2, 3):
            pred_data = pred_data.unsqueeze(0)
        if isinstance(target_data, torch.Tensor) and target_data.dim() in (2, 3):
            target_data = target_data.unsqueeze(0)

        # Add an explicit channel dimension for common cases:
        # - 2D: (B, H, W) -> (B, 1, H, W)
        # - 3D: (B, Z, Y, X) -> (B, 1, Z, Y, X)
        if isinstance(pred_data, torch.Tensor):
            if pred_data.dim() == 3:
                pred_data = pred_data.unsqueeze(1)
            elif pred_data.dim() == 4:
                # Heuristic: if axis-1 looks like channels (small), keep as-is.
                # Otherwise treat as (B, Z, Y, X) and add a channel.
                c_or_z = int(pred_data.shape[1])
                if c_or_z not in (1, 2, 3, 4):
                    pred_data = pred_data.unsqueeze(1)

        if isinstance(target_data, torch.Tensor):
            if target_data.dim() == 3:
                target_data = target_data.unsqueeze(1)
            elif target_data.dim() == 4:
                c_or_z = int(target_data.shape[1])
                if c_or_z not in (1, 2, 3, 4):
                    target_data = target_data.unsqueeze(1)

    # Debug: final tensor shapes after normalization
    try:
        if isinstance(pred_data, torch.Tensor) and isinstance(target_data, torch.Tensor):
            logger.debug(
                "Normalized shapes - pred: %s target: %s",
                tuple(pred_data.shape),
                tuple(target_data.shape),
            )
    except Exception:
        pass

    # Get spacing
    spacing = None
    if spacing_col and spacing_col in row and pd.notna(row[spacing_col]):
        spacing_val = row[spacing_col]
        if isinstance(spacing_val, str):
            spacing = tuple(map(float, spacing_val.split(",")))
        elif isinstance(spacing_val, (list, tuple)):
            spacing = tuple(spacing_val)
    else:
        # Try to extract from file
        if pred_path.endswith((".nii", ".nii.gz")):
            try:
                spacing = get_nifti_spacing(pred_path)
            except Exception:
                pass
    logger.debug("Resolved spacing for %s: %s", pred_path, spacing)

    # Get metadata
    patient_id = str(row[patient_col]) if patient_col and patient_col in row and pd.notna(row[patient_col]) else None
    strata = str(row[strata_col]) if strata_col and strata_col in row and pd.notna(row[strata_col]) else None

    # Safety check: warn if prediction and target shapes differ after normalization
    if isinstance(pred_data, torch.Tensor) and isinstance(target_data, torch.Tensor):
        if pred_data.shape != target_data.shape:
            logger.warning(
                "Prediction/target shape mismatch after normalization: pred=%s target=%s (paths: %s vs %s)",
                tuple(pred_data.shape),
                tuple(target_data.shape),
                pred_path,
                target_path,
            )

    pred = MedicalPrediction(
        data=pred_data,
        spacing=spacing,
        patient_id=patient_id,
        strata=strata,
    )
    target = MedicalPrediction(
        data=target_data,
        spacing=spacing,
        patient_id=patient_id,
        strata=strata,
    )

    return pred, target


def _compute_segmentation_metrics(
    pred: MedicalPrediction,
    target: MedicalPrediction,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Compute segmentation metrics for a single case."""
    from medeval.metrics.segmentation import compute_segmentation_metrics

    # Only compute surface metrics when spacing is present.
    include_surface = bool(config.get("include_surface", True))
    if pred.spacing is None:
        include_surface = False

    results = compute_segmentation_metrics(
        pred=pred.to_tensor(),
        target=target.to_tensor(),
        spacing=pred.spacing,
        threshold=config.get("threshold", 0.5),
        ignore_index=config.get("ignore_index"),
        include_surface=include_surface,
        include_calibration=config.get("include_calibration", False),
        reduction="none",
    )

    # Convert tensors to floats and sanitize non-finite values.
    output: Dict[str, float] = {}
    for k, v in results.items():
        if isinstance(v, torch.Tensor):
            val = float(v.mean().item() if v.numel() > 1 else v.item())
        else:
            val = float(v)

        output[k] = float("nan") if not math.isfinite(val) else val

    # Record whether surface metrics were skipped for this case.
    if bool(config.get("include_surface", True)) and pred.spacing is None:
        output["_surface_metrics_skipped"] = 1.0
    else:
        output["_surface_metrics_skipped"] = 0.0

    return output


def _compute_classification_metrics(
    pred: MedicalPrediction,
    target: MedicalPrediction,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Compute classification metrics for a single case."""
    from medeval.metrics.classification import compute_classification_metrics

    results = compute_classification_metrics(
        pred=pred.to_tensor(),
        target=target.to_tensor(),
        compute_ci=False,
        include_calibration=config.get("include_calibration", True),
        reduction="none",
    )

    # Convert tensors to floats
    output = {}
    for k, v in results.items():
        if isinstance(v, torch.Tensor):
            # Handle both scalar and multi-element tensors
            tensor_v: torch.Tensor = v  # Type narrowing for type checker
            if tensor_v.numel() > 1:
                output[k] = float(tensor_v.mean().item())
            else:
                output[k] = float(tensor_v.item())
        elif isinstance(v, tuple):
            output[k] = float(v[0])  # Take value, not CI
        else:
            output[k] = float(v)
    return output


def _compute_detection_metrics(
    pred: MedicalPrediction,
    target: MedicalPrediction,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Compute detection metrics for a single case."""
    from medeval.metrics.detection import compute_detection_metrics

    pred_boxes = pred.to_tensor()
    target_boxes = target.to_tensor()

    # Boxes are expected as a table:
    # - 2D: (N, 6)  -> [x1,y1,x2,y2,score,class_id]
    # - 3D: (N, 8)  -> [x1,y1,z1,x2,y2,z2,score,class_id]
    if pred_boxes.dim() != 2 or target_boxes.dim() != 2:
        raise ValueError(
            f"Detection expects (N,6) or (N,8) box tables; got pred={tuple(pred_boxes.shape)} target={tuple(target_boxes.shape)}"
        )

    use_3d = bool(config.get("use_3d", int(pred_boxes.shape[1]) == 8 or int(target_boxes.shape[1]) == 8))
    iou_thresholds = config.get("iou_thresholds", [0.5, 0.75])
    include_froc = bool(config.get("include_froc", False))

    results = compute_detection_metrics(
        pred_boxes,
        target_boxes,
        iou_thresholds=iou_thresholds,
        use_3d=use_3d,
        include_froc=include_froc,
    )

    # Keep only scalar outputs for CSV aggregation.
    out: Dict[str, float] = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            # FROC curve arrays are not per-case scalars; skip them in CSV output.
            continue
        if isinstance(v, torch.Tensor):
            out[k] = float(v.item())
        else:
            out[k] = float(v)
    return out


def _compute_registration_metrics(
    pred: MedicalPrediction,
    target: MedicalPrediction,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Compute registration metrics for a single case."""
    from medeval.metrics.registration import compute_registration_metrics

    results = compute_registration_metrics(
        pred_image=pred.to_tensor(),
        target_image=target.to_tensor(),
        spacing=pred.spacing,
        include_image_similarity=config.get("include_image_similarity", True),
        include_deformation_quality=False,
    )

    # Flatten nested results
    output = {}
    for k, v in results.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if not isinstance(v2, np.ndarray):
                    output[f"{k}_{k2}"] = float(v2)
        elif not isinstance(v, np.ndarray):
            output[k] = float(v)
    return output


TASK_METRICS = {
    "segmentation": _compute_segmentation_metrics,
    "classification": _compute_classification_metrics,
    "detection": _compute_detection_metrics,
    "registration": _compute_registration_metrics,
}


def _load_manifest(manifest_path: Path) -> pd.DataFrame:
    """
    Load manifest from CSV or JSON file.

    Parameters
    ----------
    manifest_path : Path
        Path to manifest file

    Returns
    -------
    pd.DataFrame
        Loaded manifest as DataFrame
    """
    if manifest_path.suffix == ".csv":
        return pd.read_csv(manifest_path)
    elif manifest_path.suffix == ".json":
        return pd.read_json(manifest_path)
    else:
        raise ValueError(f"Unsupported manifest format: {manifest_path.suffix}")


def _validate_manifest(df: pd.DataFrame, required_cols: List[str]) -> None:
    """
    Validate manifest has required columns.

    Parameters
    ----------
    df : pd.DataFrame
        Manifest DataFrame
    required_cols : List[str]
        List of required column names

    Raises
    ------
    ValueError
        If any required column is missing
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")


def _process_single_case(
    row: pd.Series,
    idx: int,
    metric_fn: Callable,
    metric_config: Dict[str, Any],
    col_config: Dict[str, str],
    base_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, float]], Optional[str]]:
    """
    Process a single case and return metrics.

    Parameters
    ----------
    row : pd.Series
        Manifest row
    idx : int
        Row index
    metric_fn : Callable
        Metric computation function
    metric_config : Dict[str, Any]
        Metric configuration
    col_config : Dict[str, str]
        Column name mappings

    Returns
    -------
    Tuple[Dict[str, Any], Optional[Dict[str, float]], Optional[str]]
        (result_dict, metrics_dict, strata)
    """
    try:
        # Load prediction and target
        pred, target = _load_prediction_target(
            row,
            pred_col=col_config["prediction"],
            target_col=col_config["target"],
            base_dir=base_dir,
            spacing_col=col_config.get("spacing"),
            patient_col=col_config.get("patient_id"),
            strata_col=col_config.get("strata"),
        )

        # Compute metrics
        metrics = metric_fn(pred, target, metric_config)

        # Build result dict
        result = {
            "index": idx,
            "status": "success",
            "patient_id": pred.patient_id,
            "strata": pred.strata,
            **metrics,
        }

        return result, metrics, pred.strata

    except Exception as e:
        logger.warning(f"Error processing row {idx}: {e}")
        return {
            "index": idx,
            "status": "error",
            "error": str(e),
        }, None, None


def _save_results(all_results: List[Dict[str, Any]], output_dir: Path) -> None:
    """
    Save per-case results to CSV.

    Parameters
    ----------
    all_results : List[Dict[str, Any]]
        List of result dictionaries
    output_dir : Path
        Output directory
    """
    results_df = pd.DataFrame(all_results)
    results_path = output_dir / "results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Per-case results saved to {results_path}")


def _compute_summary(
    task: str,
    df: pd.DataFrame,
    all_results: List[Dict[str, Any]],
    all_metrics: Dict[str, List[float]],
    aggregated: Dict[str, Union[float, Tuple[float, float, float]]],
    stratified_results: Optional[Dict],
    config: Dict,
) -> Dict[str, Any]:
    """
    Compute aggregated summary with statistics.

    Parameters
    ----------
    task : str
        Task name
    df : pd.DataFrame
        Original manifest DataFrame
    all_results : List[Dict[str, Any]]
        All per-case results
    all_metrics : Dict[str, List[float]]
        Accumulated metrics
    aggregated : Dict[str, Union[float, Tuple[float, float, float]]]
        Aggregated metrics with CI
    stratified_results : Optional[Dict]
        Stratified aggregation results
    config : Dict
        Configuration dictionary

    Returns
    -------
    Dict[str, Any]
        Summary dictionary
    """
    summary = {
        "task": task,
        "n_samples": len(df),
        "n_processed": len([r for r in all_results if r.get("status") == "success"]),
        "n_errors": len([r for r in all_results if r.get("status") == "error"]),
        "config": config,
        "metrics": {
            k: {
                "mean": v[0] if isinstance(v, tuple) else v,
                "ci_lower": v[1] if isinstance(v, tuple) else None,
                "ci_upper": v[2] if isinstance(v, tuple) else None,
            }
            for k, v in aggregated.items()
        },
    }

    if stratified_results:
        summary["stratified_metrics"] = stratified_results

    # Compute additional statistics
    for metric_name, values in all_metrics.items():
        values_arr = np.asarray(values, dtype=float)
        finite_mask = np.isfinite(values_arr)
        summary["metrics"][metric_name]["n_finite"] = int(finite_mask.sum())

        finite_vals = values_arr[finite_mask]
        if finite_vals.size == 0:
            summary["metrics"][metric_name]["std"] = None
            summary["metrics"][metric_name]["median"] = None
            summary["metrics"][metric_name]["iqr"] = None
            continue

        summary["metrics"][metric_name]["std"] = float(np.std(finite_vals))
        summary["metrics"][metric_name]["median"] = float(np.median(finite_vals))
        summary["metrics"][metric_name]["iqr"] = [
            float(np.percentile(finite_vals, 25)),
            float(np.percentile(finite_vals, 75)),
        ]

    return summary


def _print_summary(summary: Dict[str, Any]) -> None:
    """
    Print evaluation summary to console.

    Parameters
    ----------
    summary : Dict[str, Any]
        Summary dictionary
    """
    task = summary["task"]
    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY - {task.upper()}")
    print("=" * 60)
    print(f"Processed: {summary['n_processed']}/{summary['n_samples']} cases")
    if summary['n_errors'] > 0:
        print(f"Errors: {summary['n_errors']}")
    print("-" * 60)
    print("METRICS (mean [95% CI]):")
    for metric_name, metric_data in summary["metrics"].items():
        if metric_name.startswith("_"):
            continue
        mean = metric_data["mean"]
        ci_lower = metric_data.get("ci_lower")
        ci_upper = metric_data.get("ci_upper")
        if ci_lower is not None and ci_upper is not None:
            print(f"  {metric_name}: {mean:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
        else:
            print(f"  {metric_name}: {mean:.4f}")
    print("=" * 60 + "\n")


def evaluate_command(args, config: Dict) -> int:
    """
    Execute evaluation command.

    Parameters
    ----------
    args
        Parsed command-line arguments
    config : Dict
        Configuration dictionary

    Returns
    -------
    int
        Exit code
    """
    manifest_path = args.manifest
    output_dir = args.output or Path("results")
    task = args.task or config.get("task", "segmentation")

    logger.info(f"Loading manifest from {manifest_path}")
    logger.info(f"Task: {task}")
    logger.info(f"Output directory: {output_dir}")

    # Validate task
    if task not in TASK_METRICS:
        logger.error(f"Unknown task: {task}. Valid tasks: {list(TASK_METRICS.keys())}")
        return 1

    # Load manifest
    try:
        df = _load_manifest(manifest_path)
    except ValueError as e:
        logger.error(str(e))
        return 1

    logger.info(f"Loaded {len(df)} entries from manifest")

    # Get column mappings from config
    col_config = config.get("columns", {})
    pred_col = col_config.get("prediction", "prediction")
    target_col = col_config.get("target", "target")
    col_config = {
        "prediction": pred_col,
        "target": target_col,
        "spacing": col_config.get("spacing", "spacing"),
        "patient_id": col_config.get("patient_id", "patient_id"),
        "strata": col_config.get("strata", "strata"),
    }

    # Validate required columns
    try:
        _validate_manifest(df, [pred_col, target_col])
    except ValueError as e:
        logger.error(str(e))
        return 1

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get metric function
    metric_fn = TASK_METRICS[task]
    metric_config = config.get("metrics", {})

    # Process entries with progress bar
    all_results: List[Dict[str, Any]] = []
    all_metrics: Dict[str, List[float]] = {}
    strata_data: List[Optional[str]] = []

    base_dir = Path(manifest_path).parent

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating", unit="case"):
        result, metrics, strata = _process_single_case(
            row, idx, metric_fn, metric_config, col_config, base_dir=base_dir
        )
        all_results.append(result)

        if metrics is not None:
            # Accumulate for aggregation
            for k, v in metrics.items():
                if k not in all_metrics:
                    all_metrics[k] = []
                all_metrics[k].append(v)
            strata_data.append(strata)

    # Save per-case results
    _save_results(all_results, output_dir)

    # Aggregate metrics with CI
    logger.info("Aggregating metrics...")
    agg_config = config.get("aggregation", {})
    n_bootstrap = agg_config.get("n_bootstrap", 1000)
    confidence = agg_config.get("confidence", 0.95)

    # Omit non-finite values (NaN/inf) from aggregation.
    metrics_for_agg: Dict[str, np.ndarray] = {}
    for k, vals in all_metrics.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        metrics_for_agg[k] = arr

    aggregated = aggregate_metrics(
        metrics_for_agg,
        method="mean",
        compute_ci=True,
        ci_method="bootstrap",
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        seed=agg_config.get("seed", 42),
    )

    # Stratified aggregation if strata available.
    # NOTE: metrics can contain NaN/inf (e.g., surface metrics when spacing is missing).
    # We compute stratified summaries per-metric using finite values only.
    stratified_results = None

    # Collect successful rows into a DataFrame for easy filtering
    success_rows = [r for r in all_results if r.get("status") == "success"]
    if success_rows:
        success_df = pd.DataFrame(success_rows)
        if "strata" in success_df.columns:
            success_df["strata"] = success_df["strata"].fillna("unknown").astype(str)

            unique_strata = sorted(set(success_df["strata"].tolist()))
            if len(unique_strata) > 1:
                logger.info(f"Computing stratified metrics for {len(unique_strata)} strata...")

                stratified_results = {}
                seed = agg_config.get("seed", 42)

                # Only stratify over real metric keys (skip helper fields)
                metric_keys = [k for k in all_metrics.keys() if not str(k).startswith("_")]

                for stratum in unique_strata:
                    sub = success_df[success_df["strata"] == stratum]
                    if sub.empty:
                        continue

                    stratified_results[stratum] = {}

                    for k in metric_keys:
                        if k not in sub.columns:
                            continue
                        vals = pd.to_numeric(sub[k], errors="coerce").to_numpy(dtype=float)
                        vals = vals[np.isfinite(vals)]
                        if vals.size == 0:
                            continue

                        agg_k = aggregate_metrics(
                            {k: vals},
                            method="mean",
                            compute_ci=True,
                            ci_method="bootstrap",
                            confidence=confidence,
                            n_bootstrap=n_bootstrap,
                            seed=seed,
                        )
                        # aggregate_metrics returns {k: (mean, lo, hi)} when compute_ci=True
                        stratified_results[stratum][k] = agg_k[k]

                # If everything got filtered out, keep None
                if not stratified_results:
                    stratified_results = None

    # Build and save summary
    summary = _compute_summary(
        task, df, all_results, all_metrics, aggregated, stratified_results, config
    )

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Summary saved to {summary_path}")

    # Save resolved config for reproducibility
    resolved_config = {
        "task": task,
        "manifest": str(manifest_path),
        "columns": col_config,
        "metrics": metric_config,
        "aggregation": {
            "method": "mean",
            "ci_method": "bootstrap",
            "confidence": confidence,
            "n_bootstrap": n_bootstrap,
            "seed": agg_config.get("seed", 42),
        },
        "environment": _get_version_fingerprint(),
    }
    config_path = output_dir / "run_config_resolved.yaml"
    with open(config_path, "w") as f:
        yaml.dump(resolved_config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Resolved config saved to {config_path}")

    # Print summary to console
    _print_summary(summary)

    return 0
