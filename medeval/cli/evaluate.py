"""Evaluation command implementation."""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from medeval.core.aggregate import aggregate_metrics, stratified_aggregate
from medeval.core.containers import EvaluationBatch, MedicalPrediction
from medeval.core.io import get_nifti_spacing, load_image, load_nifti
from medeval.core.typing import Tensor

logger = logging.getLogger(__name__)


def _load_prediction_target(
    row: pd.Series,
    pred_col: str = "prediction",
    target_col: str = "target",
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
    pred_path = str(row[pred_col])
    target_path = str(row[target_col])

    # Load data
    pred_data = load_image(pred_path, as_torch=True)
    target_data = load_image(target_path, as_torch=True)

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

    # Get metadata
    patient_id = str(row[patient_col]) if patient_col and patient_col in row and pd.notna(row[patient_col]) else None
    strata = str(row[strata_col]) if strata_col and strata_col in row and pd.notna(row[strata_col]) else None

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

    results = compute_segmentation_metrics(
        pred=pred.to_tensor(),
        target=target.to_tensor(),
        spacing=pred.spacing,
        threshold=config.get("threshold", 0.5),
        ignore_index=config.get("ignore_index"),
        include_surface=config.get("include_surface", True),
        include_calibration=config.get("include_calibration", False),
        reduction="none",
    )

    # Convert tensors to floats
    return {k: float(v.mean().item()) if isinstance(v, torch.Tensor) else float(v) 
            for k, v in results.items()}


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
            output[k] = float(v.mean().item())
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
    # Detection requires special handling for boxes/scores
    # This is a simplified version
    results = {
        "status": "detection_not_fully_implemented",
    }
    return results


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
    if manifest_path.suffix == ".csv":
        df = pd.read_csv(manifest_path)
    elif manifest_path.suffix == ".json":
        df = pd.read_json(manifest_path)
    else:
        logger.error(f"Unsupported manifest format: {manifest_path.suffix}")
        return 1

    logger.info(f"Loaded {len(df)} entries from manifest")

    # Get column mappings from config
    col_config = config.get("columns", {})
    pred_col = col_config.get("prediction", "prediction")
    target_col = col_config.get("target", "target")
    spacing_col = col_config.get("spacing", "spacing")
    patient_col = col_config.get("patient_id", "patient_id")
    strata_col = col_config.get("strata", "strata")

    # Validate required columns
    if pred_col not in df.columns:
        logger.error(f"Missing required column: {pred_col}")
        return 1
    if target_col not in df.columns:
        logger.error(f"Missing required column: {target_col}")
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

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating", unit="case"):
        try:
            # Load prediction and target
            pred, target = _load_prediction_target(
                row,
                pred_col=pred_col,
                target_col=target_col,
                spacing_col=spacing_col,
                patient_col=patient_col,
                strata_col=strata_col,
            )

            # Compute metrics
            metrics = metric_fn(pred, target, metric_config)

            # Store results
            result = {
                "index": idx,
                "status": "success",
                "patient_id": pred.patient_id,
                "strata": pred.strata,
                **metrics,
            }
            all_results.append(result)

            # Accumulate for aggregation
            for k, v in metrics.items():
                if k not in all_metrics:
                    all_metrics[k] = []
                all_metrics[k].append(v)
            strata_data.append(pred.strata)

        except Exception as e:
            logger.warning(f"Error processing row {idx}: {e}")
            all_results.append({
                "index": idx,
                "status": "error",
                "error": str(e),
            })

    # Save per-case results
    results_df = pd.DataFrame(all_results)
    results_path = output_dir / "results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Per-case results saved to {results_path}")

    # Aggregate metrics with CI
    logger.info("Aggregating metrics...")
    agg_config = config.get("aggregation", {})
    n_bootstrap = agg_config.get("n_bootstrap", 1000)
    confidence = agg_config.get("confidence", 0.95)

    aggregated = aggregate_metrics(
        {k: np.array(v) for k, v in all_metrics.items()},
        method="mean",
        compute_ci=True,
        ci_method="bootstrap",
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        seed=agg_config.get("seed", 42),
    )

    # Stratified aggregation if strata available
    stratified_results = None
    unique_strata = [s for s in set(strata_data) if s is not None]
    if len(unique_strata) > 1:
        logger.info(f"Computing stratified metrics for {len(unique_strata)} strata...")
        stratified_results = stratified_aggregate(
            {k: np.array(v) for k, v in all_metrics.items()},
            strata=np.array([s or "unknown" for s in strata_data]),
            method="mean",
            compute_ci=True,
            ci_method="bootstrap",
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            seed=agg_config.get("seed", 42),
        )

    # Build summary
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
        values_arr = np.array(values)
        summary["metrics"][metric_name]["std"] = float(np.std(values_arr))
        summary["metrics"][metric_name]["median"] = float(np.median(values_arr))
        summary["metrics"][metric_name]["iqr"] = [
            float(np.percentile(values_arr, 25)),
            float(np.percentile(values_arr, 75)),
        ]

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Summary saved to {summary_path}")

    # Print summary to console
    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY - {task.upper()}")
    print("=" * 60)
    print(f"Processed: {summary['n_processed']}/{summary['n_samples']} cases")
    if summary['n_errors'] > 0:
        print(f"Errors: {summary['n_errors']}")
    print("-" * 60)
    print("METRICS (mean [95% CI]):")
    for metric_name, metric_data in summary["metrics"].items():
        mean = metric_data["mean"]
        ci_lower = metric_data.get("ci_lower")
        ci_upper = metric_data.get("ci_upper")
        if ci_lower is not None and ci_upper is not None:
            print(f"  {metric_name}: {mean:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
        else:
            print(f"  {metric_name}: {mean:.4f}")
    print("=" * 60 + "\n")

    return 0
