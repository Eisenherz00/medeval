# MedEval

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://img.shields.io/pypi/v/medeval.svg)](https://pypi.org/project/medeval/)
[![Downloads](https://img.shields.io/pypi/dm/medeval.svg)](https://pypi.org/project/medeval/)

A PyTorch-native library for computing, aggregating, and visualizing evaluation metrics for medical imaging tasks with 2D/3D support, correct physical spacing handling, confidence intervals, and clean APIs (Python + CLI).

## Features

- **🏥 Medical Imaging Focus**: Designed specifically for medical imaging evaluation tasks
- **📐 2D/3D Support**: Handle both 2D and 3D medical images with proper dimension handling
- **📏 Physical Spacing**: Correct handling of anisotropic voxels and physical dimensions
- **📊 Confidence Intervals**: Bootstrap and jackknife methods for statistical reporting
- **🔄 Interoperability**: Works seamlessly with MONAI and torchmetrics
- **💻 Clean APIs**: Both Python API and CLI for batch evaluation
- **🚫 No PHI**: Designed with privacy in mind, no protected health information handling

## Conventions

### Input Shape Support

| Input Shape | Interpretation | Normalized Shape |
|-------------|----------------|------------------|
| `(H, W)` | 2D single image | `(1, 1, H, W)` |
| `(B, H, W)` | 2D batch (B≤4) | `(B, 1, H, W)` |
| `(B, C, H, W)` | 2D batch+channel | unchanged |
| `(Z, Y, X)` | 3D single volume (Z>4) | `(1, 1, Z, Y, X)` |
| `(B, Z, Y, X)` | 3D batch (Z>4) | `(B, 1, Z, Y, X)` |
| `(B, C, Z, Y, X)` | 3D batch+channel | unchanged |

> **Note**: For 3-dim inputs, a heuristic distinguishes 2D-batched from 3D-unbatched by the first dimension size (threshold: 4). Use explicit 4D/5D shapes or provide `spacing` to disambiguate.

### Spacing Axis Order

- **2D**: `(dy, dx)` — row spacing, column spacing
- **3D**: `(dz, dy, dx)` — slice spacing, row spacing, column spacing

### Surface Metrics

Surface-based metrics (Hausdorff, ASSD, Surface Dice) require physical `spacing`. Without spacing:
- **Default**: metrics are skipped (no error, logged as skipped)
- **Strict mode**: use `require_spacing=True` to raise `ValueError`

## Installation

```bash
# Install from PyPI (recommended)
pip install medeval

# With DICOM support
pip install medeval[dicom]

# Minimal install (metrics only, no IO / plotting)
pip install medeval[minimal]

# For development
pip install -e ".[dev]"
```

## What MedEval Is (and Is Not)

**MedEval is an evaluation library, not a training framework.**

It assumes you already have model predictions and ground truth labels.
MedEval’s responsibility is to:
- Compute correct, spacing-aware medical metrics
- Aggregate results across cases, patients, and strata
- Produce reproducible CSV / JSON artifacts and plots

MedEval deliberately does **not**:
- Train models
- Tune hyperparameters
- Declare a “best model”

This separation mirrors real clinical research and deployment workflows.

## End-to-End Demo (External User Perspective)

This demo reflects how MedEval is intended to be used **after a model has already produced predictions**.

The `demo_external_user/` directory provides a **realistic, end-to-end example** of how an external user would interact with MedEval using their own data.  
It demonstrates **data preparation → evaluation → aggregation → visualization** using both the Python API and the CLI.

### TL;DR (10 lines)

1. `python demo_external_user/gen_data.py` generates synthetic segmentation + classification data (no PHI).
2. It writes a segmentation manifest with **100 cases**, patient IDs, and `strata` (site/scanner-like).
3. `python demo_external_user/run_example.py` is the recommended “real user” entry point.
4. It loads `manifest_seg.csv`, then iterates cases and reads each NIfTI (pred/target) + spacing metadata.
5. It normalizes shapes to `(B, C, *spatial)` and computes per-case metrics (Dice/IoU/HD95/ASSD/Surface Dice).
6. It saves per-case results to `out_example/segmentation_results.csv`.
7. It aggregates by `strata` and saves `out_example/segmentation_summary_by_strata.csv`.
8. It loads classification probs/labels, computes sample-level metrics + calibration (AUROC/AUPRC/Accuracy/ECE).
9. It aggregates classification to patient-level (grouped AUROC) and saves `out_example/classification_results.json`.
10. If `matplotlib` is installed, it also writes plots (Dice hist/box, HD95 hist, ROC, reliability diagram).

### What This Demo Simulates
- Multiple **patients**, **cases**, and **strata** (e.g. sites/scanners)
- 3D segmentation with physical voxel spacing
- Binary classification with per-patient aggregation
- Per-case metrics, stratified summaries, and plots

### Demo Structure

```
demo_external_user/
├── gen_data.py              # Generate synthetic medical data + manifests
├── manifest_seg.csv         # Segmentation manifest (100 cases, patients, strata)
├── run_example.py           # Main end-to-end Python workflow (recommended)
├── run_python_api.py        # Minimal direct API usage example
├── run_cli.sh               # CLI-based evaluation example
├── config_seg.yaml          # CLI configuration
├── data/                    # Generated synthetic inputs
└── out_example/             # Outputs: CSVs, JSON, plots
```

### Recommended Entry Point

Run the full example as a real user would:

```bash
python demo_external_user/gen_data.py
python demo_external_user/run_example.py
```

This will:
- Load a segmentation manifest with **100 cases**
- Compute per-case metrics (Dice, HD95, ASSD, etc.)
- Aggregate results by **strata**
- Compute classification metrics (sample-level + patient-level)
- Save CSV/JSON outputs
- Generate plots (Dice distribution, HD95 histogram, ROC, calibration)

### Typical Outputs

- `segmentation_results.csv` — per-case metrics  
- `segmentation_summary_by_strata.csv` — aggregated statistics  
- `classification_results.json` — sample-level and patient-level metrics  
- `dice_hist.png`, `dice_by_strata_box.png`, `hd95_hist.png`  
- `roc_curve.png`, `reliability_diagram.png`

This demo is the **best starting point** to understand MedEval’s design, APIs, and evaluation philosophy.

## Evaluation Scope and Design Philosophy

MedEval evaluates **one experiment (one model’s predictions) per run**.

It is intentionally designed to be:
- **Model-agnostic**: MedEval does not know or assume how predictions were generated.
- **Metric-neutral**: It computes metrics but does not decide which metric “matters most”.
- **Decision-neutral**: It does not declare a “best model”.

This design reflects real-world medical research and deployment practice, where:
- Different metrics capture different clinical trade-offs.
- Performance may vary across patients, strata (e.g. sites/scanners), or endpoints.
- Model selection and conclusions depend on study-specific goals, not on the evaluation tool itself.

MedEval’s role is to produce **reliable, comparable, and statistically sound evaluation artifacts**  
(CSV/JSON summaries, confidence intervals, plots) that enable downstream analysis.

## Multi-Model Comparison (External Analysis Layer)

While MedEval evaluates one model per run, it is **explicitly designed to support multi-model comparison** at a higher analysis layer.

The recommended workflow is:

1. Run MedEval **once per model**, using the same evaluation protocol.
2. Store each model’s outputs (CSVs, JSON summaries, plots).
3. Compare models using a separate analysis script or notebook.

### Recommended Directory Structure

```
experiments/
├── model_A/
│   └── out_example/
│       ├── segmentation_results.csv
│       ├── segmentation_summary_by_strata.csv
│       ├── classification_results.json
│       └── plots/
├── model_B/
│   └── out_example/
├── model_C/
│   └── out_example/
└── compare_models.ipynb
```

In this structure:
- **MedEval** is responsible only for generating per-model evaluation results.
- **compare_models.ipynb** (or an equivalent script) performs:
  - Cross-model aggregation
  - Statistical testing
  - Visualization of differences
  - Study-specific conclusions

This separation keeps MedEval reusable, transparent, and scientifically neutral.

This design avoids hidden coupling between models and metrics and ensures that
every comparison is explicit, auditable, and study-specific.

## Quick Start

### Segmentation Evaluation

```python
import torch
from medeval.metrics.segmentation import dice_score, hausdorff_distance_95, compute_segmentation_metrics

# Your predictions and ground truth
pred = torch.rand(1, 1, 64, 64, 64) > 0.5
target = torch.rand(1, 1, 64, 64, 64) > 0.5
spacing = (2.0, 1.0, 1.0)  # Physical spacing in mm (z, y, x)

# Compute individual metrics
dice = dice_score(pred, target, reduction="mean-case")
print(f"Dice: {dice.item():.4f}")

# Compute spacing-aware surface distance
hd95 = hausdorff_distance_95(pred, target, spacing=spacing)
print(f"HD95: {hd95.item():.2f} mm")

# Compute all metrics at once
results = compute_segmentation_metrics(
    pred, target,
    spacing=spacing,
    include_surface=True,
    include_calibration=True,
)
```

### Classification Evaluation

```python
import numpy as np
from medeval.metrics.classification import auroc, compute_classification_metrics

# Your predictions and labels
probs = np.random.rand(1000)
labels = (np.random.rand(1000) > 0.7).astype(int)

# Compute AUROC with DeLong confidence interval
score, ci_lower, ci_upper = auroc(probs, labels, compute_ci=True)
print(f"AUROC: {score:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")

# Comprehensive metrics with calibration
results = compute_classification_metrics(
    probs, labels,
    compute_ci=True,
    include_calibration=True,
)
```

### Detection Evaluation

```python
import torch
from medeval.metrics.detection import mean_average_precision, froc

# Predicted and ground truth boxes
pred_boxes = [torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]])]
pred_scores = [torch.tensor([0.9, 0.7])]
pred_labels = [torch.tensor([0, 1])]
target_boxes = [torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]])]
target_labels = [torch.tensor([0, 1])]

# Compute mAP
results = mean_average_precision(
    pred_boxes, pred_scores, pred_labels,
    target_boxes, target_labels,
)
print(f"mAP@[.50:.95]: {results['mAP@[.50:.95]']:.4f}")
```

### Registration Evaluation

```python
import numpy as np
from medeval.metrics.registration import target_registration_error, normalized_mutual_information

# Landmark-based evaluation
pred_landmarks = np.array([[10, 20, 30], [40, 50, 60]])
target_landmarks = np.array([[11, 21, 31], [41, 51, 61]])
spacing = (1.0, 1.0, 2.0)

tre_results = target_registration_error(pred_landmarks, target_landmarks, spacing=spacing)
print(f"TRE: {tre_results['mean']:.2f} mm (95th: {tre_results['95th_percentile']:.2f} mm)")
```

## CLI Usage

```bash
# Evaluate segmentation from manifest
medeval evaluate --manifest data/manifest.csv --task segmentation --output results/

# With configuration file
medeval evaluate --manifest data/manifest.csv --config config.yaml -v
```

### Manifest Format

```csv
prediction,target,spacing,patient_id,strata
/data/pred1.nii.gz,/data/gt1.nii.gz,"1.0,1.0,2.0",patient_001,site_A
/data/pred2.nii.gz,/data/gt2.nii.gz,"1.0,1.0,2.0",patient_002,site_B
```

## Supported Metrics

### Segmentation
| Metric | Description |
|--------|-------------|
| Dice Score (F1) | Overlap-based similarity |
| Jaccard Index (IoU) | Intersection over union |
| Precision/Recall | True positive rates |
| Volumetric Similarity | Volume-based similarity |
| Hausdorff Distance | Maximum surface distance |
| HD95 | 95th percentile Hausdorff |
| ASSD | Average symmetric surface distance |
| Surface Dice | Surface overlap at tolerance |

### Classification
| Metric | Description |
|--------|-------------|
| AUROC | Area under ROC curve (with DeLong CI) |
| AUPRC | Area under precision-recall curve |
| Accuracy | Overall correctness |
| Balanced Accuracy | Class-balanced accuracy |
| F1, MCC, Cohen's κ | Agreement metrics |
| ECE, AECE, TACE | Calibration errors |
| Brier Score | Probabilistic accuracy |

### Detection
| Metric | Description |
|--------|-------------|
| IoU | Box intersection over union (2D/3D) |
| mAP@[.50:.95] | Mean average precision |
| FROC | Free-response ROC |
| Average Recall | Detection recall |

### Registration
| Metric | Description |
|--------|-------------|
| TRE | Target registration error |
| NMI | Normalized mutual information |
| NCC | Normalized cross-correlation |
| Jacobian | Deformation field quality |

## Visualization

```python
from medeval.vis import plot_roc_curve, plot_reliability_diagram, plot_segmentation_overlay

# Plot ROC curve
fig, ax = plot_roc_curve(fpr, tpr, auc=auc_score, title="Model Performance")

# Plot calibration diagram
fig, ax = plot_reliability_diagram(
    bin_centers, accuracies, confidences, counts,
    ece=ece_value,
)

# Plot segmentation overlay
fig, ax = plot_segmentation_overlay(image, mask=ground_truth, prediction=pred)
```

## Statistical Reporting

MedEval provides comprehensive statistical reporting:

```python
from medeval.core.aggregate import aggregate_metrics, stratified_aggregate

# Aggregate with bootstrap CI
results = aggregate_metrics(
    {"dice": dice_scores, "hd95": hd95_scores},
    method="mean",
    compute_ci=True,
    confidence=0.95,
    n_bootstrap=1000,
)

# Stratified by site/scanner
stratified = stratified_aggregate(
    {"dice": dice_scores},
    strata=site_labels,
    compute_ci=True,
)
```

## MONAI Integration

```python
from medeval.core.interop import MedEvalMetricWrapper
from medeval.metrics.segmentation import dice_score

# Use medeval metrics with torchmetrics-style interface
metric = MedEvalMetricWrapper(dice_score, reduction="none")
metric.update(preds, targets)
result = metric.compute()
```

## Documentation

- [API Reference](docs/api/index.md)
- [Metric Definitions](docs/metrics/index.md)
- [CLI Reference](docs/cli.md)
- [Tutorials](docs/tutorials/index.md)

## Examples

See the [examples/](examples/) directory for:
- `segmentation_demo.py` - Comprehensive segmentation evaluation
- `classification_demo.py` - Classification with calibration analysis
- `cli_usage.sh` - Command-line interface examples

## Development

```bash
# Clone and install
git clone https://github.com/your-repo/medeval.git
cd medeval
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run tests with coverage
pytest --cov=medeval --cov-report=html

# Build docs
mkdocs serve
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use MedEval in your research, please cite:

```bibtex
@software{medeval2024,
  title = {MedEval: Medical Imaging Evaluation Metrics},
  year = {2024},
  url = {https://github.com/Eisenherz00/medeval}
}
```

## License

Apache-2.0. See [LICENSE](LICENSE) for details.
