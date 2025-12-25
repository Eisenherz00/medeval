# MedEval

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A PyTorch-native library for computing, aggregating, and visualizing evaluation metrics for medical imaging tasks with 2D/3D support, correct physical spacing handling, confidence intervals, and clean APIs (Python + CLI).

## Features

- **🏥 Medical Imaging Focus**: Designed specifically for medical imaging evaluation tasks
- **📐 2D/3D Support**: Handle both 2D and 3D medical images with proper dimension handling
- **📏 Physical Spacing**: Correct handling of anisotropic voxels and physical dimensions
- **📊 Confidence Intervals**: Bootstrap and jackknife methods for statistical reporting
- **🔄 Interoperability**: Works seamlessly with MONAI and torchmetrics
- **💻 Clean APIs**: Both Python API and CLI for batch evaluation
- **🚫 No PHI**: Designed with privacy in mind, no protected health information handling

## Installation

```bash
# Basic installation
pip install medeval

# With DICOM support
pip install medeval[dicom]

# For development
pip install -e ".[dev]"
```

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
  url = {https://github.com/your-repo/medeval}
}
```

## License

Apache-2.0. See [LICENSE](LICENSE) for details.
