# MedEval Examples

This directory contains example scripts and notebooks demonstrating how to use MedEval for medical imaging evaluation.

## Available Examples

### Python Scripts

1. **`segmentation_demo.py`** - Demonstrates segmentation metrics:
   - Dice, Jaccard, Precision, Recall
   - Surface metrics (HD, HD95, ASSD, Surface Dice)
   - Spacing-aware computations
   - Metric aggregation with confidence intervals
   - Visualization

2. **`classification_demo.py`** - Demonstrates classification metrics:
   - AUROC, AUPRC with DeLong CI
   - Calibration metrics (ECE, AECE)
   - Youden threshold optimization
   - Decision curve analysis
   - ROC/PR curve plotting

3. **`cli_usage.sh`** - Shows CLI usage patterns:
   - Manifest format
   - Configuration files
   - Command examples

## Quick Start

### Segmentation Example

```python
import torch
from medeval.metrics.segmentation import compute_segmentation_metrics

# Your predictions and targets
pred = torch.rand(1, 1, 64, 64, 64) > 0.5
target = torch.rand(1, 1, 64, 64, 64) > 0.5
spacing = (2.0, 1.0, 1.0)  # Physical spacing in mm

# Compute all metrics
results = compute_segmentation_metrics(
    pred, target,
    spacing=spacing,
    include_surface=True,
    include_calibration=True,
)

print(f"Dice: {results['dice']:.4f}")
print(f"HD95: {results['hausdorff_95']:.2f} mm")
```

### Classification Example

```python
import numpy as np
from medeval.metrics.classification import auroc, compute_classification_metrics

# Your predictions and labels
probs = np.random.rand(1000)
labels = (np.random.rand(1000) > 0.7).astype(int)

# Compute AUROC with confidence interval
score, ci_lower, ci_upper = auroc(probs, labels, compute_ci=True)
print(f"AUROC: {score:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")

# Comprehensive metrics
results = compute_classification_metrics(probs, labels, include_calibration=True)
```

### CLI Usage

```bash
# Evaluate segmentation from manifest
medeval evaluate --manifest data/manifest.csv --task segmentation --output results/

# With configuration
medeval evaluate --manifest data/manifest.csv --config config.yaml -v
```

### Manifest Format

CSV manifest should have columns:
- `prediction`: Path to prediction file
- `target`: Path to ground truth file
- `spacing` (optional): Comma-separated spacing values
- `patient_id` (optional): Patient identifier
- `strata` (optional): Stratification label (e.g., site)

Example:
```csv
prediction,target,spacing,patient_id,strata
/data/pred1.nii.gz,/data/gt1.nii.gz,"1.0,1.0,2.0",patient_001,site_A
/data/pred2.nii.gz,/data/gt2.nii.gz,"1.0,1.0,2.0",patient_002,site_B
```

## Running Examples

```bash
# Install medeval with dev dependencies
pip install -e ".[dev]"

# Run segmentation demo
python examples/segmentation_demo.py

# Run classification demo
python examples/classification_demo.py

# View CLI usage
bash examples/cli_usage.sh
```

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- NumPy, SciPy, scikit-learn
- matplotlib (for visualization examples)

