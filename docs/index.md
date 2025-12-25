# MedEval Documentation

**MedEval** is a PyTorch-native library for computing, aggregating, and visualizing evaluation metrics for medical imaging tasks.

## Features

- **2D/3D Support**: Handle both 2D and 3D medical images
- **Physical Spacing**: Correct handling of anisotropic voxels and physical dimensions
- **Confidence Intervals**: Bootstrap and jackknife methods for statistical reporting
- **Clean APIs**: Python API and CLI for batch evaluation
- **Interoperability**: Works with MONAI and torchmetrics

## Quick Start

```python
import torch
from medeval.metrics.segmentation import dice_score, hausdorff_distance_95

# Create predictions and targets
pred = torch.rand(1, 1, 64, 64, 64) > 0.5
target = torch.rand(1, 1, 64, 64, 64) > 0.5
spacing = (2.0, 1.0, 1.0)  # Physical spacing in mm

# Compute Dice score
dice = dice_score(pred, target, reduction="mean-case")
print(f"Dice: {dice.item():.4f}")

# Compute spacing-aware surface distance
hd95 = hausdorff_distance_95(pred, target, spacing=spacing)
print(f"HD95: {hd95.item():.2f} mm")
```

## Installation

```bash
# Basic installation
pip install medeval

# With DICOM support
pip install medeval[dicom]

# For development
pip install medeval[dev]
```

## Supported Tasks

### Segmentation
- Overlap metrics: Dice, Jaccard, Precision, Recall, Volumetric Similarity
- Surface metrics: Hausdorff Distance (HD, HD95), ASSD, Surface Dice
- Calibration: Soft Dice, Brier Score

### Classification
- Discrimination: AUROC, AUPRC, Accuracy, Balanced Accuracy, F1, MCC, Cohen's κ
- Calibration: ECE, AECE, TACE, Brier Score, Reliability Diagrams
- Decision Analysis: Decision Curves, Youden Threshold

### Detection
- Object Detection: IoU, mAP@[.50:.95], FROC, Average Recall
- Instance Matching: Hungarian algorithm

### Registration
- Landmark-based: TRE (mean, median, 95th percentile)
- Image similarity: NMI, NCC, MIND-SSD
- Deformation quality: Jacobian determinants, Bending energy

## Contents

- [API Reference](api/index.md)
- [Tutorials](tutorials/index.md)
- [Metric Definitions](metrics/index.md)
- [CLI Reference](cli.md)
- [Contributing](contributing.md)

## License

Apache-2.0

