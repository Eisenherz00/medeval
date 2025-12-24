# MedEval

A PyTorch-native library for computing, aggregating, and visualizing evaluation metrics for medical imaging tasks with 2D/3D support, correct physical spacing handling, confidence intervals, and clean APIs (Python + CLI).

## Features

- **2D/3D Support**: Handle both 2D and 3D medical images with proper spacing awareness
- **Physical Spacing**: Correct handling of anisotropic voxels and physical dimensions
- **Confidence Intervals**: Bootstrap and jackknife methods for statistical reporting
- **Clean APIs**: Python API and CLI for batch evaluation
- **Interoperability**: Works with MONAI and torchmetrics

## Installation

```bash
pip install medeval
```

For DICOM support:
```bash
pip install medeval[dicom]
```

For development:
```bash
pip install medeval[dev]
```

## Quick Start

```python
import torch
from medeval.core import load_nifti, aggregate_metrics

# Load image with spacing awareness
image = load_nifti("path/to/image.nii.gz", as_torch=TRUE, device="cpu")

# Aggregate metrics with confidence intervals
metrics = {"dice": torch.rand(100) * 0.5 + 0.5}
results = aggregate_metrics(metrics, compute_ci = True)
```

## License

Apache-2.0