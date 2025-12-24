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
from medeval.core import compute_metric
```

## License

Apache-2.0

# test acc1
# test acc2