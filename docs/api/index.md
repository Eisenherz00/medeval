# API Reference

## Core Modules

### medeval.core.typing

Type aliases and device/dtype utilities.

```python
from medeval.core.typing import as_tensor, to_device, to_dtype
```

### medeval.core.io

IO adapters for medical image formats.

```python
from medeval.core.io import load_nifti, save_nifti, get_nifti_spacing
```

### medeval.core.utils

Core utilities for spacing, label mapping, and reduction.

```python
from medeval.core.utils import apply_spacing, compute_one_hot, reduce_metrics
```

### medeval.core.aggregate

Statistical aggregation with confidence intervals.

```python
from medeval.core.aggregate import bootstrap_ci, jackknife_ci, aggregate_metrics, stratified_aggregate
```

### medeval.core.containers

Data containers for predictions and targets.

```python
from medeval.core.containers import MedicalPrediction, EvaluationBatch
```

## Metrics Modules

### medeval.metrics.segmentation

Segmentation metrics.

```python
from medeval.metrics.segmentation import (
    dice_score,
    jaccard_index,
    precision_score,
    recall_score,
    volumetric_similarity,
    hausdorff_distance,
    hausdorff_distance_95,
    average_symmetric_surface_distance,
    surface_dice,
    soft_dice_score,
    brier_score,
    compute_segmentation_metrics,
)
```

### medeval.metrics.classification

Classification and calibration metrics.

```python
from medeval.metrics.classification import (
    auroc,
    auprc,
    accuracy,
    balanced_accuracy,
    sensitivity,
    specificity,
    f1_score_metric,
    mcc,
    cohen_kappa,
    expected_calibration_error,
    reliability_diagram,
    youden_threshold,
    decision_curve,
    compute_classification_metrics,
)
```

### medeval.metrics.detection

Detection metrics.

```python
from medeval.metrics.detection import (
    box_iou_2d,
    box_iou_3d,
    mean_average_precision,
    froc,
    average_recall,
    hungarian_matching,
    instance_segmentation_matching,
)
```

### medeval.metrics.registration

Registration metrics.

```python
from medeval.metrics.registration import (
    target_registration_error,
    normalized_mutual_information,
    normalized_cross_correlation,
    jacobian_determinant,
    bending_energy,
    compute_registration_metrics,
)
```

## Visualization Module

### medeval.vis

Visualization utilities (requires matplotlib).

```python
from medeval.vis import (
    # Curves
    plot_roc_curve,
    plot_pr_curve,
    plot_froc_curve,
    # Calibration
    plot_reliability_diagram,
    plot_decision_curve,
    # Overlays
    plot_segmentation_overlay,
    plot_prediction_comparison,
    # Histograms
    plot_error_histogram,
    plot_metric_distribution,
)
```

