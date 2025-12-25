"""Core utilities for medical imaging evaluation.

This module provides core functionality including:
- Type aliases and device/dtype utilities
- IO adapters for medical image formats
- Aggregation with confidence intervals
- Data containers for predictions and targets
- Interoperability with MONAI and torchmetrics
"""

from medeval.core.aggregate import (
    AggregationMethod,
    aggregate_metrics,
    bootstrap_ci,
    jackknife_ci,
    stratified_aggregate,
)
from medeval.core.containers import (
    EvaluationBatch,
    MedicalPrediction,
)
from medeval.core.interop import (
    MedEvalMetricWrapper,
    MONAITransformWrapper,
    medeval_to_torchmetrics,
    torchmetrics_to_medeval,
)
from medeval.core.io import (
    get_dicom_spacing,
    get_nifti_spacing,
    get_sitk_spacing,
    get_spacing_from_header,
    load_dicom,
    load_image,
    load_nifti,
    load_sitk,
    save_image,
    save_nifti,
    save_sitk,
)
from medeval.core.typing import (
    ArrayLike,
    Device,
    DType,
    Tensor,
    TensorLike,
    as_tensor,
    get_device,
    get_dtype,
    to_device,
    to_dtype,
)
from medeval.core.utils import (
    ReductionType,
    apply_spacing,
    compute_one_hot,
    compute_weights,
    get_spatial_dims_from_spacing,
    label_mapping,
    normalize_input_shapes,
    reduce_metrics,
    sample_with_spacing,
)

__all__ = [
    # Typing
    "Tensor",
    "TensorLike",
    "ArrayLike",
    "DType",
    "Device",
    "as_tensor",
    "to_device",
    "to_dtype",
    "get_device",
    "get_dtype",
    # Utils
    "ReductionType",
    "apply_spacing",
    "sample_with_spacing",
    "label_mapping",
    "compute_one_hot",
    "reduce_metrics",
    "compute_weights",
    "normalize_input_shapes",
    "get_spatial_dims_from_spacing",
    # IO
    "load_image",
    "save_image",
    "load_nifti",
    "save_nifti",
    "get_nifti_spacing",
    "load_sitk",
    "save_sitk",
    "get_sitk_spacing",
    "get_dicom_spacing",
    "get_spacing_from_header",
    "load_dicom",
    # Aggregate
    "AggregationMethod",
    "aggregate_metrics",
    "bootstrap_ci",
    "jackknife_ci",
    "stratified_aggregate",
    # Containers
    "MedicalPrediction",
    "EvaluationBatch",
    # Interop
    "MONAITransformWrapper",
    "MedEvalMetricWrapper",
    "torchmetrics_to_medeval",
    "medeval_to_torchmetrics",
]
