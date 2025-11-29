"""Core utilities for medical imaging evaluation."""

from medeval.core.aggregate import (
    aggregate_metrics,
    bootstrap_ci,
    jackknife_ci,
    stratified_aggregate,
)
from medeval.core.io import (
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
    apply_spacing,
    compute_one_hot,
    compute_weights,
    label_mapping,
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
    "apply_spacing",
    "sample_with_spacing",
    "label_mapping",
    "compute_one_hot",
    "reduce_metrics",
    "compute_weights",
    # IO
    "load_image",
    "save_image",
    "load_nifti",
    "save_nifti",
    "load_sitk",
    "save_sitk",
    # Aggregate
    "aggregate_metrics",
    "bootstrap_ci",
    "jackknife_ci",
    "stratified_aggregate",
]
