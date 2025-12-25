"""Data containers for medical imaging predictions and targets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from medeval.core.typing import ArrayLike, Device, Tensor, as_tensor


@dataclass
class MedicalPrediction:
    """
    Container for medical imaging predictions with metadata.

    Attributes
    ----------
    data : ArrayLike
        Prediction/target data (numpy array or torch tensor)
        Typical shapes (depending on pipeline conventions):
        - (H, W)                    2D image
        - (H, W, D)                 3D volume (channel-last, no batch/channel)
        - (B, H, W)                 batch of 2D images
        - (B, Z, Y, X)              batch of 3D volumes without channel
        - (B, C, Y, X)              batch of 2D images with channel
        - (B, C, Z, Y, X)           batch of 3D volumes with channel
    spacing : Tuple[float, ...], optional
        Physical spacing per spatial axis:
        - 2D: (dy, dx)
        - 3D: (dz, dy, dx)   (or any consistent convention, but length must match spatial dims)
    origin, direction, patient_id, study_id, strata, metadata
        Additional metadata.
    """

    data: ArrayLike
    spacing: Optional[Tuple[float, ...]] = None
    origin: Optional[Tuple[float, ...]] = None
    direction: Optional[np.ndarray] = None
    patient_id: Optional[str] = None
    study_id: Optional[str] = None
    strata: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize data after initialization."""
        if self.data is None:
            raise ValueError("data cannot be None")

        if self.spacing is not None:
            spatial_dims = self._get_spatial_dims()
            if len(self.spacing) != spatial_dims:
                data_shape = self.to_tensor().shape
                raise ValueError(
                    f"Spacing dimension {len(self.spacing)} does not match."
                    f"spatial dimensions {spatial_dims} for data shape {data_shape}."
                    f"Expected spacing with {spatial_dims} values, got {len(self.spacing)}."
                )

    def _get_spatial_dims(self) -> int:
        """Infer number of spatial dimensions from tensor shape.

        This repo is not fully standardized yet, so we use a small set of safe rules:
        - dim<=2: return dim ((), (N,), (H,W))
        - dim==3: treat as 3D by default (common for volumes: (H,W,D))
        - dim==4: either (B,C,Y,X)[2D] or (B,Z,Y,X)[3D]
                 heuristic: axis-1 in {1,2,3,4} => channel => 2D else 3D
        - dim==5: (B,C,Z,Y,X) => 3D
        - else: fallback assume (B,C,...) => spatial dims = dim - 2
        """
        data = self.to_tensor()
        d = data.dim()

        if d <= 2:
            return d

        if d == 3:
            return 3

        if d == 4:
            c_or_z = int(data.shape[1])
            if c_or_z in (1, 2, 3, 4):
                return 2
            return 3

        if d == 5:
            return 3

        return max(d - 2, 1)

    def to_tensor(self, device: Optional[Device] = None) -> Tensor:
        """Convert data to PyTorch tensor."""
        return as_tensor(self.data, device=device)

    def to_numpy(self) -> np.ndarray:
        """Convert data to numpy array."""
        tensor = self.to_tensor()
        return tensor.cpu().numpy()

    def get_physical_shape(self) -> Optional[Tuple[float, ...]]:
        """Get physical dimensions of the data (spacing * voxel count)."""
        if self.spacing is None:
            return None

        data = self.to_tensor()
        spatial_shape = data.shape[-len(self.spacing) :]
        return tuple(s * n for s, n in zip(self.spacing, spatial_shape))

    def get_affine(self) -> np.ndarray:
        """Get an approximate 4x4 affine matrix from spacing/origin/direction (if provided)."""
        affine = np.eye(4)

        if self.spacing is not None:
            for i, s in enumerate(self.spacing[:3]):
                affine[i, i] = s

        if self.origin is not None:
            for i, o in enumerate(self.origin[:3]):
                affine[i, 3] = o

        if self.direction is not None:
            for i in range(min(3, self.direction.shape[0])):
                for j in range(min(3, self.direction.shape[1])):
                    affine[i, j] = self.direction[i, j] * (self.spacing[i] if self.spacing else 1.0)

        return affine

    @classmethod
    def from_nifti(cls, path: str, **kwargs) -> "MedicalPrediction":
        """Create MedicalPrediction from a NIfTI file."""
        from medeval.core.io import get_nifti_spacing, load_nifti

        data = load_nifti(path, as_torch=True)
        spacing = get_nifti_spacing(path)
        return cls(data=data, spacing=spacing, **kwargs)

    @classmethod
    def from_sitk(cls, path: str, **kwargs) -> "MedicalPrediction":
        """Create MedicalPrediction from a SimpleITK-loadable image file."""
        from medeval.core.io import get_sitk_spacing, load_sitk

        data = load_sitk(path, as_torch=True)
        spacing = get_sitk_spacing(path)
        return cls(data=data, spacing=spacing, **kwargs)


@dataclass
class EvaluationBatch:
    """Container for a batch of predictions and targets for evaluation."""

    predictions: List[MedicalPrediction]
    targets: List[MedicalPrediction]

    def __post_init__(self) -> None:
        if len(self.predictions) != len(self.targets):
            raise ValueError(
                f"Number of predictions ({len(self.predictions)}) must match "
                f"number of targets ({len(self.targets)})"
            )

    def __len__(self) -> int:
        return len(self.predictions)

    def __iter__(self):
        return zip(self.predictions, self.targets)

    def get_patient_ids(self) -> List[Optional[str]]:
        return [p.patient_id for p in self.predictions]

    def get_strata(self) -> List[Optional[str]]:
        return [p.strata for p in self.predictions]

    def group_by_patient(self) -> Dict[str, "EvaluationBatch"]:
        groups: Dict[str, Tuple[List[MedicalPrediction], List[MedicalPrediction]]] = {}
        for pred, target in self:
            patient_id = pred.patient_id or "unknown"
            if patient_id not in groups:
                groups[patient_id] = ([], [])
            groups[patient_id][0].append(pred)
            groups[patient_id][1].append(target)

        return {
            patient_id: EvaluationBatch(predictions=preds, targets=targets)
            for patient_id, (preds, targets) in groups.items()
        }

    def group_by_stratum(self) -> Dict[str, "EvaluationBatch"]:
        groups: Dict[str, Tuple[List[MedicalPrediction], List[MedicalPrediction]]] = {}
        for pred, target in self:
            stratum = pred.strata or "unknown"
            if stratum not in groups:
                groups[stratum] = ([], [])
            groups[stratum][0].append(pred)
            groups[stratum][1].append(target)

        return {
            stratum: EvaluationBatch(predictions=preds, targets=targets)
            for stratum, (preds, targets) in groups.items()
        }

    def to_tensors(self, device: Optional[Device] = None) -> Tuple[Tensor, Tensor]:
        pred_tensors = [p.to_tensor(device) for p in self.predictions]
        target_tensors = [t.to_tensor(device) for t in self.targets]
        return torch.stack(pred_tensors), torch.stack(target_tensors)

    def get_common_spacing(self) -> Optional[Tuple[float, ...]]:
        spacings = [p.spacing for p in self.predictions if p.spacing is not None]
        if not spacings:
            return None

        first_spacing = spacings[0]
        for spacing in spacings[1:]:
            if spacing != first_spacing:
                return None
        return first_spacing