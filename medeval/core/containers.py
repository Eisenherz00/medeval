"""Data containers for medical imaging predictions and targets."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from medeval.core.typing import ArrayLike, Device, Tensor, as_tensor


@dataclass
class MedicalPrediction:
    """
    Container for medical imaging predictions with metadata.
    
    This class provides a standardized way to pass predictions along with
    their associated metadata (spacing, origin, direction, patient/study ID, etc.)
    through the evaluation pipeline.
    
    Attributes
    ----------
    data : ArrayLike
        Prediction data (numpy array or torch tensor)
        Shape: (B, C, Z, Y, X), (B, C, Y, X), (B, Z, Y, X), or (B, Y, X)
    spacing : Tuple[float, ...], optional
        Physical spacing (dz, dy, dx) or (dy, dx) for 2D
    origin : Tuple[float, ...], optional
        Physical origin coordinates
    direction : np.ndarray, optional
        Direction cosine matrix (3x3 for 3D, 2x2 for 2D)
    patient_id : str, optional
        Patient or subject identifier
    study_id : str, optional
        Study or session identifier
    strata : str, optional
        Stratification label (e.g., site, scanner, institution)
    metadata : Dict[str, Any], optional
        Additional metadata
        
    Examples
    --------
    >>> pred = MedicalPrediction(
    ...     data=torch.rand(1, 1, 64, 64, 64),
    ...     spacing=(2.0, 1.0, 1.0),
    ...     patient_id="patient_001",
    ...     strata="site_A"
    ... )
    >>> pred.to_tensor()
    >>> pred.get_physical_shape()
    """
    
    data: ArrayLike
    spacing: Optional[Tuple[float, ...]] = None
    origin: Optional[Tuple[float, ...]] = None
    direction: Optional[np.ndarray] = None
    patient_id: Optional[str] = None
    study_id: Optional[str] = None
    strata: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize data after initialization."""
        # Ensure data is valid
        if self.data is None:
            raise ValueError("data cannot be None")
        
        # Validate spacing dimensions match data
        if self.spacing is not None:
            spatial_dims = self._get_spatial_dims()
            if len(self.spacing) != spatial_dims:
                raise ValueError(
                    f"Spacing dimension {len(self.spacing)} does not match "
                    f"spatial dimensions {spatial_dims}"
                )
    
    def _get_spatial_dims(self) -> int:
        """Get number of spatial dimensions."""
        data = self.to_tensor()
        # Assume (B, C, ...) or (B, ...) format
        if data.dim() <= 2:
            return data.dim()
        elif data.dim() == 3:
            return 2  # (B, H, W) or (B, C, spatial)
        elif data.dim() == 4:
            return 2 if data.shape[1] > 1 else 3  # (B, C, H, W) or (B, Z, H, W)
        elif data.dim() == 5:
            return 3  # (B, C, Z, H, W)
        else:
            return data.dim() - 2  # Assume (B, C, ...)
    
    def to_tensor(self, device: Optional[Device] = None) -> Tensor:
        """
        Convert data to PyTorch tensor.
        
        Parameters
        ----------
        device : Device, optional
            Target device
            
        Returns
        -------
        Tensor
            Data as PyTorch tensor
        """
        return as_tensor(self.data, device=device)
    
    def to_numpy(self) -> np.ndarray:
        """
        Convert data to numpy array.
        
        Returns
        -------
        np.ndarray
            Data as numpy array
        """
        tensor = self.to_tensor()
        return tensor.cpu().numpy()
    
    def get_physical_shape(self) -> Optional[Tuple[float, ...]]:
        """
        Get physical dimensions of the data.
        
        Returns
        -------
        Tuple[float, ...], optional
            Physical dimensions (spacing * voxel count) or None if no spacing
        """
        if self.spacing is None:
            return None
        
        data = self.to_tensor()
        spatial_shape = data.shape[-len(self.spacing):]
        return tuple(s * n for s, n in zip(self.spacing, spatial_shape))
    
    def get_affine(self) -> np.ndarray:
        """
        Get affine transformation matrix.
        
        Returns
        -------
        np.ndarray
            4x4 affine matrix
        """
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
        """
        Create MedicalPrediction from NIfTI file.
        
        Parameters
        ----------
        path : str
            Path to NIfTI file
        **kwargs
            Additional arguments for MedicalPrediction
            
        Returns
        -------
        MedicalPrediction
            Loaded prediction with metadata
        """
        from medeval.core.io import load_nifti, get_nifti_spacing
        
        data = load_nifti(path, as_torch=True)
        spacing = get_nifti_spacing(path)
        
        return cls(data=data, spacing=spacing, **kwargs)
    
    @classmethod
    def from_sitk(cls, path: str, **kwargs) -> "MedicalPrediction":
        """
        Create MedicalPrediction from SimpleITK image.
        
        Parameters
        ----------
        path : str
            Path to image file
        **kwargs
            Additional arguments for MedicalPrediction
            
        Returns
        -------
        MedicalPrediction
            Loaded prediction with metadata
        """
        from medeval.core.io import load_sitk, get_sitk_spacing
        
        data = load_sitk(path, as_torch=True)
        spacing = get_sitk_spacing(path)
        
        return cls(data=data, spacing=spacing, **kwargs)


@dataclass
class EvaluationBatch:
    """
    Container for a batch of predictions and targets for evaluation.
    
    Attributes
    ----------
    predictions : List[MedicalPrediction]
        List of predictions
    targets : List[MedicalPrediction]
        List of corresponding targets
    """
    
    predictions: List[MedicalPrediction]
    targets: List[MedicalPrediction]
    
    def __post_init__(self):
        """Validate batch after initialization."""
        if len(self.predictions) != len(self.targets):
            raise ValueError(
                f"Number of predictions ({len(self.predictions)}) must match "
                f"number of targets ({len(self.targets)})"
            )
    
    def __len__(self) -> int:
        """Return batch size."""
        return len(self.predictions)
    
    def __iter__(self):
        """Iterate over (prediction, target) pairs."""
        return zip(self.predictions, self.targets)
    
    def get_patient_ids(self) -> List[Optional[str]]:
        """Get patient IDs from predictions."""
        return [p.patient_id for p in self.predictions]
    
    def get_strata(self) -> List[Optional[str]]:
        """Get strata labels from predictions."""
        return [p.strata for p in self.predictions]
    
    def group_by_patient(self) -> Dict[str, "EvaluationBatch"]:
        """
        Group batch by patient ID.
        
        Returns
        -------
        Dict[str, EvaluationBatch]
            Dictionary mapping patient IDs to sub-batches
        """
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
        """
        Group batch by stratum.
        
        Returns
        -------
        Dict[str, EvaluationBatch]
            Dictionary mapping strata to sub-batches
        """
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
        """
        Stack all predictions and targets into tensors.
        
        Parameters
        ----------
        device : Device, optional
            Target device
            
        Returns
        -------
        Tuple[Tensor, Tensor]
            Stacked predictions and targets
        """
        pred_tensors = [p.to_tensor(device) for p in self.predictions]
        target_tensors = [t.to_tensor(device) for t in self.targets]
        
        return torch.stack(pred_tensors), torch.stack(target_tensors)
    
    def get_common_spacing(self) -> Optional[Tuple[float, ...]]:
        """
        Get common spacing if all samples have the same spacing.
        
        Returns
        -------
        Tuple[float, ...], optional
            Common spacing or None if inconsistent
        """
        spacings = [p.spacing for p in self.predictions if p.spacing is not None]
        if not spacings:
            return None
        
        first_spacing = spacings[0]
        for spacing in spacings[1:]:
            if spacing != first_spacing:
                return None
        
        return first_spacing

