"""Interoperability helpers for MONAI and torchmetrics."""

from typing import Optional, Union

import torch

try:
    import monai
    from monai.transforms import Transform
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False
    Transform = type(None)

try:
    import torchmetrics
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False

from medeval.core.typing import ArrayLike, Device, Tensor, as_tensor


class MONAITransformWrapper:
    """
    Wrapper to use medeval utilities with MONAI transforms.
    """

    @staticmethod
    def apply_spacing_transform(
        data: dict,
        spacing_key: str = "spacing",
        keys: Optional[list] = None,
    ) -> dict:
        """
        Apply spacing-aware transform to MONAI data dictionary.

        Parameters
        ----------
        data : dict
            MONAI data dictionary
        spacing_key : str
            Key for spacing in data dict
        keys : list, optional
            Keys to transform. If None, uses all image keys.

        Returns
        -------
        dict
            Transformed data dictionary
        """
        if not HAS_MONAI:
            raise ImportError("MONAI is required. Install with: pip install monai")

        if keys is None:
            keys = [k for k in data.keys() if k.endswith("image") or k.endswith("label")]

        spacing = data.get(spacing_key)
        if spacing is None:
            return data

        result = data.copy()
        for key in keys:
            if key in data:
                from medeval.core.utils import sample_with_spacing

                img = as_tensor(data[key])
                resampled, _ = sample_with_spacing(img, spacing)
                result[key] = resampled

        return result


def torchmetrics_to_medeval(
    metric_value: Union[float, torch.Tensor],
    device: Optional[Device] = None,
) -> Tensor:
    """
    Convert torchmetrics metric value to medeval tensor format.

    Parameters
    ----------
    metric_value : float or Tensor
        Metric value from torchmetrics
    device : Device, optional
        Target device

    Returns
    -------
    Tensor
        MedEval tensor
    """
    if isinstance(metric_value, torch.Tensor):
        tensor = metric_value
    else:
        tensor = torch.tensor(metric_value)

    if device is not None:
        tensor = tensor.to(device)

    return tensor


def medeval_to_torchmetrics(
    metric_value: Tensor,
    compute: bool = True,
) -> Union[torch.Tensor, float]:
    """
    Convert medeval metric to torchmetrics-compatible format.

    Parameters
    ----------
    metric_value : Tensor
        MedEval metric tensor
    compute : bool
        If True, return computed value (float); else return tensor

    Returns
    -------
    Tensor or float
        Metric value compatible with torchmetrics
    """
    if compute:
        return float(metric_value.item())
    return metric_value


class MedEvalMetricWrapper:
    """
    Wrapper to use medeval metrics with torchmetrics interface.
    """

    def __init__(self, metric_fn, **metric_kwargs):
        """
        Initialize wrapper.

        Parameters
        ----------
        metric_fn : callable
            MedEval metric function
        **metric_kwargs
            Additional arguments for metric function
        """
        self.metric_fn = metric_fn
        self.metric_kwargs = metric_kwargs
        self.reset()

    def reset(self):
        """Reset metric state."""
        self.values = []

    def update(self, preds: ArrayLike, target: ArrayLike, **kwargs):
        """
        Update metric with new predictions and targets.

        Parameters
        ----------
        preds : ArrayLike
            Predictions
        target : ArrayLike
            Ground truth
        **kwargs
            Additional arguments passed to metric function
        """
        preds_tensor = as_tensor(preds)
        target_tensor = as_tensor(target)

        value = self.metric_fn(preds_tensor, target_tensor, **self.metric_kwargs, **kwargs)
        self.values.append(value)

    def compute(self) -> torch.Tensor:
        """
        Compute final metric value.

        Returns
        -------
        Tensor
            Aggregated metric value
        """
        if not self.values:
            return torch.tensor(0.0)

        values_tensor = torch.stack(self.values) if isinstance(self.values[0], torch.Tensor) else torch.tensor(self.values)
        return values_tensor.mean()

