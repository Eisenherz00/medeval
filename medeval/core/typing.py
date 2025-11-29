"""Type aliases and device/dtype utilities for tensor operations."""

from typing import Union

import numpy as np
import torch

# Type aliases
Tensor = torch.Tensor
ArrayLike = Union[np.ndarray, torch.Tensor, list]
TensorLike = Union[torch.Tensor, np.ndarray]
DType = Union[torch.dtype, np.dtype, type]
Device = Union[torch.device, str]


def as_tensor(
    x: ArrayLike,
    dtype: Union[DType, None] = None,
    device: Union[Device, None] = None,
) -> Tensor:
    """
    Convert array-like input to PyTorch tensor.

    Parameters
    ----------
    x : ArrayLike
        Input array (numpy array, torch tensor, or list)
    dtype : DType, optional
        Target dtype. If None, inferred from input.
    device : Device, optional
        Target device. If None, uses CPU.

    Returns
    -------
    Tensor
        PyTorch tensor on specified device with specified dtype.
    """
    if isinstance(x, torch.Tensor):
        tensor = x
    elif isinstance(x, np.ndarray):
        tensor = torch.from_numpy(x)
    else:
        tensor = torch.tensor(x)

    if dtype is not None:
        tensor = to_dtype(tensor, dtype)
    if device is not None:
        tensor = to_device(tensor, device)

    return tensor


def to_device(x: Tensor, device: Device) -> Tensor:
    """
    Move tensor to specified device.

    Parameters
    ----------
    x : Tensor
        Input tensor
    device : Device
        Target device (torch.device or string like 'cuda:0', 'cpu')

    Returns
    -------
    Tensor
        Tensor on target device
    """
    if isinstance(device, str):
        device = torch.device(device)
    return x.to(device)


def to_dtype(x: Tensor, dtype: DType) -> Tensor:
    """
    Convert tensor to specified dtype.

    Parameters
    ----------
    x : Tensor
        Input tensor
    dtype : DType
        Target dtype (torch.dtype, np.dtype, or Python type)

    Returns
    -------
    Tensor
        Tensor with target dtype
    """
    if isinstance(dtype, np.dtype):
        # Map numpy dtypes to torch dtypes
        dtype_map = {
            np.float32: torch.float32,
            np.float64: torch.float64,
            np.int32: torch.int32,
            np.int64: torch.int64,
            np.uint8: torch.uint8,
            np.int8: torch.int8,
            np.int16: torch.int16,
        }
        dtype = dtype_map.get(dtype.type, torch.float32)
    elif isinstance(dtype, type):
        # Map Python types to torch dtypes
        dtype_map = {
            float: torch.float32,
            int: torch.int64,
            bool: torch.bool,
        }
        dtype = dtype_map.get(dtype, torch.float32)
    elif not isinstance(dtype, torch.dtype):
        dtype = torch.float32  # fallback

    return x.to(dtype)


def get_device(x: Tensor) -> torch.device:
    """
    Get device of tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor

    Returns
    -------
    torch.device
        Device of the tensor
    """
    return x.device


def get_dtype(x: Tensor) -> torch.dtype:
    """
    Get dtype of tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor

    Returns
    -------
    torch.dtype
        Dtype of the tensor
    """
    return x.dtype

