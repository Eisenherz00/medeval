"""Tests for typing utilities."""

import numpy as np
import pytest
import torch

from medeval.core.typing import (
    as_tensor,
    get_device,
    get_dtype,
    to_device,
    to_dtype,
)


def test_as_tensor_from_numpy():
    """Test converting numpy array to tensor."""
    arr = np.array([1, 2, 3], dtype=np.float32)
    tensor = as_tensor(arr)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert torch.equal(tensor, torch.tensor([1, 2, 3], dtype=torch.float32))


def test_as_tensor_from_list():
    """Test converting list to tensor."""
    lst = [1, 2, 3]
    tensor = as_tensor(lst)

    assert isinstance(tensor, torch.Tensor)


def test_as_tensor_with_dtype():
    """Test as_tensor with dtype specification."""
    arr = np.array([1, 2, 3], dtype=np.int32)
    tensor = as_tensor(arr, dtype=torch.float64)

    assert tensor.dtype == torch.float64


def test_as_tensor_with_device():
    """Test as_tensor with device specification."""
    arr = np.array([1, 2, 3])
    tensor = as_tensor(arr, device="cpu")

    assert tensor.device.type == "cpu"


def test_to_device():
    """Test moving tensor to device."""
    tensor = torch.tensor([1, 2, 3])
    tensor_cpu = to_device(tensor, "cpu")

    assert tensor_cpu.device.type == "cpu"


def test_to_dtype():
    """Test converting tensor dtype."""
    tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
    tensor_float = to_dtype(tensor, torch.float32)

    assert tensor_float.dtype == torch.float32


def test_get_device():
    """Test getting tensor device."""
    tensor = torch.tensor([1, 2, 3])
    device = get_device(tensor)

    assert isinstance(device, torch.device)


def test_get_dtype():
    """Test getting tensor dtype."""
    tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
    dtype = get_dtype(tensor)

    assert dtype == torch.float32

