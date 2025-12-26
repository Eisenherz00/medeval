"""Tests for IO module, especially NIfTI spacing handling."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from medeval.core.io import (
    get_nifti_spacing,
    load_nifti,
    save_nifti,
)

# Skip all NIfTI tests if nibabel is not installed
nib = pytest.importorskip("nibabel", reason="nibabel is required for NIfTI tests")


@pytest.mark.acceptance
def test_nifti_spacing_roundtrip():
    """Test that NIfTI spacing is correctly read and written."""
    # Create test data with anisotropic spacing
    data = np.random.rand(10, 20, 30).astype(np.float32)
    spacing = (2.0, 1.5, 0.5)  # Anisotropic: dz=2.0, dy=1.5, dx=0.5
    origin = (0.0, 0.0, 0.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        nifti_path = Path(tmpdir) / "test.nii.gz"

        # Save with spacing
        save_nifti(data, str(nifti_path), spacing=spacing, origin=origin)

        # Load and check spacing
        loaded_spacing = get_nifti_spacing(str(nifti_path))

        # Check that spacing is preserved (allowing for small floating point differences)
        assert len(loaded_spacing) == 3
        assert np.allclose(loaded_spacing, spacing, rtol=1e-5)


@pytest.mark.acceptance
def test_nifti_anisotropic_voxels():
    """Test handling of anisotropic voxels in NIfTI files."""
    # Create 3D data with known anisotropic spacing
    data = np.ones((5, 10, 15), dtype=np.float32)
    spacing = (3.0, 2.0, 1.0)  # Strongly anisotropic

    with tempfile.TemporaryDirectory() as tmpdir:
        nifti_path = Path(tmpdir) / "anisotropic.nii.gz"

        save_nifti(data, str(nifti_path), spacing=spacing)

        # Load data
        loaded_data = load_nifti(str(nifti_path), as_torch=False)
        loaded_spacing = get_nifti_spacing(str(nifti_path))

        # Verify data integrity
        assert loaded_data.shape == data.shape
        assert np.allclose(loaded_data, data)

        # Verify spacing
        assert np.allclose(loaded_spacing, spacing, rtol=1e-5)


@pytest.mark.acceptance
def test_nifti_2d_3d_support():
    """Test that both 2D and 3D images are handled correctly."""
    # Test 3D
    data_3d = np.random.rand(10, 20, 30).astype(np.float32)
    spacing_3d = (1.0, 1.0, 1.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        path_3d = Path(tmpdir) / "test_3d.nii.gz"
        save_nifti(data_3d, str(path_3d), spacing=spacing_3d)
        loaded_3d = load_nifti(str(path_3d), as_torch=False)
        assert loaded_3d.shape == data_3d.shape

        # Test 2D (squeeze one dimension)
        data_2d = data_3d[0, :, :]
        spacing_2d = (1.0, 1.0)
        path_2d = Path(tmpdir) / "test_2d.nii.gz"
        save_nifti(data_2d, str(path_2d), spacing=spacing_2d)
        loaded_2d = load_nifti(str(path_2d), as_torch=False)
        assert loaded_2d.shape == data_2d.shape


def test_load_nifti_as_torch():
    """Test loading NIfTI as PyTorch tensor."""
    data = np.random.rand(5, 10, 15).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        nifti_path = Path(tmpdir) / "test.nii.gz"
        save_nifti(data, str(nifti_path))

        # Load as torch tensor
        tensor = load_nifti(str(nifti_path), as_torch=True, device="cpu")
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == data.shape
        assert torch.allclose(tensor, torch.from_numpy(data))

