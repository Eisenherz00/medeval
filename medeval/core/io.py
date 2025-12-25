"""IO adapters for NumPy, PyTorch, SimpleITK/NIfTI, and optional DICOM."""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

from medeval.core.typing import ArrayLike, Device, Tensor, as_tensor


def load_image(
    path: str,
    as_torch: bool = True,
    device: Optional[Device] = None,
) -> Union[Tensor, np.ndarray]:
    """
    Load image from file, auto-detecting format.

    Parameters
    ----------
    path : str
        Path to image file
    as_torch : bool
        If True, return PyTorch tensor; else numpy array
    device : Device, optional
        Device for tensor (if as_torch=True)

    Returns
    -------
    Tensor or np.ndarray
        Loaded image
    """
    path_lower = path.lower()
    if path_lower.endswith((".nii", ".nii.gz")):
        return load_nifti(path, as_torch=as_torch, device=device)
    elif HAS_SITK and path_lower.endswith((".mha", ".mhd", ".nrrd", ".nii", ".nii.gz")):
        return load_sitk(path, as_torch=as_torch, device=device)
    else:
        # Try numpy
        arr = np.load(path)
        if as_torch:
            return as_tensor(arr, device=device)
        return arr


def save_image(
    data: ArrayLike,
    path: str,
    spacing: Optional[Tuple[float, ...]] = None,
    origin: Optional[Tuple[float, ...]] = None,
    direction: Optional[np.ndarray] = None,
) -> None:
    """
    Save image to file, auto-detecting format from extension.

    Parameters
    ----------
    data : ArrayLike
        Image data (numpy array or torch tensor)
    path : str
        Output path
    spacing : Tuple[float, ...], optional
        Physical spacing (dx, dy, dz) or (dx, dy)
    origin : Tuple[float, ...], optional
        Physical origin
    direction : np.ndarray, optional
        Direction matrix
    """
    path_lower = path.lower()
    if path_lower.endswith((".nii", ".nii.gz")):
        save_nifti(data, path, spacing=spacing, origin=origin, direction=direction)
    elif HAS_SITK and path_lower.endswith((".mha", ".mhd", ".nrrd")):
        save_sitk(data, path, spacing=spacing, origin=origin, direction=direction)
    else:
        # Fallback to numpy
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        np.save(path, data)


def load_nifti(
    path: str,
    as_torch: bool = True,
    device: Optional[Device] = None,
    dtype: Optional[torch.dtype] = torch.float32,
) -> Union[Tensor, np.ndarray]:
    """
    Load NIfTI file and extract image data with spacing information.

    Parameters
    ----------
    path : str
        Path to .nii or .nii.gz file
    as_torch : bool
        If True, return PyTorch tensor; else numpy array
    device : Device, optional
        Device for tensor (if as_torch=True)
    dtype : torch.dtype, optional
        Data type for output tensor (default: float32)

    Returns
    -------
    Tensor or np.ndarray
        Image data
    """
    if not HAS_NIBABEL:
        raise ImportError("nibabel is required for NIfTI support. Install with: pip install nibabel")

    nii = nib.load(path)
    data = nii.get_fdata()

    if as_torch:
        tensor = as_tensor(data, device=device)
        if dtype is not None:
            tensor = tensor.to(dtype)
        return tensor
    return data.astype(np.float32) if dtype == torch.float32 else data


def save_nifti(
    data: ArrayLike,
    path: str,
    spacing: Optional[Tuple[float, ...]] = None,
    origin: Optional[Tuple[float, ...]] = None,
    direction: Optional[np.ndarray] = None,
    affine: Optional[np.ndarray] = None,
) -> None:
    """
    Save data as NIfTI file with proper spacing handling.

    Parameters
    ----------
    data : ArrayLike
        Image data
    path : str
        Output path
    spacing : Tuple[float, ...], optional
        Physical spacing (dx, dy, dz) or (dx, dy). Defaults to (1.0, 1.0, 1.0) for 3D.
    origin : Tuple[float, ...], optional
        Physical origin. Defaults to (0.0, 0.0, 0.0).
    direction : np.ndarray, optional
        Direction matrix (3x3 or 2x2)
    affine : np.ndarray, optional
        Full affine matrix (4x4). If provided, overrides spacing/origin/direction.
    """
    if not HAS_NIBABEL:
        raise ImportError("nibabel is required for NIfTI support. Install with: pip install nibabel")

    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    # Determine spatial dimensions
    ndim = data.ndim
    if ndim > 3:
        # Assume (B, C, Z, Y, X) or similar - take first element
        data = data.reshape(-1, *data.shape[-3:])[0]

    spatial_dims = min(ndim, 3)

    # Build affine matrix
    if affine is None:
        affine = np.eye(4)
        if spacing is not None:
            for i, s in enumerate(spacing):
                if i < 3:
                    affine[i, i] = s
        else:
            # Default spacing
            for i in range(spatial_dims):
                affine[i, i] = 1.0

        if origin is not None:
            for i, o in enumerate(origin):
                if i < 3:
                    affine[i, 3] = o

        if direction is not None:
            for i in range(min(direction.shape[0], 3)):
                for j in range(min(direction.shape[1], 3)):
                    affine[i, j] = direction[i, j]

    # Create NIfTI image
    nii = nib.Nifti1Image(data, affine)
    nib.save(nii, path)


def get_nifti_spacing(path: str) -> Tuple[float, ...]:
    """
    Extract spacing from NIfTI file header.

    Parameters
    ----------
    path : str
        Path to NIfTI file

    Returns
    -------
    Tuple[float, ...]
        Spacing (dx, dy, dz) or (dx, dy)
    """
    if not HAS_NIBABEL:
        raise ImportError("nibabel is required for NIfTI support. Install with: pip install nibabel")

    nii = nib.load(path)
    affine = nii.affine
    header = nii.header

    # Extract spacing from affine diagonal or header
    spacing = header.get_zooms()[:3]  # Get first 3 zooms

    return tuple(float(s) for s in spacing)


def load_sitk(
    path: str,
    as_torch: bool = True,
    device: Optional[Device] = None,
) -> Union[Tensor, np.ndarray]:
    """
    Load image using SimpleITK.

    Parameters
    ----------
    path : str
        Path to image file
    as_torch : bool
        If True, return PyTorch tensor; else numpy array
    device : Device, optional
        Device for tensor (if as_torch=True)

    Returns
    -------
    Tensor or np.ndarray
        Image data
    """
    if not HAS_SITK:
        raise ImportError("SimpleITK is required. Install with: pip install SimpleITK")

    sitk_image = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(sitk_image)

    if as_torch:
        return as_tensor(data, device=device)
    return data


def save_sitk(
    data: ArrayLike,
    path: str,
    spacing: Optional[Tuple[float, ...]] = None,
    origin: Optional[Tuple[float, ...]] = None,
    direction: Optional[np.ndarray] = None,
) -> None:
    """
    Save image using SimpleITK with proper spacing.

    Parameters
    ----------
    data : ArrayLike
        Image data
    path : str
        Output path
    spacing : Tuple[float, ...], optional
        Physical spacing
    origin : Tuple[float, ...], optional
        Physical origin
    direction : np.ndarray, optional
        Direction matrix
    """
    if not HAS_SITK:
        raise ImportError("SimpleITK is required. Install with: pip install SimpleITK")

    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    sitk_image = sitk.GetImageFromArray(data)

    if spacing is not None:
        sitk_image.SetSpacing(spacing)
    if origin is not None:
        sitk_image.SetOrigin(origin)
    if direction is not None:
        sitk_image.SetDirection(direction.flatten().tolist())

    sitk.WriteImage(sitk_image, path)


def get_sitk_spacing(path: str) -> Tuple[float, ...]:
    """
    Extract spacing from SimpleITK image.

    Parameters
    ----------
    path : str
        Path to image file

    Returns
    -------
    Tuple[float, ...]
        Spacing
    """
    if not HAS_SITK:
        raise ImportError("SimpleITK is required. Install with: pip install SimpleITK")

    sitk_image = sitk.ReadImage(path)
    return tuple(sitk_image.GetSpacing())


def get_spacing_from_header(path: str) -> Tuple[float, ...]:
    """
    Extract spacing from image file header, auto-detecting format.

    This is a unified function that automatically detects the file format
    and extracts spacing information from the appropriate header.

    Parameters
    ----------
    path : str
        Path to image file. Supported formats:
        - NIfTI (.nii, .nii.gz)
        - SimpleITK-readable formats (.mha, .mhd, .nrrd)
        - DICOM (.dcm)

    Returns
    -------
    Tuple[float, ...]
        Physical spacing (dx, dy, dz) or (dx, dy) depending on dimensionality.
        For 3D images, returns (dz, dy, dx) in the order matching array indexing.

    Raises
    ------
    ValueError
        If the file format is not supported
    ImportError
        If required library is not installed

    Example
    -------
    >>> spacing = get_spacing_from_header("brain.nii.gz")
    >>> print(spacing)  # e.g., (1.0, 1.0, 3.0) for 1mm in-plane, 3mm slice thickness
    """
    path_lower = path.lower()

    # NIfTI files
    if path_lower.endswith((".nii", ".nii.gz")):
        return get_nifti_spacing(path)

    # SimpleITK-readable formats
    if HAS_SITK and path_lower.endswith((".mha", ".mhd", ".nrrd")):
        return get_sitk_spacing(path)

    # DICOM files
    if path_lower.endswith(".dcm") or path_lower.endswith(".dicom"):
        return get_dicom_spacing(path)

    # Try SimpleITK as fallback for other formats
    if HAS_SITK:
        try:
            return get_sitk_spacing(path)
        except Exception:
            pass

    raise ValueError(
        f"Unsupported file format or unable to read spacing from: {path}. "
        f"Supported formats: .nii, .nii.gz, .mha, .mhd, .nrrd, .dcm"
    )


def get_dicom_spacing(path: str) -> Tuple[float, ...]:
    """
    Extract spacing from DICOM file.

    Parameters
    ----------
    path : str
        Path to DICOM file

    Returns
    -------
    Tuple[float, ...]
        Spacing (slice_thickness, pixel_spacing_y, pixel_spacing_x)
        or (pixel_spacing_y, pixel_spacing_x) if no slice thickness
    """
    if not HAS_PYDICOM:
        raise ImportError("pydicom is required for DICOM support. Install with: pip install pydicom")

    ds = pydicom.dcmread(path, stop_before_pixels=True)

    # Get pixel spacing (row spacing, column spacing)
    pixel_spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
    if hasattr(pixel_spacing, "__iter__"):
        pixel_spacing = [float(x) for x in pixel_spacing]
    else:
        pixel_spacing = [float(pixel_spacing), float(pixel_spacing)]

    # Get slice thickness or spacing between slices
    slice_thickness = getattr(ds, "SliceThickness", None)
    spacing_between_slices = getattr(ds, "SpacingBetweenSlices", None)

    z_spacing = slice_thickness or spacing_between_slices

    if z_spacing is not None:
        return (float(z_spacing), pixel_spacing[0], pixel_spacing[1])
    else:
        return (pixel_spacing[0], pixel_spacing[1])


def load_dicom(
    path: str,
    strip_metadata: bool = True,
    as_torch: bool = True,
    device: Optional[Device] = None,
) -> Union[Tensor, np.ndarray, Tuple[Union[Tensor, np.ndarray], Dict]]:
    """
    Load DICOM file, optionally stripping PHI metadata.

    Parameters
    ----------
    path : str
        Path to DICOM file
    strip_metadata : bool
        If True, remove PHI-containing tags before returning metadata
    as_torch : bool
        If True, return PyTorch tensor; else numpy array
    device : Device, optional
        Device for tensor (if as_torch=True)

    Returns
    -------
    Tensor or np.ndarray
        Image pixel data
    Dict, optional
        Metadata (if strip_metadata=False or if returning tuple)
    """
    if not HAS_PYDICOM:
        raise ImportError("pydicom is required for DICOM support. Install with: pip install pydicom")

    ds = pydicom.dcmread(path)

    # Extract pixel array
    pixel_array = ds.pixel_array.astype(np.float32)

    # Apply rescale slope/intercept if present
    if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
        pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept

    # Extract metadata
    metadata = {}
    if not strip_metadata:
        # Include all non-binary attributes
        for elem in ds:
            if elem.VR != "OB" and elem.VR != "OW" and elem.VR != "OF":  # Skip binary
                try:
                    metadata[elem.keyword] = elem.value
                except Exception:
                    pass
    else:
        # Only include safe metadata (spacing, etc.)
        safe_tags = ["SliceThickness", "PixelSpacing", "SpacingBetweenSlices"]
        for tag in safe_tags:
            if hasattr(ds, tag):
                metadata[tag] = getattr(ds, tag)

    if as_torch:
        pixel_array = as_tensor(pixel_array, device=device)

    if strip_metadata:
        return pixel_array
    else:
        return pixel_array, metadata

