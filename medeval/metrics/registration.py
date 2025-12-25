"""Registration metrics for medical imaging evaluation.

This module provides metrics for evaluating image registration quality,
including landmark-based errors, image similarity, and deformation field quality.
"""

from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, sobel

from medeval.core.typing import ArrayLike, Device, Tensor, as_tensor

try:
    from scipy.stats import entropy
    HAS_ENTROPY = True
except ImportError:
    HAS_ENTROPY = False


def target_registration_error(
    pred_landmarks: ArrayLike,
    target_landmarks: ArrayLike,
    spacing: Optional[Tuple[float, ...]] = None,
    per_case: bool = False,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compute Target Registration Error (TRE).

    TRE is the Euclidean distance between corresponding landmarks after registration.

    Parameters
    ----------
    pred_landmarks : ArrayLike
        Predicted landmark positions, shape (N, D) where D is spatial dimension
    target_landmarks : ArrayLike
        Ground truth landmark positions, shape (N, D)
    spacing : Tuple[float, ...], optional
        Physical spacing for distance computation
    per_case : bool
        If True, return per-case TRE (assuming first dimension is cases)

    Returns
    -------
    Dict[str, Union[float, np.ndarray]]
        Dictionary with TRE statistics: mean, median, 95th percentile, and per-landmark/per-case
    """
    pred_landmarks = as_tensor(pred_landmarks).cpu().numpy()
    target_landmarks = as_tensor(target_landmarks).cpu().numpy()

    if spacing is None:
        spacing = np.ones(pred_landmarks.shape[-1])

    # Compute Euclidean distances
    if per_case:
        # Shape: (N_cases, N_landmarks, D)
        diffs = pred_landmarks - target_landmarks
        # Apply spacing
        for i, s in enumerate(spacing):
            diffs[..., i] *= s
        distances = np.linalg.norm(diffs, axis=-1)  # (N_cases, N_landmarks)
        per_landmark = distances.mean(axis=0)  # Average over cases
        per_case_tre = distances.mean(axis=1)  # Average over landmarks
    else:
        # Shape: (N_landmarks, D)
        diffs = pred_landmarks - target_landmarks
        # Apply spacing
        for i, s in enumerate(spacing):
            diffs[:, i] *= s
        distances = np.linalg.norm(diffs, axis=1)  # (N_landmarks,)
        per_landmark = distances
        per_case_tre = None

    results = {
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "95th_percentile": float(np.percentile(distances, 95)),
        "per_landmark": per_landmark,
    }

    if per_case_tre is not None:
        results["per_case"] = per_case_tre

    return results


def normalized_mutual_information(
    image1: ArrayLike,
    image2: ArrayLike,
    bins: int = 256,
) -> float:
    """
    Compute Normalized Mutual Information (NMI) between two images.

    Uses the bounded formula: NMI = 2 * I(X,Y) / (H(X) + H(Y))
    where I(X,Y) = H(X) + H(Y) - H(X,Y) is the mutual information.
    Result is bounded to [0, 1].

    Parameters
    ----------
    image1 : ArrayLike
        First image
    image2 : ArrayLike
        Second image
    bins : int
        Number of bins for histogram

    Returns
    -------
    float
        NMI value in [0, 1]
    """
    if not HAS_ENTROPY:
        raise ImportError("scipy.stats.entropy is required for NMI computation")

    image1 = as_tensor(image1).cpu().numpy()
    image2 = as_tensor(image2).cpu().numpy()

    # Flatten images
    image1_flat = image1.flatten()
    image2_flat = image2.flatten()

    # Normalize to [0, 1]
    image1_norm = (image1_flat - image1_flat.min()) / (image1_flat.max() - image1_flat.min() + 1e-10)
    image2_norm = (image2_flat - image2_flat.min()) / (image2_flat.max() - image2_flat.min() + 1e-10)

    # Compute joint histogram
    hist_2d, x_edges, y_edges = np.histogram2d(
        image1_norm, image2_norm, bins=bins, range=[[0, 1], [0, 1]]
    )
    hist_2d = hist_2d / hist_2d.sum()

    # Compute marginal histograms
    hist_1d_x = hist_2d.sum(axis=1)
    hist_1d_y = hist_2d.sum(axis=0)

    # Compute entropies
    h_x = entropy(hist_1d_x[hist_1d_x > 0])
    h_y = entropy(hist_1d_y[hist_1d_y > 0])
    h_xy = entropy(hist_2d[hist_2d > 0])

    # Compute mutual information: I(X,Y) = H(X) + H(Y) - H(X,Y)
    mi = h_x + h_y - h_xy

    # Compute bounded NMI: 2 * MI / (H(X) + H(Y))
    if (h_x + h_y) > 0:
        nmi = 2.0 * mi / (h_x + h_y)
        # Clamp to [0, 1] to handle numerical issues
        nmi = max(0.0, min(1.0, nmi))
    else:
        nmi = 0.0

    return float(nmi)


def normalized_cross_correlation(
    image1: ArrayLike,
    image2: ArrayLike,
    local: bool = False,
    window_size: int = 9,
) -> float:
    """
    Compute Normalized Cross-Correlation (NCC) between two images.

    NCC = sum((I1 - mean1) * (I2 - mean2)) / sqrt(sum((I1 - mean1)^2) * sum((I2 - mean2)^2))

    Parameters
    ----------
    image1 : ArrayLike
        First image
    image2 : ArrayLike
        Second image
    local : bool
        If True, compute local NCC (average over local windows)
    window_size : int
        Window size for local NCC

    Returns
    -------
    float
        NCC value
    """
    image1 = as_tensor(image1).float()
    image2 = as_tensor(image2).float()

    if local:
        # Local NCC: compute NCC in sliding windows and average
        return _compute_local_ncc(image1, image2, window_size)

    # Global NCC
    # Flatten images
    img1_flat = image1.flatten()
    img2_flat = image2.flatten()

    # Compute means
    mean1 = img1_flat.mean()
    mean2 = img2_flat.mean()

    # Center images
    img1_centered = img1_flat - mean1
    img2_centered = img2_flat - mean2

    # Compute NCC
    numerator = (img1_centered * img2_centered).sum()
    denominator = torch.sqrt((img1_centered ** 2).sum() * (img2_centered ** 2).sum())

    if denominator > 0:
        ncc = numerator / denominator
    else:
        ncc = torch.tensor(0.0)

    return float(ncc.item())


def _compute_local_ncc(
    image1: Tensor,
    image2: Tensor,
    window_size: int = 9,
) -> float:
    """
    Compute local NCC by averaging NCC over sliding windows.

    Parameters
    ----------
    image1 : Tensor
        First image
    image2 : Tensor
        Second image
    window_size : int
        Window size for local computation

    Returns
    -------
    float
        Average local NCC
    """
    # Ensure images have batch and channel dimensions for conv operations
    if image1.dim() == 2:
        image1 = image1.unsqueeze(0).unsqueeze(0)
        image2 = image2.unsqueeze(0).unsqueeze(0)
    elif image1.dim() == 3:
        image1 = image1.unsqueeze(0)
        image2 = image2.unsqueeze(0)

    # Create uniform kernel for local mean computation
    ndim = image1.dim() - 2  # Spatial dimensions
    kernel_size = [window_size] * ndim
    kernel = torch.ones(1, 1, *kernel_size, device=image1.device, dtype=image1.dtype)
    kernel = kernel / kernel.numel()

    # Padding for same output size
    padding = window_size // 2

    if ndim == 2:
        # 2D local NCC
        local_mean1 = F.conv2d(image1, kernel, padding=padding)
        local_mean2 = F.conv2d(image2, kernel, padding=padding)

        # Local variance terms
        img1_centered = image1 - local_mean1
        img2_centered = image2 - local_mean2

        local_cov = F.conv2d(img1_centered * img2_centered, kernel, padding=padding)
        local_var1 = F.conv2d(img1_centered ** 2, kernel, padding=padding)
        local_var2 = F.conv2d(img2_centered ** 2, kernel, padding=padding)

    else:
        # 3D local NCC
        local_mean1 = F.conv3d(image1, kernel, padding=padding)
        local_mean2 = F.conv3d(image2, kernel, padding=padding)

        img1_centered = image1 - local_mean1
        img2_centered = image2 - local_mean2

        local_cov = F.conv3d(img1_centered * img2_centered, kernel, padding=padding)
        local_var1 = F.conv3d(img1_centered ** 2, kernel, padding=padding)
        local_var2 = F.conv3d(img2_centered ** 2, kernel, padding=padding)

    # Compute local NCC
    epsilon = 1e-10
    denominator = torch.sqrt(local_var1 * local_var2 + epsilon)
    local_ncc = local_cov / denominator

    # Average over all locations
    return float(local_ncc.mean().item())


def mind_ssd(
    image1: ArrayLike,
    image2: ArrayLike,
    radius: int = 2,
    sigma: float = 0.8,
) -> float:
    """
    Compute MIND-SSD (Modality Independent Neighbourhood Descriptor - Sum of Squared Differences).

    This is an optional advanced similarity metric.

    Parameters
    ----------
    image1 : ArrayLike
        First image
    image2 : ArrayLike
        Second image
    radius : int
        Neighbourhood radius
    sigma : float
        Gaussian smoothing parameter

    Returns
    -------
    float
        MIND-SSD value (lower is better)
    """
    image1 = as_tensor(image1).float()
    image2 = as_tensor(image2).float()

    # Convert to numpy for processing
    img1_np = image1.cpu().numpy()
    img2_np = image2.cpu().numpy()

    # Apply Gaussian smoothing
    img1_smooth = gaussian_filter(img1_np, sigma=sigma)
    img2_smooth = gaussian_filter(img2_np, sigma=sigma)

    # Compute MIND descriptors (simplified version)
    # Full MIND implementation is complex - this is a simplified approximation
    # Compute local mean and variance in neighbourhoods
    mind1 = _compute_mind_descriptor(img1_smooth, radius)
    mind2 = _compute_mind_descriptor(img2_smooth, radius)

    # Compute SSD
    ssd = np.mean((mind1 - mind2) ** 2)

    return float(ssd)


def _compute_mind_descriptor(image: np.ndarray, radius: int) -> np.ndarray:
    """Compute simplified MIND descriptor."""
    # This is a simplified version - full MIND uses more sophisticated neighbourhood descriptors
    # For now, we'll use local statistics
    from scipy.ndimage import uniform_filter

    local_mean = uniform_filter(image.astype(float), size=2 * radius + 1)
    local_var = uniform_filter((image.astype(float) - local_mean) ** 2, size=2 * radius + 1)

    # Normalize
    epsilon = 1e-10
    descriptor = (image.astype(float) - local_mean) / np.sqrt(local_var + epsilon)

    return descriptor


def jacobian_determinant(
    deformation_field: ArrayLike,
    spacing: Optional[Tuple[float, ...]] = None,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compute Jacobian determinant of deformation field.

    The Jacobian determinant indicates local volume change and folding.
    - |J| < 0: indicates folding (physically impossible deformation)
    - |J| = 1: volume-preserving
    - |J| > 1: expansion
    - 0 < |J| < 1: compression

    Parameters
    ----------
    deformation_field : ArrayLike
        Deformation field, shape (D, H, W) for 2D or (D, Z, H, W) for 3D
        where D is the number of spatial dimensions (2 for 2D, 3 for 3D)
    spacing : Tuple[float, ...], optional
        Physical spacing (dy, dx) for 2D or (dz, dy, dx) for 3D

    Returns
    -------
    Dict[str, Union[float, np.ndarray]]
        Dictionary with Jacobian statistics and folding percentage
    """
    deformation_field = as_tensor(deformation_field).float()

    # Determine spatial dimensions based on shape
    if deformation_field.dim() == 3:
        # 2D: (2, H, W)
        spatial_dims = 2
        n_components = deformation_field.shape[0]
        if n_components != 2:
            raise ValueError(f"For 2D, expected 2 components, got {n_components}")
        h, w = deformation_field.shape[1], deformation_field.shape[2]
    elif deformation_field.dim() == 4:
        # 3D: (3, Z, H, W)
        spatial_dims = 3
        n_components = deformation_field.shape[0]
        if n_components != 3:
            raise ValueError(f"For 3D, expected 3 components, got {n_components}")
        z, h, w = deformation_field.shape[1], deformation_field.shape[2], deformation_field.shape[3]
    else:
        raise ValueError(f"Unsupported deformation field shape: {deformation_field.shape}")

    if spacing is None:
        spacing = (1.0,) * spatial_dims

    def_field_np = deformation_field.cpu().numpy()

    if spatial_dims == 2:
        jacobians = _compute_jacobian_2d(def_field_np, spacing)
    else:
        jacobians = _compute_jacobian_3d(def_field_np, spacing)

    # Compute statistics
    jacobians_flat = jacobians.flatten()
    folding_percentage = float(np.sum(jacobians_flat < 0) / len(jacobians_flat) * 100)

    return {
        "mean": float(np.mean(jacobians_flat)),
        "median": float(np.median(jacobians_flat)),
        "std": float(np.std(jacobians_flat)),
        "min": float(np.min(jacobians_flat)),
        "max": float(np.max(jacobians_flat)),
        "folding_percentage": folding_percentage,
        "jacobians": jacobians,
    }


def _compute_jacobian_2d(
    def_field: np.ndarray,
    spacing: Tuple[float, ...],
) -> np.ndarray:
    """Compute 2D Jacobian determinant using central differences."""
    h, w = def_field.shape[1], def_field.shape[2]
    jacobians = np.ones((h, w))

    # Use gradient for numerical differentiation
    # def_field[0] = displacement in y direction
    # def_field[1] = displacement in x direction

    # Compute gradients using numpy gradient (central differences)
    # dy_dy: gradient of y-displacement w.r.t. y
    # dy_dx: gradient of y-displacement w.r.t. x
    # dx_dy: gradient of x-displacement w.r.t. y
    # dx_dx: gradient of x-displacement w.r.t. x

    dy_dy, dy_dx = np.gradient(def_field[0], spacing[0], spacing[1])
    dx_dy, dx_dx = np.gradient(def_field[1], spacing[0], spacing[1])

    # Jacobian matrix at each point:
    # J = [[1 + dy_dy, dy_dx],
    #      [dx_dy, 1 + dx_dx]]
    # det(J) = (1 + dy_dy)(1 + dx_dx) - dy_dx * dx_dy

    jacobians = (1 + dy_dy) * (1 + dx_dx) - dy_dx * dx_dy

    return jacobians


def _compute_jacobian_3d(
    def_field: np.ndarray,
    spacing: Tuple[float, ...],
) -> np.ndarray:
    """Compute 3D Jacobian determinant using central differences."""
    z, h, w = def_field.shape[1], def_field.shape[2], def_field.shape[3]

    # def_field[0] = displacement in z direction
    # def_field[1] = displacement in y direction
    # def_field[2] = displacement in x direction

    # Compute gradients for each component
    dz_dz, dz_dy, dz_dx = np.gradient(def_field[0], spacing[0], spacing[1], spacing[2])
    dy_dz, dy_dy, dy_dx = np.gradient(def_field[1], spacing[0], spacing[1], spacing[2])
    dx_dz, dx_dy, dx_dx = np.gradient(def_field[2], spacing[0], spacing[1], spacing[2])

    # Jacobian matrix at each point:
    # J = [[1 + dz_dz, dz_dy, dz_dx],
    #      [dy_dz, 1 + dy_dy, dy_dx],
    #      [dx_dz, dx_dy, 1 + dx_dx]]

    # Compute determinant using the rule for 3x3 matrices
    j00 = 1 + dz_dz
    j01 = dz_dy
    j02 = dz_dx
    j10 = dy_dz
    j11 = 1 + dy_dy
    j12 = dy_dx
    j20 = dx_dz
    j21 = dx_dy
    j22 = 1 + dx_dx

    # det = j00*(j11*j22 - j12*j21) - j01*(j10*j22 - j12*j20) + j02*(j10*j21 - j11*j20)
    jacobians = (
        j00 * (j11 * j22 - j12 * j21)
        - j01 * (j10 * j22 - j12 * j20)
        + j02 * (j10 * j21 - j11 * j20)
    )

    return jacobians


def bending_energy(
    deformation_field: ArrayLike,
    spacing: Optional[Tuple[float, ...]] = None,
) -> float:
    """
    Compute bending energy of deformation field.

    Bending energy measures the smoothness of the deformation.

    Parameters
    ----------
    deformation_field : ArrayLike
        Deformation field
    spacing : Tuple[float, ...], optional
        Physical spacing

    Returns
    -------
    float
        Bending energy
    """
    deformation_field = as_tensor(deformation_field).float()

    if spacing is None:
        spacing = (1.0,) * (deformation_field.shape[0] if deformation_field.dim() > 1 else 1)

    # Compute second derivatives (Laplacian)
    def_field_np = deformation_field.cpu().numpy()

    # Compute Laplacian for each component
    laplacians = []
    for i in range(deformation_field.shape[0]):
        component = def_field_np[i]
        # Compute Laplacian using Sobel filters
        if component.ndim == 2:
            laplacian = sobel(sobel(component, axis=0), axis=0) + sobel(sobel(component, axis=1), axis=1)
        else:
            # 3D case
            laplacian = (
                sobel(sobel(component, axis=0), axis=0)
                + sobel(sobel(component, axis=1), axis=1)
                + sobel(sobel(component, axis=2), axis=2)
            )
        laplacians.append(laplacian)

    # Bending energy = sum of squared Laplacians
    bending_energy_value = np.sum([np.sum(l ** 2) for l in laplacians])

    return float(bending_energy_value)


def deformation_smoothness(
    deformation_field: ArrayLike,
    spacing: Optional[Tuple[float, ...]] = None,
) -> Dict[str, float]:
    """
    Compute smoothness metrics for deformation field.

    Parameters
    ----------
    deformation_field : ArrayLike
        Deformation field
    spacing : Tuple[float, ...], optional
        Physical spacing

    Returns
    -------
    Dict[str, float]
        Dictionary with smoothness metrics
    """
    deformation_field = as_tensor(deformation_field).float()

    # Compute gradients
    def_field_np = deformation_field.cpu().numpy()

    # Compute gradient magnitude
    gradients = []
    for i in range(deformation_field.shape[0]):
        component = def_field_np[i]
        if component.ndim == 2:
            grad_x = np.gradient(component, axis=1)
            grad_y = np.gradient(component, axis=0)
            grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        else:
            grad_x = np.gradient(component, axis=2)
            grad_y = np.gradient(component, axis=1)
            grad_z = np.gradient(component, axis=0)
            grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2)
        gradients.append(grad_mag)

    # Average gradient magnitude
    avg_grad_mag = np.mean([np.mean(g) for g in gradients])

    # Bending energy
    be = bending_energy(deformation_field, spacing)

    return {
        "average_gradient_magnitude": float(avg_grad_mag),
        "bending_energy": be,
    }


def compute_registration_metrics(
    pred_image: Optional[ArrayLike] = None,
    target_image: Optional[ArrayLike] = None,
    pred_landmarks: Optional[ArrayLike] = None,
    target_landmarks: Optional[ArrayLike] = None,
    deformation_field: Optional[ArrayLike] = None,
    spacing: Optional[Tuple[float, ...]] = None,
    include_image_similarity: bool = True,
    include_deformation_quality: Optional[bool] = None,
) -> Dict[str, Union[float, Dict]]:
    """
    Compute comprehensive registration metrics.

    This function computes various metrics to evaluate image registration quality:
    - Landmark-based: Target Registration Error (TRE)
    - Image similarity: Normalized Mutual Information (NMI), Normalized Cross-Correlation (NCC)
    - Deformation quality: Jacobian determinant, bending energy, smoothness

    Parameters
    ----------
    pred_image : ArrayLike, optional
        Registered/predicted image
    target_image : ArrayLike, optional
        Target/reference image
    pred_landmarks : ArrayLike, optional
        Landmark positions after registration, shape (N, D) where D is spatial dimension
    target_landmarks : ArrayLike, optional
        Ground truth landmark positions, shape (N, D)
    deformation_field : ArrayLike, optional
        Deformation field, shape (D, ...) where D is the number of spatial dimensions
    spacing : Tuple[float, ...], optional
        Physical spacing for distance/TRE computation
    include_image_similarity : bool
        If True, compute image similarity metrics (NMI, NCC) when images are provided
    include_deformation_quality : bool, optional
        If True, compute deformation field quality metrics (Jacobian, bending energy).
        If None (default), automatically enabled when deformation_field is provided.

    Returns
    -------
    Dict[str, Union[float, Dict]]
        Dictionary of registration metrics:
        - tre: dict with mean, median, 95th_percentile, per_landmark (if landmarks provided)
        - nmi: float, Normalized Mutual Information (if images provided)
        - ncc: float, Normalized Cross-Correlation (if images provided)
        - jacobian: dict with mean, median, std, min, max, folding_percentage (if deformation provided)
        - bending_energy: float (if deformation provided)
        - smoothness: dict with average_gradient_magnitude, bending_energy (if deformation provided)

    Example
    -------
    >>> # Landmark-based evaluation
    >>> pred_landmarks = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    >>> gt_landmarks = torch.tensor([[10.5, 20.2, 30.1], [40.8, 50.5, 60.3]])
    >>> results = compute_registration_metrics(
    ...     pred_landmarks=pred_landmarks,
    ...     target_landmarks=gt_landmarks,
    ...     spacing=(1.0, 1.0, 1.0)
    ... )
    >>> print(f"Mean TRE: {results['tre']['mean']:.2f} mm")
    
    >>> # With deformation field analysis
    >>> deformation = torch.randn(3, 16, 32, 32) * 0.1
    >>> results = compute_registration_metrics(
    ...     pred_landmarks=pred_landmarks,
    ...     target_landmarks=gt_landmarks,
    ...     deformation_field=deformation,
    ...     spacing=(2.0, 1.0, 1.0)
    ... )
    >>> print(f"Folding: {results['jacobian']['folding_percentage']:.2f}%")
    """
    results = {}

    # Auto-enable deformation quality if deformation_field is provided
    if include_deformation_quality is None:
        include_deformation_quality = deformation_field is not None

    # Landmark-based metrics
    if pred_landmarks is not None and target_landmarks is not None:
        tre_results = target_registration_error(
            pred_landmarks, target_landmarks, spacing=spacing
        )
        results["tre"] = tre_results

    # Image similarity metrics
    if include_image_similarity and pred_image is not None and target_image is not None:
        results["nmi"] = normalized_mutual_information(pred_image, target_image)
        results["ncc"] = normalized_cross_correlation(pred_image, target_image)

    # Deformation field quality
    if include_deformation_quality and deformation_field is not None:
        # Get spacing for deformation field (use spacing or default to 1.0)
        def_field = as_tensor(deformation_field)
        if def_field.dim() == 4:
            # (B, D, H, W) - remove batch if present, or (D, Z, H, W)
            if def_field.shape[0] in (1, 2, 3):
                # Likely (D, Z, H, W) where D is displacement components
                pass
            else:
                # Likely (B, D, H, W) - take first batch
                def_field = def_field[0]
        elif def_field.dim() == 5:
            # (B, D, Z, H, W) - take first batch
            def_field = def_field[0]
        
        # Compute Jacobian determinant
        jacobian_results = jacobian_determinant(def_field, spacing=spacing)
        # Remove the full jacobian array from results (too large)
        results["jacobian"] = {
            k: v for k, v in jacobian_results.items() if k != "jacobians"
        }
        
        results["bending_energy"] = bending_energy(def_field, spacing=spacing)
        smoothness_results = deformation_smoothness(def_field, spacing=spacing)
        results["smoothness"] = smoothness_results

    return results

