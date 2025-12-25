"""Qualitative overlay visualization for segmentation."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    Figure = None

from medeval.core.typing import ArrayLike, as_tensor


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


# Default colormap for segmentation overlays
SEGMENTATION_COLORS = [
    (0, 0, 0, 0),        # Background (transparent)
    (1, 0, 0, 0.5),      # Red
    (0, 1, 0, 0.5),      # Green
    (0, 0, 1, 0.5),      # Blue
    (1, 1, 0, 0.5),      # Yellow
    (1, 0, 1, 0.5),      # Magenta
    (0, 1, 1, 0.5),      # Cyan
    (1, 0.5, 0, 0.5),    # Orange
    (0.5, 0, 1, 0.5),    # Purple
]


def plot_segmentation_overlay(
    image: ArrayLike,
    mask: Optional[ArrayLike] = None,
    prediction: Optional[ArrayLike] = None,
    alpha: float = 0.5,
    slice_idx: Optional[int] = None,
    axis: int = 0,
    ax: Optional["plt.Axes"] = None,
    cmap: str = "gray",
    mask_colors: Optional[List[Tuple[float, ...]]] = None,
    show_contours: bool = False,
    contour_color: str = "red",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> Tuple[Figure, "plt.Axes"]:
    """
    Plot image with segmentation overlay.

    Parameters
    ----------
    image : ArrayLike
        Background image (2D or 3D)
    mask : ArrayLike, optional
        Ground truth mask
    prediction : ArrayLike, optional
        Predicted mask
    alpha : float
        Transparency of overlay
    slice_idx : int, optional
        Slice index for 3D images (middle slice if None)
    axis : int
        Axis along which to take slice for 3D images
    ax : plt.Axes, optional
        Axes to plot on
    cmap : str
        Colormap for background image
    mask_colors : List[Tuple], optional
        Colors for each class in mask
    show_contours : bool
        If True, show contours instead of filled overlay
    contour_color : str
        Color for contours
    title : str, optional
        Plot title
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    Tuple[Figure, plt.Axes]
        Figure and axes objects
    """
    _check_matplotlib()

    image = np.asarray(as_tensor(image).cpu().numpy() if hasattr(image, 'cpu') else image)
    
    # Handle 3D images
    if image.ndim == 3:
        if slice_idx is None:
            slice_idx = image.shape[axis] // 2
        image = np.take(image, slice_idx, axis=axis)
    
    if mask is not None:
        mask = np.asarray(as_tensor(mask).cpu().numpy() if hasattr(mask, 'cpu') else mask)
        if mask.ndim == 3:
            mask = np.take(mask, slice_idx, axis=axis)
    
    if prediction is not None:
        prediction = np.asarray(as_tensor(prediction).cpu().numpy() if hasattr(prediction, 'cpu') else prediction)
        if prediction.ndim == 3:
            prediction = np.take(prediction, slice_idx, axis=axis)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot background image
    ax.imshow(image, cmap=cmap, aspect="equal")

    # Setup colors
    if mask_colors is None:
        mask_colors = SEGMENTATION_COLORS

    # Plot mask overlay
    if mask is not None:
        if show_contours:
            _plot_contours(ax, mask, contour_color, linestyle="-", linewidth=2)
        else:
            mask_rgba = _create_rgba_overlay(mask, mask_colors, alpha)
            ax.imshow(mask_rgba, aspect="equal")

    # Plot prediction overlay
    if prediction is not None:
        if show_contours:
            _plot_contours(ax, prediction, "blue", linestyle="--", linewidth=2)
        else:
            # Use different alpha or color for prediction
            pred_colors = [(c[0], c[1], c[2], alpha * 0.7) for c in mask_colors]
            pred_rgba = _create_rgba_overlay(prediction, pred_colors, alpha)
            ax.imshow(pred_rgba, aspect="equal")

    ax.axis("off")
    ax.set_title(title or "Segmentation Overlay", fontsize=14)

    return fig, ax


def _create_rgba_overlay(
    mask: np.ndarray,
    colors: List[Tuple[float, ...]],
    alpha: float,
) -> np.ndarray:
    """Create RGBA overlay from integer mask."""
    h, w = mask.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.float32)

    unique_labels = np.unique(mask)
    for label in unique_labels:
        if label == 0:  # Skip background
            continue
        color_idx = int(label) % len(colors)
        color = colors[color_idx]
        mask_region = mask == label
        for c in range(4):
            rgba[mask_region, c] = color[c] if c < 3 else alpha

    return rgba


def _plot_contours(
    ax: "plt.Axes",
    mask: np.ndarray,
    color: str,
    linestyle: str = "-",
    linewidth: float = 2,
) -> None:
    """Plot contours of a mask."""
    from scipy import ndimage
    
    unique_labels = np.unique(mask)
    for label in unique_labels:
        if label == 0:
            continue
        binary_mask = (mask == label).astype(np.float32)
        # Find contours using gradient
        edges = ndimage.sobel(binary_mask)
        edges = np.abs(edges) > 0
        ax.contour(binary_mask, levels=[0.5], colors=[color], linestyles=[linestyle], linewidths=[linewidth])


def plot_prediction_comparison(
    image: ArrayLike,
    ground_truth: ArrayLike,
    prediction: ArrayLike,
    slice_idx: Optional[int] = None,
    axis: int = 0,
    figsize: Tuple[int, int] = (16, 5),
    cmap: str = "gray",
    title: Optional[str] = None,
) -> Tuple[Figure, List["plt.Axes"]]:
    """
    Plot side-by-side comparison of ground truth and prediction.

    Parameters
    ----------
    image : ArrayLike
        Background image
    ground_truth : ArrayLike
        Ground truth mask
    prediction : ArrayLike
        Predicted mask
    slice_idx : int, optional
        Slice index for 3D
    axis : int
        Axis for 3D slicing
    figsize : Tuple[int, int]
        Figure size
    cmap : str
        Colormap for image
    title : str, optional
        Overall title

    Returns
    -------
    Tuple[Figure, List[plt.Axes]]
        Figure and list of axes
    """
    _check_matplotlib()

    image = np.asarray(as_tensor(image).cpu().numpy() if hasattr(image, 'cpu') else image)
    ground_truth = np.asarray(as_tensor(ground_truth).cpu().numpy() if hasattr(ground_truth, 'cpu') else ground_truth)
    prediction = np.asarray(as_tensor(prediction).cpu().numpy() if hasattr(prediction, 'cpu') else prediction)

    # Handle 3D
    if image.ndim == 3:
        if slice_idx is None:
            slice_idx = image.shape[axis] // 2
        image = np.take(image, slice_idx, axis=axis)
        ground_truth = np.take(ground_truth, slice_idx, axis=axis)
        prediction = np.take(prediction, slice_idx, axis=axis)

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Original image
    axes[0].imshow(image, cmap=cmap)
    axes[0].set_title("Image", fontsize=12)
    axes[0].axis("off")

    # Ground truth
    axes[1].imshow(image, cmap=cmap)
    gt_rgba = _create_rgba_overlay(ground_truth, SEGMENTATION_COLORS, 0.5)
    axes[1].imshow(gt_rgba)
    axes[1].set_title("Ground Truth", fontsize=12)
    axes[1].axis("off")

    # Prediction
    axes[2].imshow(image, cmap=cmap)
    pred_rgba = _create_rgba_overlay(prediction, SEGMENTATION_COLORS, 0.5)
    axes[2].imshow(pred_rgba)
    axes[2].set_title("Prediction", fontsize=12)
    axes[2].axis("off")

    # Difference (TP, FP, FN)
    axes[3].imshow(image, cmap=cmap)
    diff_overlay = _create_diff_overlay(ground_truth, prediction)
    axes[3].imshow(diff_overlay)
    axes[3].set_title("Difference (TP=green, FP=red, FN=blue)", fontsize=10)
    axes[3].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()

    return fig, axes


def _create_diff_overlay(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    alpha: float = 0.6,
) -> np.ndarray:
    """Create difference overlay showing TP, FP, FN."""
    h, w = ground_truth.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.float32)

    gt_binary = ground_truth > 0
    pred_binary = prediction > 0

    tp = gt_binary & pred_binary  # True Positive - Green
    fp = ~gt_binary & pred_binary  # False Positive - Red
    fn = gt_binary & ~pred_binary  # False Negative - Blue

    # Green for TP
    rgba[tp, 0] = 0.0
    rgba[tp, 1] = 1.0
    rgba[tp, 2] = 0.0
    rgba[tp, 3] = alpha

    # Red for FP
    rgba[fp, 0] = 1.0
    rgba[fp, 1] = 0.0
    rgba[fp, 2] = 0.0
    rgba[fp, 3] = alpha

    # Blue for FN
    rgba[fn, 0] = 0.0
    rgba[fn, 1] = 0.0
    rgba[fn, 2] = 1.0
    rgba[fn, 3] = alpha

    return rgba


def plot_3d_slices(
    volume: ArrayLike,
    mask: Optional[ArrayLike] = None,
    n_slices: int = 5,
    axis: int = 0,
    figsize: Tuple[int, int] = (15, 3),
    cmap: str = "gray",
    title: Optional[str] = None,
) -> Tuple[Figure, List["plt.Axes"]]:
    """
    Plot multiple slices from a 3D volume.

    Parameters
    ----------
    volume : ArrayLike
        3D volume
    mask : ArrayLike, optional
        3D mask to overlay
    n_slices : int
        Number of slices to show
    axis : int
        Axis along which to slice
    figsize : Tuple[int, int]
        Figure size
    cmap : str
        Colormap
    title : str, optional
        Plot title

    Returns
    -------
    Tuple[Figure, List[plt.Axes]]
        Figure and axes
    """
    _check_matplotlib()

    volume = np.asarray(as_tensor(volume).cpu().numpy() if hasattr(volume, 'cpu') else volume)
    if mask is not None:
        mask = np.asarray(as_tensor(mask).cpu().numpy() if hasattr(mask, 'cpu') else mask)

    n_total = volume.shape[axis]
    slice_indices = np.linspace(0, n_total - 1, n_slices, dtype=int)

    fig, axes = plt.subplots(1, n_slices, figsize=figsize)
    if n_slices == 1:
        axes = [axes]

    for i, idx in enumerate(slice_indices):
        slice_img = np.take(volume, idx, axis=axis)
        axes[i].imshow(slice_img, cmap=cmap)
        
        if mask is not None:
            slice_mask = np.take(mask, idx, axis=axis)
            mask_rgba = _create_rgba_overlay(slice_mask, SEGMENTATION_COLORS, 0.5)
            axes[i].imshow(mask_rgba)
        
        axes[i].set_title(f"Slice {idx}", fontsize=10)
        axes[i].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()

    return fig, axes

