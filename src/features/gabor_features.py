import numpy as np
import cv2
from typing import List


def create_gabor_kernels(num_scales: int = 5, num_orientations: int = 4) -> List:
    """
    Create a bank of Gabor filters with different scales and orientations.

    Args:
        num_scales: Number of scales (frequencies)
        num_orientations: Number of orientations

    Returns:
        List of Gabor kernels
    """
    kernels = []
    ksize = 11  # Kernel size

    # Different scales (sigma values)
    scales = [3, 5, 7, 9, 11][:num_scales]

    # Different orientations (in radians)
    orientations = [i * np.pi / num_orientations for i in range(num_orientations)]

    for sigma in scales:
        for theta in orientations:
            # Create Gabor kernel
            kernel = cv2.getGaborKernel(
                ksize=(ksize, ksize),
                sigma=sigma,
                theta=theta,
                lambd=10.0,  # Wavelength
                gamma=0.5,   # Aspect ratio
                psi=0,       # Phase offset
                ktype=cv2.CV_32F
            )
            kernels.append(kernel)

    return kernels


def extract_gabor_features(cell_gray: np.ndarray, num_scales: int = 5,
                          num_orientations: int = 4) -> np.ndarray:
    """
    Extract 20 Gabor filter features from a grayscale cell.

    Gabor filters detect edges and textures at multiple scales and orientations.
    Useful for detecting:
        - Circular ball pattern (isotropic response)
        - Elongated bat shape (anisotropic response)
        - Vertical stumps (strong 90° response)

    Features:
        - 20 features from 5 scales × 4 orientations = 20 filters
        - Each feature is the mean absolute response of a Gabor filter

    Args:
        cell_gray: Grayscale image cell of shape (H, W)
        num_scales: Number of scales (default: 5)
        num_orientations: Number of orientations (default: 4)

    Returns:
        Array of 20 features
    """
    # Create Gabor kernel bank
    kernels = create_gabor_kernels(num_scales, num_orientations)

    # Extract features
    features = []
    for kernel in kernels:
        # Apply Gabor filter
        filtered = cv2.filter2D(cell_gray, cv2.CV_32F, kernel)

        # Compute mean absolute response (energy)
        response = np.mean(np.abs(filtered))
        features.append(response)

    # Ensure exactly 20 features
    features = np.array(features, dtype=np.float32)

    if len(features) > 20:
        features = features[:20]
    elif len(features) < 20:
        # Pad with zeros if needed
        padded = np.zeros(20, dtype=np.float32)
        padded[:len(features)] = features
        features = padded

    return features


def get_gabor_feature_names() -> List[str]:
    """
    Get the names of Gabor features.

    Returns:
        List of 20 feature names
    """
    names = []
    scales = [3, 5, 7, 9, 11]
    orientations = [0, 45, 90, 135]

    for scale in scales:
        for orientation in orientations:
            names.append(f'gabor_s{scale}_o{orientation}')

    return names[:20]
