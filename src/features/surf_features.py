import cv2
import numpy as np
from typing import List


def extract_surf_features(cell_gray: np.ndarray, max_features=128) -> np.ndarray:
    """
    Extract a fixed-length SURF descriptor vector from a grayscale image cell.

    WHAT THIS FUNCTION ACTUALLY DOES:
    ----------------------------------------------------------------------
    - Detects SURF keypoints inside the grayscale cell
    - Computes SURF descriptors (each descriptor = 64 dimensions)
    - Keeps the first `max_features` descriptors
    - Flattens them into a 1D vector
    - Pads with zeros if fewer than `max_features` descriptors are detected

    FINAL FEATURE LENGTH:
        max_features × 64

    Example:
        max_features = 128 → 128 × 64 = **8192 features**

    WHY ZERO-PADDING?
    ----------------------------------------------------------------------
    SURF may detect fewer keypoints in low-texture cells.  
    Padding ensures that **every cell outputs a fixed-length feature vector**,  
    which is required for machine learning models and pipelines.

    IMPORTANT:
        SURF is a patented algorithm.  
        It requires **opencv-contrib-python** AND enabling **nonfree** during build.
        If unavailable, cv2.SURF_create() will raise an error.

    INPUT:
        cell_gray : grayscale cell (H × W)
        max_features : maximum SURF keypoints/descriptors to include

    OUTPUT:
        np.ndarray of shape (max_features × 64,), dtype=float32
    """

    # Create SURF detector (requires nonfree build of OpenCV)
    surf = cv2.xfeatures2d.SURF_create() if hasattr(cv2, "xfeatures2d") else cv2.SURF_create()

    # Detect keypoints + descriptors
    kps, desc = surf.detectAndCompute(cell_gray, None)

    # No descriptors found → return zero vector
    if desc is None:
        return np.zeros((max_features * 64,), dtype=np.float32)

    # Keep only the first `max_features` descriptors
    desc = desc[:max_features]

    # Flatten (N × 64 → 1D)
    desc = desc.flatten()

    expected_len = max_features * 64

    # Zero-pad if fewer descriptors detected
    if len(desc) < expected_len:
        desc = np.pad(desc, (0, expected_len - len(desc)))

    return desc.astype(np.float32)



def get_surf_feature_names(max_features=128) -> List[str]:
    """
    Generate feature names for SURF descriptor features.

    Each SURF descriptor = 64 elements  
    For `max_features` descriptors:

        total_features = max_features × 64

    Example:
        max_features = 128 → 8192 names:
            surf_bin_0, surf_bin_1, ..., surf_bin_8191

    OUTPUT:
        List[str] : SURF feature names
    """

    return [f'surf_bin_{i}' for i in range(max_features * 64)]
