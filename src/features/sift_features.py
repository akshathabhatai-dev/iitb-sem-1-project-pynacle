import cv2
import numpy as np
from typing import List


def extract_sift_features(cell_gray: np.ndarray, max_features=128) -> np.ndarray:
    """
    Extract a fixed-length SIFT descriptor vector from a grayscale image cell.

    WHAT THIS FUNCTION ACTUALLY DOES:
    -------------------------------------------------------------
    - Detects SIFT keypoints inside the cell
    - Computes SIFT descriptors (each descriptor = 128 dimensions)
    - Takes the first `max_features` descriptors
    - Flattens them into a 1D vector
    - Pads with zeros if fewer descriptors were detected

    FINAL FEATURE LENGTH:
        max_features × 128

    Example:
        max_features = 128  → 128 × 128 = **16384 features**

    WHY ZERO-PADDING?
    -------------------------------------------------------------
    SIFT detects fewer keypoints in smooth or low-texture cells,
    so padding ensures **every cell outputs a fixed-length feature vector**,
    which is required for ML models and pipelines.

    INPUT:
        cell_gray : grayscale cell image (H × W)
        max_features : maximum SIFT keypoints to include (default = 128)

    OUTPUT:
        np.ndarray of shape (max_features × 128,), dtype=float32
    """

    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Compute keypoints + descriptors
    kps, desc = sift.detectAndCompute(cell_gray, None)

    # If no descriptors, return a zero vector
    if desc is None:
        return np.zeros((max_features * 128,), dtype=np.float32)

    # Keep only the first max_features descriptors
    desc = desc[:max_features]

    # Flatten descriptor matrix (N × 128 → 1D vector)
    desc = desc.flatten()

    expected_len = max_features * 128

    # If fewer descriptors detected → pad with zeros
    if len(desc) < expected_len:
        desc = np.pad(desc, (0, expected_len - len(desc)))

    return desc.astype(np.float32)



def get_sift_feature_names(max_features=128) -> List[str]:
    """
    Generate feature names for SIFT descriptor features.

    TOTAL FEATURES:
        max_features × 128

    Example:
        max_features = 128 → 16384 names:
            sift_bin_0, sift_bin_1, ..., sift_bin_16383

    OUTPUT:
        List[str]: SIFT feature names
    """

    return [f'sift_bin_{i}' for i in range(max_features * 128)]
