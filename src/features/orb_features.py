import cv2
import numpy as np
from typing import List


def extract_orb_features(cell_gray: np.ndarray, max_features=128) -> np.ndarray:
    """
    Extract a fixed-length ORB descriptor feature vector from a grayscale image cell.

    WHAT ORB PROVIDES:
    ---------------------------------------------------------
    ORB (Oriented FAST and Rotated BRIEF) computes:
        - Keypoints (interest points)
        - Binary descriptors (each descriptor = 32 bytes = 256 bits)

    THIS FUNCTION RETURNS:
    ---------------------------------------------------------
    A **flattened ORB descriptor vector of length:**

        max_features × 32

    Example:
        max_features = 128  → output = 128 × 32 = **4096 features**

    The function ensures *fixed length* by:
        - Taking only the first N descriptors
        - Flattening into 1D
        - Zero-padding if fewer than max_features are detected

    INPUT:
        cell_gray : grayscale cell (H × W)
        max_features : maximum ORB keypoints/descriptors to include

    OUTPUT:
        features : np.ndarray of shape (max_features × 32,), dtype=float32
    """

    # Initialize ORB extractor
    orb = cv2.ORB_create(nfeatures=max_features)

    # Detect keypoints + compute descriptors
    kps, desc = orb.detectAndCompute(cell_gray, None)

    # If no descriptors found → return zero vector
    if desc is None:
        return np.zeros((max_features * 32,), dtype=np.float32)

    # Keep only first max_features descriptors
    desc = desc[:max_features]

    # Flatten N × 32 descriptor matrix → 1D vector
    desc = desc.flatten()

    # Expected final length
    expected_len = max_features * 32

    # Pad with zeros if fewer descriptors were detected
    if len(desc) < expected_len:
        desc = np.pad(desc, (0, expected_len - len(desc)))

    return desc.astype(np.float32)



def get_orb_feature_names(max_features=128) -> List[str]:
    """
    Generate feature names for ORB descriptors.

    Each ORB descriptor has 32 dimensions.
    For max_features keypoints → total = max_features × 32 features.

    Example:
        max_features = 128 → 4096 feature names:
            orb_bin_0, orb_bin_1, ..., orb_bin_4095

    OUTPUT:
        List[str] of length (max_features × 32)
    """

    return [f'orb_bin_{i}' for i in range(max_features * 32)]
