import cv2
import numpy as np
from typing import List


def extract_sift_features(cell_gray: np.ndarray, max_features=128) -> np.ndarray:
    """
    Extract 14 RGB histogram-based features from a cell.

    Features:
        - 6 basic statistics: r_mean, g_mean, b_mean, r_std, g_std, b_std
        - 3 channel ratios: r_ratio, g_ratio, b_ratio
        - 1 dominant channel: which color is strongest (0=R, 1=G, 2=B)
        - 1 color variance: uniformity of color
        - 3 inter-channel differences: r_g_diff, r_b_diff, g_b_diff

    Args:
        cell_rgb: RGB image cell of shape (H, W, 3)

    Returns:
        Array of 14 features
    """
    # Extract channels
    sift = cv2.SIFT_create()
    kps, desc = sift.detectAndCompute(cell_gray, None)

    if desc is None:
        return np.zeros((max_features * 128,))  # SIFT size=128

    desc = desc[:max_features]  # keep first N descriptors
    desc = desc.flatten()

    # pad if needed
    if len(desc) < max_features * 128:
        desc = np.pad(desc, (0, max_features * 128 - len(desc)))
    features = desc.astype(np.float32)


    return features


def get_sift_feature_names() -> List[str]:
    """
    Get the names of SIFT features.

    Returns:
        List of 16384 feature names
    """
    
    return [f'sift_bin_{i}' for i in range(128 * 128)]
