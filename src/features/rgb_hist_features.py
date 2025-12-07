import cv2
import numpy as np
from typing import List


def extract_rgb_hist_features(cell_rgb: np.ndarray) -> np.ndarray:
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
    hist_rgb = []
    for i in range(3):
        h = cv2.calcHist([cell_rgb], [i], None, [256], [0, 256]).flatten()
        hist_rgb.extend(h)
    features = np.array(hist_rgb, dtype=np.float32)

    return features


def get_rgb_hist_feature_names() -> List[str]:
    """
    Get the names of RGB features.

    Returns:
        List of 14 feature names
    """
    
    return [f'rgb_hist_bin_{i}' for i in range(768)]
