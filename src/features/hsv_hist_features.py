import numpy as np
import cv2
from typing import List


def extract_hsv_hist_color_features(cell_hsv: np.ndarray) -> np.ndarray:
    """
    Extract 14 HSV color features from a cell.

    Features:
        - 6 basic statistics: h_mean, s_mean, v_mean, h_std, s_std, v_std
        - 4 hue histogram bins: red_pct, green_pct, blue_pct, brown_pct
        - 1 high saturation percentage
        - 1 color purity
        - 2 max values: s_max, v_max

    Args:
        cell_hsv: HSV image cell of shape (H, W, 3)
                 In OpenCV: H in [0, 180], S in [0, 255], V in [0, 255]

    Returns:
        Array of 14 features
    """
    hsv = rgb_to_hsv(cell_hsv)
    hist_hsv = []
    for i in range(3):
        h = cv2.calcHist([hsv], [i], None, [256], [0, 256]).flatten()
        hist_hsv.extend(h)

    features = np.array(hist_hsv, dtype=np.float32)

    return features


def rgb_to_hsv(cell_rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB cell to HSV.

    Args:
        cell_rgb: RGB image cell (BGR in OpenCV)

    Returns:
        HSV image cell
    """
    return cv2.cvtColor(cell_rgb, cv2.COLOR_BGR2HSV)


def get_hsv_hist_feature_names() -> List[str]:
    """
    Get the names of HSV features.

    Returns:
        List of 14 feature names
    """
    return [f'hsv_hist_bin_{i}' for i in range(768)]

