import numpy as np
import cv2
from typing import List


def extract_hsv_color_features(cell_hsv: np.ndarray) -> np.ndarray:
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
    # Extract channels
    H = cell_hsv[:, :, 0].astype(np.float32)  # Hue: 0-180
    S = cell_hsv[:, :, 1].astype(np.float32)  # Saturation: 0-255
    V = cell_hsv[:, :, 2].astype(np.float32)  # Value: 0-255

    # Basic statistics (6 features)
    h_mean = np.mean(H)
    s_mean = np.mean(S)
    v_mean = np.mean(V)
    h_std = np.std(H)
    s_std = np.std(S)
    v_std = np.std(V)

    # Hue histogram bins - percentage in color ranges (4 features)
    # Red hue: 0-10 or 170-180 (red wraps around in HSV)
    red_mask = ((H >= 0) & (H <= 10)) | ((H >= 170) & (H <= 180))
    red_pct = np.sum(red_mask) / H.size

    # Green hue: 60-80
    green_mask = (H >= 60) & (H <= 80)
    green_pct = np.sum(green_mask) / H.size

    # Blue hue: 100-130
    blue_mask = (H >= 100) & (H <= 130)
    blue_pct = np.sum(blue_mask) / H.size

    # Brown hue: 10-30
    brown_mask = (H >= 10) & (H <= 30)
    brown_pct = np.sum(brown_mask) / H.size

    # High saturation percentage (1 feature)
    high_sat_pct = np.sum(S > 150) / S.size

    # Color purity (1 feature)
    # Normalized product of saturation and value
    color_purity = np.mean((S * V) / (255.0 * 255.0))

    # Max saturation and value (2 features)
    s_max = np.max(S)
    v_max = np.max(V)

    features = np.array([
        h_mean, s_mean, v_mean,
        h_std, s_std, v_std,
        red_pct, green_pct, blue_pct, brown_pct,
        high_sat_pct,
        color_purity,
        s_max, v_max
    ], dtype=np.float32)

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


def get_hsv_feature_names() -> List[str]:
    """
    Get the names of HSV features.

    Returns:
        List of 14 feature names
    """
    return [
        'h_mean', 's_mean', 'v_mean',
        'h_std', 's_std', 'v_std',
        'red_pct', 'green_pct', 'blue_pct', 'brown_pct',
        'high_sat_pct',
        'color_purity',
        's_max', 'v_max'
    ]
