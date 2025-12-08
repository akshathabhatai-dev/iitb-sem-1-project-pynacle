import numpy as np
import cv2
from typing import List


def extract_hsv_hist_color_features(cell_hsv: np.ndarray, no_of_bins: int = 256) -> np.ndarray:
    """
    Extract a fixed-length **768-dimensional HSV histogram feature vector** 
    from a single image cell.

    Features:
        For HSV, we compute:
            - 256-bin histogram for Hue channel
            - 256-bin histogram for Saturation channel
            - 256-bin histogram for Value channel

        Total = 256 + 256 + 256 = **768 features**

    Args:
        cell_hsv: HSV image cell of shape (H, W, 3)
                 In OpenCV: H in [0, 180], S in [0, 255], V in [0, 255]

    Returns:
        np.ndarray
            768-dim array (float32), representing the concatenated histograms:
                [Hue_256_bins, Saturation_256_bins, Value_256_bins]
    """
    # Convert BGR input to HSV (ensures a consistent HSV representation)
    hsv = rgb_to_hsv(cell_hsv)
    hist_hsv = []
    # no_of_bins = 256

    # Compute 256-bin histogram for each channel H, S, V
    for i in range(3):
        # Histogram range is [0, 256] since OpenCV expects that format
        h = cv2.calcHist([hsv], [i], None, [no_of_bins], [0, 256]).flatten()
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


def get_hsv_hist_feature_names(no_of_bins=256) -> List[str]:
    """
    Returns the **names of all 768 HSV histogram features**.

    Naming convention:
        hsv_hist_bin_0
        hsv_hist_bin_1
        ...
        hsv_hist_bin_767

    Returns:
        List[str] : 768 feature names
    """
    return [f'hsv_hist_bin_{i}' for i in range(3 * no_of_bins)]

