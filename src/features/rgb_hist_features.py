import cv2
import numpy as np
from typing import List


def extract_rgb_hist_features(cell_rgb: np.ndarray) -> np.ndarray:
    """
    Extract a **768-dimensional RGB histogram feature vector**
    from an RGB image cell.

    WHAT THIS FUNCTION ACTUALLY DOES:
    -------------------------------------------------------------
    For each of the 3 RGB channels (R, G, B):
        - Compute a 256-bin histogram
        - Flatten and concatenate them

    So total features:
        256 (R) + 256 (G) + 256 (B) = **768 features**

    WHY RGB HISTOGRAMS?
    -------------------------------------------------------------
    RGB histograms capture the **color distribution** inside a cell,
    independent of spatial arrangement. Useful for:
        • texture/color recognition
        • object/background discrimination
        • global color patterns

    INPUT:
        cell_rgb : np.ndarray
            An RGB image patch of shape (H, W, 3)

    OUTPUT:
        np.ndarray (float32)
            A 768-length feature vector:
                [R_256_bins, G_256_bins, B_256_bins]
    """

    hist_rgb = []

    # Compute histogram for each channel: 0=R, 1=G, 2=B
    for i in range(3):
        h = cv2.calcHist([cell_rgb], [i], None, [256], [0, 256]).flatten()
        hist_rgb.extend(h)

    # Final feature vector (768-dim)
    return np.array(hist_rgb, dtype=np.float32)



def get_rgb_hist_feature_names() -> List[str]:
    """
    Generate names for all 768 RGB histogram features.

    Naming pattern:
        rgb_hist_bin_0
        rgb_hist_bin_1
        ...
        rgb_hist_bin_767

    OUTPUT:
        List[str] of size 768
    """
    return [f'rgb_hist_bin_{i}' for i in range(768)]
