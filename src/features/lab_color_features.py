import cv2
import numpy as np
from typing import List

def extract_lab_color_features(cell_rgb: np.ndarray) -> np.ndarray:
    """
    Extract mean LAB features.
        L → brightness
        A → red-green axis (skin tones)
        B → yellow-blue axis
    Useful to reject humans and detect ball colors.
    """
    lab = cv2.cvtColor(cell_rgb, cv2.COLOR_BGR2LAB)

    l_mean = lab[:, :, 0].mean()
    a_mean = lab[:, :, 1].mean()
    b_mean = lab[:, :, 2].mean()

    return np.array([l_mean, a_mean, b_mean], dtype=np.float32)


def get_lab_color_feature_names() -> List[str]:
    return ["lab_l_mean", "lab_a_mean", "lab_b_mean"]
