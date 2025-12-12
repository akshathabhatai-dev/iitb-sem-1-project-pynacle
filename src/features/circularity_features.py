import cv2
import numpy as np
from typing import List

def extract_circularity_features(cell_gray: np.ndarray) -> np.ndarray:
    """
    Extract circularity feature for the given grayscale cell.
    Circularity = 4πA / P²
        Ball → high (~0.6–1.0)
        Bat/Stump → medium
        Human → low
    """
    contours, _ = cv2.findContours(cell_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        circularity = (4 * np.pi * area) / (peri * peri + 1e-7) if peri > 0 else 0
    else:
        circularity = 0

    return np.array([circularity], dtype=np.float32)


def get_circularity_feature_names() -> List[str]:
    return ["circularity"]
