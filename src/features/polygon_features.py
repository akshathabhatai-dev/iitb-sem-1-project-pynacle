import cv2
import numpy as np
from typing import List

def extract_polygon_features(cell_gray: np.ndarray) -> np.ndarray:
    """
    Computes number of polygon sides from contour approximation.
        Ball → ~8–12
        Bat/Stump → ~4
        Human → 15+
    This helps reject humans and detect structured objects.
    """
    contours, _ = cv2.findContours(cell_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        poly_sides = len(approx)
    else:
        poly_sides = 0

    return np.array([poly_sides], dtype=np.float32)


def get_polygon_feature_names() -> List[str]:
    return ["polygon_sides"]
