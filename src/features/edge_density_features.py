import cv2
import numpy as np
from typing import List

def extract_edge_density_features(cell_gray: np.ndarray) -> np.ndarray:
    """
    Edge density computed via Canny edge detector.
        Ball → low edges
        Human → high edges
        Bat/Stump → medium
    """
    edges = cv2.Canny(cell_gray, 50, 150)
    edge_density = edges.mean()  # Avg edge strength

    smoothness = float(cell_gray.std())

    texture_var = cell_gray.var()

    return np.array([edge_density, texture_var, smoothness], dtype=np.float32)

def get_edge_density_feature_names() -> List[str]:
    return ["edge_density", "texture_var", "smoothness"]