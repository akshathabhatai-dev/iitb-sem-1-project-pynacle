import numpy as np
from typing import List

def extract_smoothness_features(cell_gray: np.ndarray) -> np.ndarray:
    """
    Smoothness = standard deviation of pixel intensity.
        Ball → low (smooth)
        Bat/Stump → medium
        Human → high (texture)
    """
    smoothness = float(cell_gray.std())

    return np.array([smoothness], dtype=np.float32)


def get_smoothness_feature_names() -> List[str]:
    return ["smoothness"]
