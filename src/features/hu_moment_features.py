import cv2
import numpy as np
from typing import List


def extract_hu_moment_features(cell_gray: np.ndarray) -> np.ndarray:
    """
    Extract **7 Hu Moment features** from a grayscale cell.

    WHAT ARE HU MOMENTS?
    -------------------------------------------------------------
    Hu Moments are a set of **seven shape descriptors** derived from 
    image moments. They are:

        • Translation invariant
        • Scale invariant
        • Rotation invariant

    Meaning:
        If the object shifts position, rotates, or changes size,
        Hu moments will largely remain consistent. This makes them 
        ideal for shape recognition and contour-based classification.

    COMPUTATION STEPS:
    -------------------------------------------------------------
    1. Compute raw spatial + central moments using cv2.moments()
    2. Convert them into 7 invariant Hu moments using cv2.HuMoments()
    3. Flatten to a 1D vector of length 7
    4. Convert to float32 for ML models

    INPUT:
        cell_gray (np.ndarray)
            Grayscale image patch (H × W)

    OUTPUT:
        np.ndarray of shape (7,)
            Hu Moments in order: hu_0 ... hu_6
    """

    # Compute spatial & central moments
    moments = cv2.moments(cell_gray)

    # Compute 7 Hu moments (returns 7×1 vector)
    hu = cv2.HuMoments(moments).flatten()

    # Convert to float32 for consistency
    return hu.astype(np.float32)



def get_hu_moment_feature_names() -> List[str]:
    """
    Returns the names of the 7 Hu Moment features.

    OUTPUT:
        ['hu_moment_0', ..., 'hu_moment_6']
    """
    return [f'hu_moment_{i}' for i in range(7)]
