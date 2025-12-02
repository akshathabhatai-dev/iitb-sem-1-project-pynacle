import numpy as np
from skimage.feature import graycomatrix, graycoprops
from typing import List


def extract_glcm_features(cell_gray: np.ndarray, distances: List[int] = [1],
                         angles: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4]) -> np.ndarray:
    """
    Extract 10 GLCM (Gray Level Co-occurrence Matrix) texture features from a grayscale cell.

    GLCM measures spatial relationships between pixel intensities.
    Useful for detecting:
        - Smooth ball/bat (high homogeneity, low contrast)
        - Textured grass (low homogeneity, high contrast)
        - Directional textures (stumps)

    Features:
        - 5 properties × 2 statistics (mean, max across angles) = 10 features
        - Properties: contrast, dissimilarity, homogeneity, energy, correlation

    Args:
        cell_gray: Grayscale image cell of shape (H, W)
        distances: Pixel pair distance offsets (default: [1])
        angles: Angles in radians (default: [0, 45, 90, 135 degrees])

    Returns:
        Array of 10 features
    """
    # Normalize grayscale to reduce number of levels (for faster computation)
    # Reduce from 256 levels to 32 levels
    cell_normalized = (cell_gray / 8).astype(np.uint8)

    try:
        # Compute GLCM
        glcm = graycomatrix(
            cell_normalized,
            distances=distances,
            angles=angles,
            levels=32,
            symmetric=True,
            normed=True
        )

        # Extract 5 properties
        contrast = graycoprops(glcm, 'contrast')
        dissimilarity = graycoprops(glcm, 'dissimilarity')
        homogeneity = graycoprops(glcm, 'homogeneity')
        energy = graycoprops(glcm, 'energy')
        correlation = graycoprops(glcm, 'correlation')

        # For each property, compute mean and max across angles
        features = []

        for prop in [contrast, dissimilarity, homogeneity, energy, correlation]:
            # prop has shape (num_distances, num_angles)
            # We want statistics across angles
            prop_values = prop[0, :]  # Take first distance

            # Mean across angles
            mean_val = np.mean(prop_values)
            features.append(mean_val)

            # Max across angles (captures dominant texture direction)
            max_val = np.max(prop_values)
            features.append(max_val)

        features = np.array(features, dtype=np.float32)

    except Exception as e:
        # If GLCM fails, return zeros
        print(f"GLCM extraction failed: {e}")
        features = np.zeros(10, dtype=np.float32)

    return features


def get_glcm_feature_names() -> List[str]:
    """
    Get the names of GLCM features.

    Returns:
        List of 10 feature names
    """
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    names = []

    for prop in properties:
        names.append(f'glcm_{prop}_mean')
        names.append(f'glcm_{prop}_max')

    return names
