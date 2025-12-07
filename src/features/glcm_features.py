import numpy as np
from skimage.feature import graycomatrix, graycoprops
from typing import List


def extract_glcm_features(cell_gray: np.ndarray, distances: List[int] = [1, 3, 5],
                         angles: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4]) -> np.ndarray:
    """
    Extract 13 GLCM (Gray Level Co-occurrence Matrix) texture features from a grayscale cell.

    PHASE 4C ENHANCED: Now uses multi-distance GLCM (distances = [1, 3, 5])

    GLCM measures spatial relationships between pixel intensities.
    Useful for detecting:
        - Smooth ball/bat (high homogeneity, low contrast)
        - Textured grass (low homogeneity, high contrast)
        - Directional textures (stumps)
        - Bat wood grain spans 3-5 pixels (needs larger distances)

    Features:
        - 5 properties × 2 statistics (mean, max across angles) at distance=1 = 10 features
        - Contrast at distance=3 = 1 feature (PHASE 4C)
        - Contrast at distance=5 = 1 feature (PHASE 4C)
        - Contrast range (max - min) across distances = 1 feature (PHASE 4C)
        Total: 13 features (10 original + 3 Phase 4C)

    Args:
        cell_gray: Grayscale image cell of shape (H, W)
        distances: Pixel pair distance offsets (default: [1, 3, 5] - PHASE 4C)
        angles: Angles in radians (default: [0, 45, 90, 135 degrees])

    Returns:
        Array of 13 features (10 original + 3 Phase 4C)
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

        # For each property, compute mean and max across angles at distance=1
        features = []

        for prop in [contrast, dissimilarity, homogeneity, energy, correlation]:
            # prop has shape (num_distances, num_angles)
            # We want statistics across angles at distance=1 (index 0)
            prop_values = prop[0, :]  # Take first distance

            # Mean across angles
            mean_val = np.mean(prop_values)
            features.append(mean_val)

            # Max across angles (captures dominant texture direction)
            max_val = np.max(prop_values)
            features.append(max_val)

        # ========== PHASE 4C: Multi-Distance Features (3 features) ==========
        # Bat wood grain spans 3-5 pixels, needs larger distances to detect
        # Grass: contrast similar at all distances (random)
        # Bat: contrast changes with distance (periodic pattern)

        # Extract contrast at distance=3 and distance=5
        if len(distances) >= 3:
            # Contrast at distance=3 (mean across angles)
            contrast_dist3 = np.mean(contrast[1, :])  # Index 1 = distance 3
            features.append(contrast_dist3)

            # Contrast at distance=5 (mean across angles)
            contrast_dist5 = np.mean(contrast[2, :])  # Index 2 = distance 5
            features.append(contrast_dist5)

            # Contrast range across distances
            # Measures how much contrast changes with distance
            # Bat: high range (periodic pattern)
            # Grass: low range (random, similar at all distances)
            all_contrasts = [np.mean(contrast[i, :]) for i in range(len(distances))]
            contrast_range = max(all_contrasts) - min(all_contrasts)
            features.append(contrast_range)
        else:
            # If fewer distances, pad with zeros
            features.extend([0.0, 0.0, 0.0])

        features = np.array(features, dtype=np.float32)

    except Exception as e:
        # If GLCM fails, return zeros
        print(f"GLCM extraction failed: {e}")
        features = np.zeros(13, dtype=np.float32)

    return features


def get_glcm_feature_names() -> List[str]:
    """
    Get the names of GLCM features (PHASE 4C Enhanced).

    Returns:
        List of 13 feature names (10 original + 3 Phase 4C)
    """
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    names = []

    for prop in properties:
        names.append(f'glcm_{prop}_mean')
        names.append(f'glcm_{prop}_max')

    # Phase 4C: Multi-distance features
    names.extend([
        'glcm_contrast_dist3',
        'glcm_contrast_dist5',
        'glcm_contrast_range'
    ])

    return names
