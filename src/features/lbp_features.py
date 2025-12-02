import numpy as np
from skimage.feature import local_binary_pattern
from typing import List


def extract_lbp_features(cell_gray: np.ndarray, n_points: int = 8, radius: int = 1) -> np.ndarray:
    """
    Extract 10 Local Binary Pattern features from a grayscale cell.

    LBP captures local texture by comparing each pixel with its neighbors.
    Uniform patterns are rotation-invariant texture descriptors.

    Features:
        - 10 histogram bins from uniform LBP patterns

    Args:
        cell_gray: Grayscale image cell of shape (H, W)
        n_points: Number of circularly symmetric neighbor points (default: 8)
        radius: Radius of circle (default: 1)

    Returns:
        Array of 10 features (normalized histogram)
    """
    # Compute LBP
    # Method 'uniform' gives rotation-invariant uniform patterns
    # For 8 neighbors, this produces 10 unique patterns (including non-uniform)
    lbp_image = local_binary_pattern(cell_gray, n_points, radius, method='uniform')

    # Compute histogram
    # For n_points=8, uniform LBP has values in range [0, n_points+1] = [0, 9]
    # But we want 10 bins total
    n_bins = n_points + 2  # 10 bins for 8 neighbors
    hist, _ = np.histogram(lbp_image.ravel(),
                          bins=n_bins,
                          range=(0, n_bins),
                          density=True)  # Normalize to sum to 1

    features = hist.astype(np.float32)

    return features


def get_lbp_feature_names() -> List[str]:
    """
    Get the names of LBP features.

    Returns:
        List of 10 feature names
    """
    return [f'lbp_bin_{i}' for i in range(10)]
