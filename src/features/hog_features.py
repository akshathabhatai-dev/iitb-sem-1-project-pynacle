import numpy as np
from skimage.feature import hog
from typing import List


def extract_hog_features(cell_gray: np.ndarray) -> np.ndarray:
    """
    Extract 576 HOG (Histogram of Oriented Gradients) features from a grayscale cell.

    HOG captures edge orientations and shape information.
    Useful for detecting:
        - Vertical stumps (90° gradients)
        - Horizontal/elongated bat (0°, 45°, 135° gradients)
        - Circular ball (uniform gradient distribution)

    Features:
        - 576 features from HOG descriptor
        - For 75x100 cell: 5x5 cell grid -> 4x4 blocks -> 16 blocks * 36 features/block = 576

    Args:
        cell_gray: Grayscale image cell of shape (H, W)
                  Expected size: 75x100 pixels

    Returns:
        Array of 576 features
    """
    # HOG parameters
    orientations = 9          # Number of orientation bins (0-180° in 20° increments)
    pixels_per_cell = (15, 20)  # Cell size (75/5=15, 100/5=20 for 5x5 grid)
    cells_per_block = (2, 2)  # Block normalization

    # Compute HOG descriptor
    try:
        hog_descriptor = hog(
            cell_gray,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm='L2-Hys',
            visualize=False,
            feature_vector=True
        )

        # For 75x100 cells, this will always be 576 features
        features = hog_descriptor

    except Exception as e:
        # If HOG fails (e.g., image too small), return zeros
        print(f"HOG extraction failed: {e}")
        features = np.zeros(576, dtype=np.float32)

    return features.astype(np.float32)


def extract_hog_features_alternative(cell_gray: np.ndarray) -> np.ndarray:
    """
    Alternative HOG extraction with different parameters for better feature count control.

    This version uses smaller cells to get a more compact descriptor.

    Args:
        cell_gray: Grayscale image cell of shape (H, W)

    Returns:
        Array of 20 features
    """
    # Alternative parameters for more compact descriptor
    orientations = 9
    pixels_per_cell = (25, 25)  # Larger cells = fewer features
    cells_per_block = (1, 1)    # No block normalization

    try:
        hog_descriptor = hog(
            cell_gray,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm='L2-Hys',
            visualize=False,
            feature_vector=True
        )

        if len(hog_descriptor) >= 20:
            features = hog_descriptor[:20]
        else:
            features = np.zeros(20, dtype=np.float32)
            features[:len(hog_descriptor)] = hog_descriptor

    except Exception as e:
        print(f"HOG extraction failed: {e}")
        features = np.zeros(20, dtype=np.float32)

    return features.astype(np.float32)


def get_hog_feature_names() -> List[str]:
    """
    Get the names of HOG features.

    Returns:
        List of 576 feature names
    """
    return [f'hog_{i}' for i in range(576)]
