import numpy as np
from typing import List


def extract_rgb_histogram_features(cell_rgb: np.ndarray) -> np.ndarray:
    """
    Extract 14 RGB histogram-based features from a cell.

    Features:
        - 6 basic statistics: r_mean, g_mean, b_mean, r_std, g_std, b_std
        - 3 channel ratios: r_ratio, g_ratio, b_ratio
        - 1 dominant channel: which color is strongest (0=R, 1=G, 2=B)
        - 1 color variance: uniformity of color
        - 3 inter-channel differences: r_g_diff, r_b_diff, g_b_diff

    Args:
        cell_rgb: RGB image cell of shape (H, W, 3)

    Returns:
        Array of 14 features
    """
    # Extract channels
    R = cell_rgb[:, :, 2].astype(np.float32)  # OpenCV uses BGR, so R is channel 2
    G = cell_rgb[:, :, 1].astype(np.float32)
    B = cell_rgb[:, :, 0].astype(np.float32)

    # Basic statistics (6 features)
    r_mean = np.mean(R)
    g_mean = np.mean(G)
    b_mean = np.mean(B)
    r_std = np.std(R)
    g_std = np.std(G)
    b_std = np.std(B)

    # Channel ratios (3 features)
    total = r_mean + g_mean + b_mean + 1e-6  # Avoid division by zero
    r_ratio = r_mean / total
    g_ratio = g_mean / total
    b_ratio = b_mean / total

    # Dominant channel (1 feature)
    dominant_channel = np.argmax([r_mean, g_mean, b_mean])  # 0=R, 1=G, 2=B

    # Color variance (1 feature)
    # Variance across RGB channels at each pixel
    color_variance = np.mean(np.std(cell_rgb, axis=2))

    # Inter-channel differences (3 features)
    r_g_diff = abs(r_mean - g_mean)
    r_b_diff = abs(r_mean - b_mean)
    g_b_diff = abs(g_mean - b_mean)

    features = np.array([
        r_mean, g_mean, b_mean,
        r_std, g_std, b_std,
        r_ratio, g_ratio, b_ratio,
        dominant_channel,
        color_variance,
        r_g_diff, r_b_diff, g_b_diff
    ], dtype=np.float32)

    return features


def get_rgb_feature_names() -> List[str]:
    """
    Get the names of RGB features.

    Returns:
        List of 14 feature names
    """
    return [
        'r_mean', 'g_mean', 'b_mean',
        'r_std', 'g_std', 'b_std',
        'r_ratio', 'g_ratio', 'b_ratio',
        'dominant_channel',
        'color_variance',
        'r_g_diff', 'r_b_diff', 'g_b_diff'
    ]
