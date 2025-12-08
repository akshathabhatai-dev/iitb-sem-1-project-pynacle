import numpy as np
from skimage.feature import local_binary_pattern
from typing import List


def extract_lbp_features(cell_gray: np.ndarray, n_points: int = 8, radii: List[int] = [1, 2, 3]) -> np.ndarray:
    """
    Extract 16 Local Binary Pattern features from a grayscale cell (PHASE 4C Enhanced).

    PHASE 4C: Now uses multi-scale LBP (radii = [1, 2, 3]) for better texture discrimination.

    LBP captures local texture by comparing each pixel with its neighbors.
    Uniform patterns are rotation-invariant texture descriptors.

    Features:
        - 10 histogram bins from uniform LBP patterns at radius=1
        - 6 multi-scale statistics: entropy at r=1,2,3 + variance at r=1,2,3 (PHASE 4C)
        Total: 16 features (10 original + 6 Phase 4C)

    Bat wood grain spans 5-15 pixels, needs larger radii to capture.

    Args:
        cell_gray: Grayscale image cell of shape (H, W)
        n_points: Number of circularly symmetric neighbor points (default: 8)
        radii: List of radii to use (default: [1, 2, 3] - PHASE 4C)

    Returns:
        Array of 16 features (10 original + 6 Phase 4C)
    """
    # ========== ORIGINAL LBP FEATURES (10 features) ==========
    # Compute LBP at radius=1
    # Method 'uniform' gives rotation-invariant uniform patterns
    # For 8 neighbors, this produces 10 unique patterns (including non-uniform)
    lbp_image_r1 = local_binary_pattern(cell_gray, n_points, radii[0], method='uniform')

    # Compute histogram at radius=1
    # For n_points=8, uniform LBP has values in range [0, n_points+1] = [0, 9]
    # But we want 10 bins total
    n_bins = n_points + 2  # 10 bins for 8 neighbors
    hist_r1, _ = np.histogram(lbp_image_r1.ravel(),
                             bins=n_bins,
                             range=(0, n_bins),
                             density=True)  # Normalize to sum to 1

    features = list(hist_r1.astype(np.float32))

    # ========== PHASE 4C: MULTI-SCALE LBP FEATURES (6 features) ==========
    # Compute LBP statistics at multiple radii
    # Bat wood grain: texture changes at different scales
    # Grass: random, similar at all scales

    multi_scale_entropies = []
    multi_scale_variances = []

    for r in radii:
        # Compute LBP at this radius
        lbp_image = local_binary_pattern(cell_gray, n_points, r, method='uniform')

        # Compute histogram
        hist, _ = np.histogram(lbp_image.ravel(),
                              bins=n_bins,
                              range=(0, n_bins),
                              density=True)

        # 1. Entropy: Measures randomness of LBP distribution
        # Bat: lower entropy (structured wood grain pattern)
        # Grass: higher entropy (random texture)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        hist = hist + epsilon
        entropy = -np.sum(hist * np.log2(hist))
        multi_scale_entropies.append(entropy)

        # 2. Variance: Measures spread of LBP distribution
        # Higher variance = more diverse patterns
        variance = np.var(hist)
        multi_scale_variances.append(variance)

    # Add multi-scale statistics
    features.extend(multi_scale_entropies)   # 3 features (r=1, 2, 3)
    features.extend(multi_scale_variances)   # 3 features (r=1, 2, 3)

    features = np.array(features, dtype=np.float32)

    # Verify we have exactly 16 features (10 histogram + 3 entropy + 3 variance)
    assert len(features) == 16, f"Expected 16 features, got {len(features)}"

    # Replace any NaN or Inf values with 0.0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def get_lbp_feature_names() -> List[str]:
    """
    Get the names of LBP features (PHASE 4C Enhanced).

    Returns:
        List of 16 feature names (10 original + 6 Phase 4C)
    """
    names = [f'lbp_bin_{i}' for i in range(10)]

    # Phase 4C: Multi-scale statistics
    names.extend([
        'lbp_entropy_r1',
        'lbp_entropy_r2',
        'lbp_entropy_r3',
        'lbp_variance_r1',
        'lbp_variance_r2',
        'lbp_variance_r3'
    ])

    return names
