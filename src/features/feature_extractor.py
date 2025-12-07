import numpy as np
import cv2
from typing import List
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rgb_features import extract_rgb_histogram_features, get_rgb_feature_names
from hsv_features import extract_hsv_color_features, rgb_to_hsv, get_hsv_feature_names
from lbp_features import extract_lbp_features, get_lbp_feature_names
from hog_features import extract_hog_features, get_hog_feature_names
from gabor_features import extract_gabor_features, get_gabor_feature_names
from glcm_features import extract_glcm_features, get_glcm_feature_names

# PHASE 1 FEATURES - Enhanced detection for ball/bat/stump
from enhanced_color_features import extract_enhanced_color_features, get_enhanced_color_feature_names
from shape_features import extract_shape_features, get_shape_feature_names
from edge_features import extract_edge_features, get_edge_feature_names

# PHASE 2 FEATURES - Keypoint & Texture refinement
from morphological_features import extract_morphological_features, get_morphological_feature_names
from sift_features import extract_sift_features, get_sift_feature_names
from enhanced_texture_features import extract_enhanced_texture_features, get_enhanced_texture_feature_names

# PHASE 3 FEATURES - Advanced patterns (SLIC, Haar, Color Moments)
from superpixel_features import extract_superpixel_features, get_superpixel_feature_names
from haar_features import extract_haar_features, get_haar_feature_names
from advanced_color_moments import extract_advanced_color_moments, get_advanced_color_moment_names


def extract_features_from_cell(cell_rgb: np.ndarray) -> np.ndarray:
    """
    Extract all 332 features from a single RGB cell.

    Feature breakdown (PHASE 3 - Advanced Patterns):
        ORIGINAL FEATURES (212):
        - RGB Histogram: 14 features
        - HSV Color: 14 features
        - LBP: 10 features
        - HOG: 144 features (REDUCED from 576 to prevent PCA dominance)
        - Gabor: 20 features
        - GLCM: 10 features

        PHASE 1 ADDITIONS (54):
        - Enhanced Color Features: 18 features (red detection for ball)
        - Shape Features: 24 features (circularity, aspect ratio)
        - Edge Features: 12 features (Canny, Sobel gradients)

        PHASE 2 ADDITIONS (36):
        - Morphological Features: 8 features (H-Maxima/H-Minima for ball/stump peaks)
        - SIFT Features: 14 features (keypoint detection for objects)
        - Enhanced Texture Features: 14 features (Haralick for bat wood texture)

        PHASE 3 ADDITIONS (30):
        - SLIC Superpixel Features: 12 features (region-based segmentation)
        - Haar-like Features: 10 features (rectangle patterns for edges/lines)
        - Advanced Color Moments: 8 features (skewness, kurtosis, YCrCb)

        Total: 332 features (212 + 54 + 36 + 30)

    Args:
        cell_rgb: RGB image cell of shape (H, W, 3)
                 Expected size: 75x100 pixels

    Returns:
        Array of 332 features
    """
    # ========== COLOR FEATURES (28 total) ==========

    # 1. RGB Histogram Features (14 features)
    rgb_features = extract_rgb_histogram_features(cell_rgb)

    # 2. HSV Color Features (14 features)
    cell_hsv = rgb_to_hsv(cell_rgb)
    hsv_features = extract_hsv_color_features(cell_hsv)

    # ========== TEXTURE FEATURES (60 total) ==========

    # Convert to grayscale for texture features
    cell_gray = cv2.cvtColor(cell_rgb, cv2.COLOR_BGR2GRAY)

    # 3. Local Binary Patterns (10 features)
    lbp_features = extract_lbp_features(cell_gray)

    # 4. Histogram of Oriented Gradients (20 features)
    hog_features = extract_hog_features(cell_gray)

    # 5. Gabor Filters (20 features)
    gabor_features = extract_gabor_features(cell_gray)

    # 6. GLCM (10 features)
    glcm_features = extract_glcm_features(cell_gray)

    # ========== PHASE 1 FEATURES (54 total) ==========

    # 7. Enhanced Color Features (18 features)
    # Critical for ball detection - red-specific discrimination
    enhanced_color_features = extract_enhanced_color_features(cell_rgb)

    # 8. Shape Features (24 features)
    # Critical for ball (circularity), bat (aspect ratio), stump (vertical)
    shape_features = extract_shape_features(cell_rgb)

    # 9. Edge Features (12 features)
    # Critical for object boundaries and orientation
    edge_features = extract_edge_features(cell_rgb)

    # ========== PHASE 2 FEATURES (36 total) ==========

    # 10. Morphological Features (8 features)
    # Critical for ball (red peak), stump (white peak) via H-Maxima/H-Minima
    morphological_features = extract_morphological_features(cell_rgb)

    # 11. SIFT Features (14 features)
    # Critical for object keypoint detection and scale-invariant features
    sift_features = extract_sift_features(cell_rgb)

    # 12. Enhanced Texture Features (14 features)
    # Critical for bat (wood texture) via Haralick descriptors
    enhanced_texture_features = extract_enhanced_texture_features(cell_gray)

    # ========== PHASE 3 FEATURES (30 total) ==========

    # 13. SLIC Superpixel Features (12 features)
    # Critical for region-based color segmentation
    superpixel_features = extract_superpixel_features(cell_rgb)

    # 14. Haar-like Features (10 features)
    # Critical for stump (vertical patterns), bat (horizontal patterns)
    haar_features = extract_haar_features(cell_gray)

    # 15. Advanced Color Moments (8 features)
    # Critical for ball (red skewness, kurtosis, YCrCb chrominance)
    advanced_color_moments = extract_advanced_color_moments(cell_rgb)

    # ========== COMBINE ALL FEATURES ==========
    all_features = np.concatenate([
        # Original features (212 - HOG reduced)
        rgb_features,      # 14
        hsv_features,      # 14
        lbp_features,      # 10
        hog_features,      # 144 (REDUCED from 576)
        gabor_features,    # 20
        glcm_features,     # 10

        # Phase 1 features (54)
        enhanced_color_features,  # 18
        shape_features,           # 24
        edge_features,            # 12

        # Phase 2 features (36)
        morphological_features,   # 8
        sift_features,            # 14
        enhanced_texture_features,# 14

        # Phase 3 features (30)
        superpixel_features,      # 12
        haar_features,            # 10
        advanced_color_moments    # 8
    ])

    # Verify we have exactly 332 features (212 + 54 + 36 + 30)
    assert len(all_features) == 332, f"Expected 332 features, got {len(all_features)}"

    return all_features.astype(np.float32)


def get_all_feature_names() -> List[str]:
    """
    Get names of all 332 features (PHASE 3 - Advanced Patterns).

    Returns:
        List of 332 feature names
    """
    all_names = []

    # Original features (212 - HOG reduced)
    all_names.extend(get_rgb_feature_names())    # 14
    all_names.extend(get_hsv_feature_names())    # 14
    all_names.extend(get_lbp_feature_names())    # 10
    all_names.extend(get_hog_feature_names())    # 144 (REDUCED from 576)
    all_names.extend(get_gabor_feature_names())  # 20
    all_names.extend(get_glcm_feature_names())   # 10

    # Phase 1 features (54)
    all_names.extend(get_enhanced_color_feature_names())  # 18
    all_names.extend(get_shape_feature_names())           # 24
    all_names.extend(get_edge_feature_names())            # 12

    # Phase 2 features (36)
    all_names.extend(get_morphological_feature_names())   # 8
    all_names.extend(get_sift_feature_names())            # 14
    all_names.extend(get_enhanced_texture_feature_names()) # 14

    # Phase 3 features (30)
    all_names.extend(get_superpixel_feature_names())      # 12
    all_names.extend(get_haar_feature_names())            # 10
    all_names.extend(get_advanced_color_moment_names())   # 8

    assert len(all_names) == 332, f"Expected 332 feature names, got {len(all_names)}"

    return all_names


def extract_features_from_image(image: np.ndarray, cells: List[np.ndarray] = None) -> np.ndarray:
    """
    Extract features from all cells in an image.

    Args:
        image: Full image of shape (600, 800, 3)
        cells: Pre-divided cells (optional). If None, will divide the image.

    Returns:
        Array of shape (64, 332) containing features for all cells (PHASE 3 - Advanced Patterns)
    """
    # Import here to avoid circular imports
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_prep'))
    from grid_utils import divide_into_grid

    if cells is None:
        cells = divide_into_grid(image, grid_size=(8, 8))

    features_list = []
    for cell in cells:
        features = extract_features_from_cell(cell)
        features_list.append(features)

    return np.array(features_list, dtype=np.float32)


def extract_features_from_images(images: List[np.ndarray], verbose: bool = True) -> np.ndarray:
    """
    Extract features from multiple images.

    Args:
        images: List of images, each of shape (600, 800, 3)
        verbose: Print progress if True

    Returns:
        Array of shape (num_images * 64, 332) - PHASE 3: 332 features per cell (Advanced Patterns)
    """
    all_features = []

    for idx, image in enumerate(images):
        if verbose and (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(images)} images...")

        image_features = extract_features_from_image(image)
        all_features.append(image_features)

    # Concatenate all features
    all_features = np.vstack(all_features)

    if verbose:
        print(f"Feature extraction complete: {all_features.shape}")

    return all_features


if __name__ == '__main__':
    # Test feature extraction
    print("Testing feature extraction...")

    # Create a dummy cell
    dummy_cell = np.random.randint(0, 256, (75, 100, 3), dtype=np.uint8)

    # Extract features
    features = extract_features_from_cell(dummy_cell)

    print(f"Extracted {len(features)} features")
    print(f"Feature names ({len(get_all_feature_names())} total):")
    for i, name in enumerate(get_all_feature_names()):
        print(f"  {i+1:2d}. {name}: {features[i]:.4f}")
