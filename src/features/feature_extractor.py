import numpy as np
import cv2
from typing import List
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rgb_features import extract_rgb_histogram_features, get_rgb_feature_names
from rgb_hist_features import extract_rgb_hist_features, get_rgb_hist_feature_names
from hsv_features import extract_hsv_color_features, rgb_to_hsv, get_hsv_feature_names
from hsv_hist_features import extract_hsv_hist_color_features, get_hsv_hist_feature_names
from lbp_features import extract_lbp_features, get_lbp_feature_names
from hog_features import extract_hog_features, get_hog_feature_names
from gabor_features import extract_gabor_features, get_gabor_feature_names
from glcm_features import extract_glcm_features, get_glcm_feature_names
from hu_moment_features import extract_hu_moment_features, get_hu_moment_feature_names
from sift_features import extract_sift_features, get_sift_feature_names


def extract_features_from_cell(cell_rgb: np.ndarray) -> np.ndarray:
    """
    Extract all 88 features from a single RGB cell.

    Feature breakdown:
        - RGB Histogram: 14 features
        - HSV Color: 14 features
        - LBP: 10 features
        - HOG: 20 features
        - Gabor: 20 features
        - GLCM: 10 features
        Total: 88 features

    Args:
        cell_rgb: RGB image cell of shape (H, W, 3)
                 Expected size: 75x100 pixels

    Returns:
        Array of 88 features
    """
    # ========== COLOR FEATURES (28 total) ==========

    # 1. RGB Histogram Features (14 features)
    # rgb_features = extract_rgb_histogram_features(cell_rgb)
    rgb_hist_features = extract_rgb_hist_features(cell_rgb)

    # 2. HSV Color Features (14 features)
    cell_hsv = rgb_to_hsv(cell_rgb)
    # hsv_features = extract_hsv_color_features(cell_hsv)
    hsv_hist_features = extract_hsv_hist_color_features(cell_hsv)

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

    hu_moment_features = extract_hu_moment_features(cell_gray)
    sift_features = extract_sift_features(cell_gray)

    

    # ========== COMBINE ALL FEATURES ==========
    all_features = np.concatenate([
        # rgb_features,      # 14
        rgb_hist_features, # 768
        # hsv_features,      # 14
        hsv_hist_features, # 768
        lbp_features,      # 10
        hog_features,      # 20
        gabor_features,    # 20
        glcm_features,     # 10
        hu_moment_features, # 7
        # sift_features      # 16384
    ])

    # print(f"Extracted {len(all_features)} features from cell.")
    # Verify we have exactly 88 features
    # assert len(all_features) == 88, f"Expected 88 features, got {len(all_features)}"
    # assert len(all_features) == 842, f"Expected 842 features, got {len(all_features)}"
    assert len(all_features) == 1605, f"Expected 1605 features, got {len(all_features)}"
    # assert len(all_features) == 17989, f"Expected 17989 features, got {len(all_features)}"
    # assert len(all_features) == 856, f"Expected 856 features, got {len(all_features)}"

    return all_features.astype(np.float32)


def get_all_feature_names() -> List[str]:
    """
    Get names of all 88 features.

    Returns:
        List of 88 feature names
    """
    all_names = []

    # all_names.extend(get_rgb_feature_names())    # 14
    all_names.extend(get_rgb_hist_feature_names())    # 768
    # all_names.extend(get_hsv_feature_names())    # 14
    all_names.extend(get_hsv_hist_feature_names())  # 768
    all_names.extend(get_lbp_feature_names())    # 10
    all_names.extend(get_hog_feature_names())    # 20
    all_names.extend(get_gabor_feature_names())  # 20
    all_names.extend(get_glcm_feature_names())   # 10 / 12
    all_names.extend(get_hu_moment_feature_names())   # 7
    # all_names.extend(get_sift_feature_names())   # 16384

    # assert len(all_names) == 88, f"Expected 88 feature names, got {len(all_names)}"
    assert len(all_names) == 1605, f"Expected 1605 feature names, got {len(all_names)}"
    # assert len(all_names) == 17989, f"Expected 17989 feature names, got {len(all_names)}"
    # assert len(all_names) == 856, f"Expected 856 feature names, got {len(all_names)}"

    return all_names


def extract_features_from_image(image: np.ndarray, cells: List[np.ndarray] = None) -> np.ndarray:
    """
    Extract features from all cells in an image.

    Args:
        image: Full image of shape (600, 800, 3)
        cells: Pre-divided cells (optional). If None, will divide the image.

    Returns:
        Array of shape (64, 88) containing features for all cells
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
        Array of shape (num_images * 64, 88)
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
