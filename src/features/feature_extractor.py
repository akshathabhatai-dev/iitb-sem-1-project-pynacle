import numpy as np
import cv2
from typing import List
import sys
import os

from src.features import lab_color_features

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
from orb_features import extract_orb_features, get_orb_feature_names
from neighborhood_hog_features import compute_hog_per_cell, extract_na_aware_hog_for_grid_features, get_na_aware_hog_feature_names

from circularity_features import extract_circularity_features, get_circularity_feature_names
from edge_density_features import extract_edge_density_features, get_edge_density_feature_names
from lab_color_features import extract_lab_color_features, get_lab_color_feature_names
from smoothness_features import extract_smoothness_features, get_smoothness_feature_names
from polygon_features import extract_polygon_features, get_polygon_feature_names

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

# PHASE 4A FEATURES - Advanced Color (RG Chromaticity, YUV, Fine Hue Histogram)
from chromaticity_features import extract_chromaticity_features, get_chromaticity_feature_names
from yuv_features import extract_yuv_features, get_yuv_feature_names

# PHASE 4B FEATURES - Advanced Shape (Hough Transforms)
from hough_features import extract_hough_features, get_hough_feature_names

# PHASE 5A FEATURES - Spatial Context (Neighborhood Aggregation)
from spatial_context_features import extract_spatial_context_features, get_spatial_context_feature_names

# PHASE 4C FEATURES - Advanced Texture (Laws, Fourier)
from laws_texture_features import extract_laws_texture_features, get_laws_texture_feature_names
from fourier_texture_features import extract_fourier_texture_features, get_fourier_texture_feature_names
from bg_removal import apply_mask, best_cricket_mask


def extract_features_from_cell(cell_rgb: np.ndarray, na_hog_vector=None) -> np.ndarray:
    """
    Extract all 406 features from a single RGB cell (PHASE 4C - Advanced Texture).

    Feature breakdown:
        ORIGINAL FEATURES (236):
        - RGB Histogram: 14 features
        - HSV Color: 26 features (14 original + 12 fine hue bins - PHASE 4A ENHANCED)
        - LBP: 16 features (10 original + 6 multi-scale - PHASE 4C ENHANCED)
        - HOG: 144 features (REDUCED from 576 to prevent PCA dominance)
        - Gabor: 20 features
        - GLCM: 13 features (10 original + 3 multi-distance - PHASE 4C ENHANCED)

        PHASE 1 ADDITIONS (65):
        - Enhanced Color Features: 18 features (red detection for ball)
        - Shape Features: 35 features (circularity, aspect ratio + 11 PHASE 4B enhancements)
        - Edge Features: 12 features (Canny, Sobel gradients)

        PHASE 2 ADDITIONS (36):
        - Morphological Features: 8 features (H-Maxima/H-Minima for ball/stump peaks)
        - SIFT Features: 14 features (keypoint detection for objects)
        - Enhanced Texture Features: 14 features (Haralick for bat wood texture)

        PHASE 3 ADDITIONS (30):
        - SLIC Superpixel Features: 12 features (region-based segmentation)
        - Haar-like Features: 10 features (rectangle patterns for edges/lines)
        - Advanced Color Moments: 8 features (skewness, kurtosis, YCrCb)

        PHASE 4A ADDITIONS (20):
        - Chromaticity Features: 14 features (RG chromaticity, Excess Red, Opponent color)
        - YUV Features: 6 features (YUV color space chrominance)

        PHASE 4B ADDITIONS (7):
        - Hough Features: 7 features (circle + line detection for ball/bat/stump)

        PHASE 4C ADDITIONS (15):
        - Laws' Texture Energy: 9 features (texture pattern detection for bat wood grain)
        - Fourier Power Spectrum: 6 features (frequency domain texture analysis)

        Total: 406 per-cell features (236 + 65 + 36 + 30 + 20 Phase4A + 7 Phase4B + 15 Phase4C)
        Note: +25 spatial context features added in extract_features_from_image() = 431 total

    Args:
        cell_rgb: RGB image cell of shape (H, W, 3)
                 Expected size: 75x100 pixels

    Returns:
        Array of 406 per-cell features
    """
    # mask = best_cricket_mask(cell_rgb)
    # cell_masked = apply_mask(cell_rgb, mask)
    # ========== COLOR FEATURES (40 total - original 28 + 12 HSV bins) ==========

    # 1. RGB Histogram Features (14 features)
    rgb_features = extract_rgb_histogram_features(cell_rgb)
    rgb_hist_features = extract_rgb_hist_features(cell_rgb)

    # 2. HSV Color Features (26 features - PHASE 4A ENHANCED from 14)
    # Now includes 12 fine hue bins for improved red ball detection
    cell_hsv = rgb_to_hsv(cell_rgb)
    hsv_features = extract_hsv_color_features(cell_hsv)
    hsv_hist_features = extract_hsv_hist_color_features(cell_hsv, no_of_bins=8)

    lab_color_features = extract_lab_color_features(cell_rgb)
    # ========== TEXTURE FEATURES (60 total) ==========

    # Convert to grayscale for texture features
    cell_gray = cv2.cvtColor(cell_rgb, cv2.COLOR_BGR2GRAY)
    # 3. Local Binary Patterns (16 features - PHASE 4C: 10 original + 6 multi-scale)
    lbp_features = extract_lbp_features(cell_gray)

    # 4. Histogram of Oriented Gradients (144 features)
    hog_features = extract_hog_features(cell_gray)
    # na_hog_features = na_hog_vector
    # print(hog_features.shape)

    circularity_features = extract_circularity_features(cell_gray)
    edge_density_features = extract_edge_density_features(cell_gray)
    # smoothness_features = extract_smoothness_features(cell_gray)
    polygon_features = extract_polygon_features(cell_gray)


    # if na_hog_vector is None or len(na_hog_vector) == 0:
    #     raise ValueError("Neighborhood HOG failed: na_hog_vector is empty!")
    # na_hog_features = na_hog_vector.astype(np.float32)

    # 5. Gabor Filters (20 features)
    # gabor_features = extract_gabor_features(cell_gray)

    # 6. GLCM (13 features - PHASE 4C: 10 original + 3 multi-distance)
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
    # sift_features = extract_sift_features(cell_rgb) abhi

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

    # ========== PHASE 4A FEATURES (20 total) ==========

    # 16. Chromaticity Features (14 features)
    # Critical for ball: RG chromaticity (intensity-invariant), Excess Red Index, Opponent color
    chromaticity_features = extract_chromaticity_features(cell_rgb)

    # 17. YUV Features (6 features)
    # Critical for ball: YUV V-channel (red-cyan chrominance) complements YCrCb
    yuv_features = extract_yuv_features(cell_rgb)

    # Note: HSV fine hue histogram (+12 features) already integrated above in hsv_features

    # ========== PHASE 4B FEATURES (7 total) ==========

    # 18. Hough Features (7 features)
    # Critical for ball: Hough circle detection (direct circular object identification)
    # Critical for bat/stump: Hough line detection (horizontal/vertical line patterns)
    hough_features = extract_hough_features(cell_rgb)

    # Note: Shape features enhanced (+11 features) already integrated above in shape_features (now 35 total)

    # ========== PHASE 4C FEATURES (15 total) ==========

    # 19. Laws' Texture Energy (9 features)
    # Critical for bat: L3E3 (horizontal wood grain edges), E3E3 (all edges)
    # Grass: low all (no structured patterns)
    laws_features = extract_laws_texture_features(cell_gray)

    # 20. Fourier Power Spectrum (6 features)
    # Critical for bat: High peak ratio (periodic wood grain), low entropy (structured)
    # Grass: Low peak ratio (random), high entropy (unstructured)
    fourier_features = extract_fourier_texture_features(cell_gray)

    # Note: LBP enhanced (+6 features) and GLCM enhanced (+3 features) already integrated above

    # ========== COMBINE ALL FEATURES ==========
    # all_features = np.concatenate([
    #     # Original features (236 - includes +12 HSV, +6 LBP, +3 GLCM from Phase 4C)
    #     rgb_features,      # 14
    #     hsv_features,      # 26 (PHASE 4A: 14 original + 12 fine hue bins)
    #     lbp_features,      # 16 (PHASE 4C: 10 original + 6 multi-scale)
    #     hog_features,      # 144 (REDUCED from 576)
    #     gabor_features,    # 20
    #     glcm_features,     # 13 (PHASE 4C: 10 original + 3 multi-distance)

        
    # ])

    # Verify we have exactly 406 per-cell features (236 + 65 + 36 + 30 + 20 + 7 + 15)
    # hu_moment_features = extract_hu_moment_features(cell_gray)
    # sift_features = extract_sift_features(cell_gray)
    # sift_features = extract_sift_features(cell_gray, max_features=5)  # limit to first 5 keypoints
    # orb_features = extract_orb_features(cell_gray, max_features=10)  # limit to first 10 keypoints

    

    # ========== COMBINE ALL FEATURES ==========
    all_features = np.concatenate([
        rgb_features,      # 14
        rgb_hist_features, # 768
        hsv_features,      # 14
        hsv_hist_features, # 768
        lbp_features,      # 10
        hog_features,      # 576
        # na_hog_features,   # 2700
        # gabor_features,    # 20
        glcm_features,     # 12
        # hu_moment_features, # 7
        # sift_features,      # 640
        #orb_features,      # 320

        lab_color_features, # 3
        circularity_features, # 1
        edge_density_features, # 3
        # smoothness_features, # 1
        polygon_features,  # 1

        # Phase 1 features (65 - includes +11 Shape from Phase 4B)
        enhanced_color_features,  # 18
        shape_features,           # 35 (PHASE 4B: 24 original + 11 enhancements)
        edge_features,            # 13

        # Phase 2 features (36)
        morphological_features,   # 8
        # sift_features,            # 14
        enhanced_texture_features,# 14

        # Phase 3 features (30)
        superpixel_features,      # 12
        haar_features,            # 10
        advanced_color_moments,   # 8

        # Phase 4A features (20)
        chromaticity_features,    # 14
        yuv_features,             # 6

        # Phase 4B features (7)
        hough_features,           # 7

        # Phase 4C features (15)
        laws_features,            # 9
        fourier_features          # 6
    ])
    # print("SIFT Size:", len(sift_features))
    # print("ORB Size:", len(orb_features))
    # print("NA-HOG Size:", len(hog_features))
    # print(f"Extracted {len(all_features)} features from cell.")
    # Verify we have exactly 88 features
    # assert len(all_features) == 88, f"Expected 88 features, got {len(all_features)}"
    # assert len(all_features) == 842, f"Expected 842 features, got {len(all_features)}"
    # assert len(all_features) == 1605, f"Expected 1605 features, got {len(all_features)}"
    # assert len(all_features) == 2176, f"Expected 2176 features, got {len(all_features)}" # remove few features add new feature8/12
    assert len(all_features) ==  1100, f"Expected  1100 features, got {len(all_features)}" # remove few features add new feature8/12
    # assert len(all_features) == 2169, f"Expected 2169 features, got {len(all_features)}" # remove few features 8/12
    # assert len(all_features) == 4445, f"Expected 4445 features, got {len(all_features)}" # with normal hog all features
    # assert len(all_features) == 3121, f"Expected 3121 features, got {len(all_features)}" # with normal hog all features
    # assert len(all_features) == 2565, f"Expected 2565 features, got {len(all_features)}" # with normal hog
    # assert len(all_features) == 31057, f"Expected 31057 features, got {len(all_features)}" # with na normal hog
    # assert len(all_features) == 17989, f"Expected 17989 features, got {len(all_features)}"
    # assert len(all_features) == 856, f"Expected 856 features, got {len(all_features)}"

    return all_features.astype(np.float32)


def get_all_feature_names() -> List[str]:
    """
    Get names of all 406 per-cell features (PHASE 4C - Advanced Texture).

    Note: Spatial context features (+25) are added separately in extract_features_from_image(),
    bringing total to 431 features per cell in final output.

    Returns:
        List of 406 per-cell feature names
    """
    all_names = []

    all_names.extend(get_rgb_feature_names())    # 14
    all_names.extend(get_rgb_hist_feature_names())    # 768
    all_names.extend(get_hsv_feature_names())    # 14
    all_names.extend(get_hsv_hist_feature_names(no_of_bins=8))  # 768
    all_names.extend(get_lbp_feature_names())    # 10
    all_names.extend(get_hog_feature_names())    # 576
    # all_names.extend(get_na_aware_hog_feature_names(
    #     orientations=6,
    #     pixels_per_cell=(16, 16),
    #     cells_per_block=(1, 1)
    # ))    # 2700
    # all_names.extend(get_gabor_feature_names())  # 20
    all_names.extend(get_glcm_feature_names())   # 12
    # all_names.extend(get_hu_moment_feature_names())   # 7
    all_names.extend(get_lab_color_feature_names())   # 3
    all_names.extend(get_circularity_feature_names())   # 1
    all_names.extend(get_edge_density_feature_names())   # 3
    # all_names.extend(get_smoothness_feature_names())   # 1
    all_names.extend(get_polygon_feature_names())   # 1
    # all_names.extend(get_sift_feature_names(max_features=5))   # 640
    # all_names.extend(get_orb_feature_names(max_features=10))   # 320
    # all_names.extend(get_sift_feature_names())   # 16384

    # Phase 1 features (65 - includes +11 Shape from Phase 4B)
    all_names.extend(get_enhanced_color_feature_names())  # 18
    all_names.extend(get_shape_feature_names())           # 35 (PHASE 4B: 24 original + 11 enhancements)
    all_names.extend(get_edge_feature_names())            # 12

    # Phase 2 features (36)
    all_names.extend(get_morphological_feature_names())   # 8
    # all_names.extend(get_sift_feature_names())            # 14 abhi
    all_names.extend(get_enhanced_texture_feature_names()) # 14

    # Phase 3 features (30)
    all_names.extend(get_superpixel_feature_names())      # 12
    all_names.extend(get_haar_feature_names())            # 10
    all_names.extend(get_advanced_color_moment_names())   # 8

    # Phase 4A features (20)
    all_names.extend(get_chromaticity_feature_names())    # 14
    all_names.extend(get_yuv_feature_names())             # 6

    # Phase 4B features (7)
    all_names.extend(get_hough_feature_names())           # 7

    # Phase 4C features (15)
    all_names.extend(get_laws_texture_feature_names())    # 9
    all_names.extend(get_fourier_texture_feature_names()) # 6

    # all_names.extend(get_spatial_context_feature_names()) # 25

    # print(f"Extracted {len(all_names)} feature names from cell.")
    # assert len(all_names) == 88, f"Expected 88 feature names, got {len(all_names)}"
    # assert len(all_names) == 1605, f"Expected 1605 feature names, got {len(all_names)}"
    # assert len(all_names) == 2245, f"Expected 2245 feature names, got {len(all_names)}"
    assert len(all_names) ==  1100, f"Expected  1100 features, got {len(all_names)}" # remove few features add new feature8/12
    # assert len(all_names) == 2176, f"Expected 2176 features, got {len(all_names)}" # remove few features add new feature8/12
    # assert len(all_names) == 2169, f"Expected 2169 features, got {len(all_names)}" # remove few features 8/12
    # assert len(all_names) == 4445, f"Expected 4445 features, got {len(all_names)}" # with normal hog all features
    # assert len(all_names) == 3121, f"Expected 3121 features, got {len(all_names)}" # with normal hog all features
    # assert len(all_names) == 2565, f"Expected 2565 feature names, got {len(all_names)}" # with normal hog
    # assert len(all_names) == 31057, f"Expected 31057 feature names, got {len(all_names)}" # with na normal hog
    # assert len(all_names) == 17989, f"Expected 17989 feature names, got {len(all_names)}"
    # assert len(all_names) == 856, f"Expected 856 feature names, got {len(all_names)}"

    return all_names

def extract_hog_per_cell(cells: List[np.ndarray]) -> List[np.ndarray]:
    """
    Compute HOG for each cell in the 8x8 grid.
    Returns list of 64 HOG vectors.
    """
    # Convert each of 64 cells to grayscale
    gray_cells = [cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) for cell in cells]

    # 1️⃣ Compute normal HOG for each cell
    hog_list = compute_hog_per_cell(gray_cells)

    # 2️⃣ Compute neighborhood-aware HOG for each of 64 cells
    hog_na_list = extract_na_aware_hog_for_grid_features(hog_list)

    for idx, v in enumerate(hog_na_list):
        # print(f"NA-HOG vector length for cell {idx}: {len(v)}")
        if v is None or len(v) == 0:
            print(f"[ERROR] NA-HOG failed for cell index {idx}")
            print(f"Original HOG length: {len(hog_list[idx])}")
            raise ValueError("Invalid NA-HOG vector")
        
    return hog_na_list

def extract_features_from_image(image: np.ndarray, cells: List[np.ndarray] = None) -> np.ndarray:
    """
    Extract features from all cells in an image.

    FINAL FEATURE COUNT (Phase 4C + 5A): After extracting per-cell features (406),
    spatial context features (25) are computed from neighboring cells,
    resulting in 431 total features per cell.

    Args:
        image: Full image of shape (600, 800, 3)
        cells: Pre-divided cells (optional). If None, will divide the image.

    Returns:
        Array of shape (64, 431) containing features for all cells
        (406 per-cell + 25 spatial context features) - PHASE 4C + 5A COMPLETE
    """
    # Import here to avoid circular imports
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_prep'))
    from grid_utils import divide_into_grid

    if cells is None:
        cells = divide_into_grid(image, grid_size=(8, 8))
    
    # # Convert each of 64 cells to grayscale
    # gray_cells = [cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) for cell in cells]

    # # 1️⃣ Compute normal HOG for each cell
    # hog_list = compute_hog_per_cell(gray_cells)

    # 2️⃣ Compute neighborhood-aware HOG for each of 64 cells
    hog_na_list = extract_hog_per_cell(cells)

    # for idx, v in enumerate(hog_na_list):
    #     print(f"NA-HOG vector length for cell {idx}: {len(v)}")
    #     if v is None or len(v) == 0:
    #         print(f"[ERROR] NA-HOG failed for cell index {idx}")
    #         print(f"Original HOG length: {len(hog_list[idx])}")
    #         raise ValueError("Invalid NA-HOG vector")

    

    # ========== STEP 1: Extract per-cell features (382 features) ==========
    per_cell_features_list = []
    # for cell in cells:
        # features = extract_features_from_cell(cell)
        # features_list.append(features)
    for idx, cell in enumerate(cells):
        print(f"Extracting features from cell {idx+1}/64")
        print(hog_na_list[idx].shape())
        features = extract_features_from_cell(
            cell_rgb=cell,
            na_hog_vector=hog_na_list[idx]      # inject NA-HOG here
        )
        per_cell_features_list.append(features)

    per_cell_features = np.array(per_cell_features_list, dtype=np.float32)

    # ========== STEP 2: Extract spatial context features (25 features) ==========
    # Spatial context requires ALL cell features to be extracted first
    spatial_features_list = []
    for cell_idx in range(len(cells)):
        spatial_features = extract_spatial_context_features(
            per_cell_features,
            cell_idx,
            grid_shape=(8, 8)
        )
        spatial_features_list.append(spatial_features)

    spatial_features = np.array(spatial_features_list, dtype=np.float32)

    # ========== STEP 3: Concatenate per-cell + spatial features ==========
    # Resulting shape: (64, 407) = (64, 382 + 25)
    all_features = np.concatenate([per_cell_features, spatial_features], axis=1)

    return all_features


def extract_features_from_images(images: List[np.ndarray], verbose: bool = True) -> np.ndarray:
    """
    Extract features from multiple images.

    Args:
        images: List of images, each of shape (600, 800, 3)
        verbose: Print progress if True

    Returns:
        Array of shape (num_images * 64, 431) - PHASE 4C + 5A COMPLETE: 431 features per cell
        (406 per-cell: Phase 4A/B/C + 25 spatial context: Phase 5A)
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
