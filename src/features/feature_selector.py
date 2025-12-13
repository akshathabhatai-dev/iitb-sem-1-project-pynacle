"""
Feature selection module for binary models.

Provides selective feature subsets for each binary classifier (ball, bat, stump) to reduce
dimensionality and improve performance by focusing on semantically relevant features.

FEATURE GROUP INDICES (Total: 2343 features):
    RGB Hist            : [   0,  768) - 768 features
    HSV Hist            : [ 768, 1536) - 768 features
    LBP                 : [1536, 1552) -  16 features
    HOG                 : [1552, 2128) - 576 features
    GLCM                : [2128, 2143) -  15 features
    Hu Moments          : [2143, 2150) -   7 features
    Enhanced Color      : [2150, 2168) -  18 features
    Shape               : [2168, 2203) -  35 features
    Edge                : [2203, 2215) -  12 features
    Morphological       : [2215, 2223) -   8 features
    Enhanced Texture    : [2223, 2237) -  14 features
    Superpixel          : [2237, 2249) -  12 features
    Haar                : [2249, 2259) -  10 features
    Adv Color Moments   : [2259, 2267) -   8 features
    Chromaticity        : [2267, 2281) -  14 features
    YUV                 : [2281, 2287) -   6 features
    Hough               : [2287, 2294) -   7 features
    Laws                : [2294, 2303) -   9 features
    Fourier             : [2303, 2309) -   6 features
    Ball Color          : [2309, 2321) -  12 features
    Bat Color           : [2321, 2333) -  12 features
    Stump Color         : [2333, 2343) -  10 features

Author: Claude Code
Date: 2025-12-09
"""

import numpy as np
from typing import List, Dict
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all feature name functions
from rgb_hist_features import get_rgb_hist_feature_names
from hsv_hist_features import get_hsv_hist_color_feature_names as get_hsv_hist_feature_names
from lbp_features import get_lbp_feature_names
from hog_features import get_hog_feature_names
from glcm_features import get_glcm_feature_names
from hu_moment_features import get_hu_moment_feature_names
from enhanced_color_features import get_enhanced_color_feature_names
from shape_features import get_shape_feature_names
from edge_features import get_edge_feature_names
from morphological_features import get_morphological_feature_names
from enhanced_texture_features import get_enhanced_texture_feature_names
from superpixel_features import get_superpixel_feature_names
from haar_features import get_haar_feature_names
from advanced_color_moments import get_advanced_color_moment_names
from chromaticity_features import get_chromaticity_feature_names
from yuv_features import get_yuv_feature_names
from hough_features import get_hough_feature_names
from laws_texture_features import get_laws_texture_feature_names
from fourier_texture_features import get_fourier_texture_feature_names
from ball_color_features import get_ball_color_feature_names
from bat_color_features import get_bat_color_feature_names
from stump_color_features import get_stump_color_feature_names


# Feature group boundaries (verified from feature extraction order)
FEATURE_GROUPS = {
    'rgb_hist': (0, 768),
    'hsv_hist': (768, 1536),
    'lbp': (1536, 1552),
    'hog': (1552, 2128),
    'glcm': (2128, 2143),
    'hu_moments': (2143, 2150),
    'enhanced_color': (2150, 2168),
    'shape': (2168, 2203),
    'edge': (2203, 2215),
    'morphological': (2215, 2223),
    'enhanced_texture': (2223, 2237),
    'superpixel': (2237, 2249),
    'haar': (2249, 2259),
    'adv_color_moments': (2259, 2267),
    'chromaticity': (2267, 2281),
    'yuv': (2281, 2287),
    'hough': (2287, 2294),
    'laws': (2294, 2303),
    'fourier': (2303, 2309),
    # 'ball_color': (2309, 2321),
    # 'bat_color': (2321, 2333),
    # 'stump_color': (2333, 2343)
}


def get_all_feature_names() -> List[str]:
    """
    Get complete list of 2343 feature names in extraction order.

    This must match the exact order in feature_extractor.py:extract_features_from_cell()

    Returns:
        List of 2343 feature names
    """
    all_names = []

    # Base color features (1536)
    all_names.extend(get_rgb_hist_feature_names())    # 768
    all_names.extend(get_hsv_hist_feature_names())    # 768

    # Texture features (607)
    all_names.extend(get_lbp_feature_names())         # 16
    all_names.extend(get_hog_feature_names())         # 576
    all_names.extend(get_glcm_feature_names())        # 15

    # Shape features (7)
    all_names.extend(get_hu_moment_feature_names())   # 7

    # Phase 1 features (65)
    all_names.extend(get_enhanced_color_feature_names())  # 18
    all_names.extend(get_shape_feature_names())           # 35
    all_names.extend(get_edge_feature_names())            # 12

    # Phase 2 features (22)
    all_names.extend(get_morphological_feature_names())   # 8
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

    # Phase 5B features (34)
    # all_names.extend(get_ball_color_feature_names())      # 12
    # all_names.extend(get_bat_color_feature_names())       # 12
    # all_names.extend(get_stump_color_feature_names())     # 10

    assert len(all_names) == 2343, f"Expected 2343 features, got {len(all_names)}"

    return all_names


def get_ball_feature_indices() -> List[int]:
    """
    Return feature indices for ball model (~420 features).

    BALL-SPECIFIC STRATEGY:
    - Universal features: HOG (edges), Shape, LBP, GLCM, Edge, Hu Moments, Superpixel
    - Ball-specific features: All red color detection features
    - Reduced histograms: Every 5th bin (768→154 per histogram)

    Returns:
        List of feature indices for ball model
    """
    indices = []

    # ========== REDUCED HISTOGRAMS (308 features) ==========
    # RGB Histogram: Every 5th bin (768 -> 154 bins)
    rgb_start, rgb_end = FEATURE_GROUPS['rgb_hist']
    indices.extend(range(rgb_start, rgb_end, 5))

    # HSV Histogram: Every 5th bin (768 -> 154 bins)
    hsv_start, hsv_end = FEATURE_GROUPS['hsv_hist']
    indices.extend(range(hsv_start, hsv_end, 5))

    # ========== UNIVERSAL TEXTURE FEATURES (607 features) ==========
    # LBP: All 16 features
    lbp_start, lbp_end = FEATURE_GROUPS['lbp']
    indices.extend(range(lbp_start, lbp_end))

    # HOG: All 576 features - CRITICAL for edge detection
    hog_start, hog_end = FEATURE_GROUPS['hog']
    indices.extend(range(hog_start, hog_end))

    # GLCM: All 15 features
    glcm_start, glcm_end = FEATURE_GROUPS['glcm']
    indices.extend(range(glcm_start, glcm_end))

    # ========== UNIVERSAL SHAPE FEATURES (54 features) ==========
    # Hu Moments: All 7 features
    hu_start, hu_end = FEATURE_GROUPS['hu_moments']
    indices.extend(range(hu_start, hu_end))

    # Shape Features: All 35 features (circularity, Hough circle, compactness)
    shape_start, shape_end = FEATURE_GROUPS['shape']
    indices.extend(range(shape_start, shape_end))

    # Edge Features: All 12 features
    edge_start, edge_end = FEATURE_GROUPS['edge']
    indices.extend(range(edge_start, edge_end))

    # ========== UNIVERSAL REGION FEATURES (12 features) ==========
    # Superpixel Features: All 12 features
    superpixel_start, superpixel_end = FEATURE_GROUPS['superpixel']
    indices.extend(range(superpixel_start, superpixel_end))

    # ========== BALL-SPECIFIC COLOR FEATURES (82 features) ==========
    # Enhanced Color Features: All 18 features (red dominance, LAB a*)
    enhanced_color_start, enhanced_color_end = FEATURE_GROUPS['enhanced_color']
    indices.extend(range(enhanced_color_start, enhanced_color_end))

    # Morphological Features: All 8 features (red peak detection)
    morphological_start, morphological_end = FEATURE_GROUPS['morphological']
    indices.extend(range(morphological_start, morphological_end))

    # Advanced Color Moments: All 8 features (red skewness/kurtosis)
    adv_color_start, adv_color_end = FEATURE_GROUPS['adv_color_moments']
    indices.extend(range(adv_color_start, adv_color_end))

    # Chromaticity Features: All 14 features (Excess Red, RG chromaticity)
    chromaticity_start, chromaticity_end = FEATURE_GROUPS['chromaticity']
    indices.extend(range(chromaticity_start, chromaticity_end))

    # YUV Features: All 6 features (V-channel red-cyan)
    yuv_start, yuv_end = FEATURE_GROUPS['yuv']
    indices.extend(range(yuv_start, yuv_end))

    # Hough Features: First 4 features (circle detection), skip line detection
    hough_start, hough_end = FEATURE_GROUPS['hough']
    indices.extend(range(hough_start, hough_start + 4))  # Only circle features

    # Ball Color Features: All 12 features
    # ball_color_start, ball_color_end = FEATURE_GROUPS['ball_color']
    # indices.extend(range(ball_color_start, ball_color_end))

    # ========== TOTAL: 1079 features ==========
    indices = sorted(set(indices))

    print(f"Ball model: Selected {len(indices)} features out of 2343")

    return indices


def get_bat_feature_indices() -> List[int]:
    """
    Return feature indices for bat model (~380 features).

    BAT-SPECIFIC STRATEGY:
    - Universal features: HOG (horizontal edges), Shape, LBP, GLCM, Edge, Hu Moments, Superpixel
    - Bat-specific features: Brown color, wood texture features
    - Reduced histograms: Every 5th bin

    Returns:
        List of feature indices for bat model
    """
    indices = []

    # ========== REDUCED HISTOGRAMS (308 features) ==========
    rgb_start, rgb_end = FEATURE_GROUPS['rgb_hist']
    indices.extend(range(rgb_start, rgb_end, 5))

    hsv_start, hsv_end = FEATURE_GROUPS['hsv_hist']
    indices.extend(range(hsv_start, hsv_end, 5))

    # ========== UNIVERSAL TEXTURE FEATURES (607 features) ==========
    lbp_start, lbp_end = FEATURE_GROUPS['lbp']
    indices.extend(range(lbp_start, lbp_end))

    hog_start, hog_end = FEATURE_GROUPS['hog']
    indices.extend(range(hog_start, hog_end))

    glcm_start, glcm_end = FEATURE_GROUPS['glcm']
    indices.extend(range(glcm_start, glcm_end))

    # ========== UNIVERSAL SHAPE FEATURES (54 features) ==========
    hu_start, hu_end = FEATURE_GROUPS['hu_moments']
    indices.extend(range(hu_start, hu_end))

    shape_start, shape_end = FEATURE_GROUPS['shape']
    indices.extend(range(shape_start, shape_end))

    edge_start, edge_end = FEATURE_GROUPS['edge']
    indices.extend(range(edge_start, edge_end))

    # ========== UNIVERSAL REGION FEATURES (12 features) ==========
    superpixel_start, superpixel_end = FEATURE_GROUPS['superpixel']
    indices.extend(range(superpixel_start, superpixel_end))

    # ========== BAT-SPECIFIC FEATURES (47 features) ==========
    # Enhanced Texture Features: All 14 features (Haralick wood texture)
    enhanced_texture_start, enhanced_texture_end = FEATURE_GROUPS['enhanced_texture']
    indices.extend(range(enhanced_texture_start, enhanced_texture_end))

    # Hough Features: Last 3 features (line detection), skip circle
    hough_start, hough_end = FEATURE_GROUPS['hough']
    indices.extend(range(hough_end - 3, hough_end))  # Only line features

    # Laws' Texture Energy: All 9 features (wood grain edges)
    laws_start, laws_end = FEATURE_GROUPS['laws']
    indices.extend(range(laws_start, laws_end))

    # Fourier Power Spectrum: All 6 features (periodic grain pattern)
    fourier_start, fourier_end = FEATURE_GROUPS['fourier']
    indices.extend(range(fourier_start, fourier_end))

    # Bat Color Features: All 12 features
    # bat_color_start, bat_color_end = FEATURE_GROUPS['bat_color']
    # indices.extend(range(bat_color_start, bat_color_end))

    # ========== TOTAL: 1025 features ==========
    indices = sorted(set(indices))

    print(f"Bat model: Selected {len(indices)} features out of 2343")

    return indices


def get_stump_feature_indices() -> List[int]:
    """
    Return feature indices for stump model (~360 features).

    STUMP-SPECIFIC STRATEGY:
    - Universal features: HOG (vertical edges), Shape, LBP, GLCM, Edge, Hu Moments, Superpixel
    - Stump-specific features: White/yellow color, vertical patterns
    - Reduced histograms: Every 5th bin

    Returns:
        List of feature indices for stump model
    """
    indices = []

    # ========== REDUCED HISTOGRAMS (308 features) ==========
    rgb_start, rgb_end = FEATURE_GROUPS['rgb_hist']
    indices.extend(range(rgb_start, rgb_end, 5))

    hsv_start, hsv_end = FEATURE_GROUPS['hsv_hist']
    indices.extend(range(hsv_start, hsv_end, 5))

    # ========== UNIVERSAL TEXTURE FEATURES (607 features) ==========
    lbp_start, lbp_end = FEATURE_GROUPS['lbp']
    indices.extend(range(lbp_start, lbp_end))

    hog_start, hog_end = FEATURE_GROUPS['hog']
    indices.extend(range(hog_start, hog_end))

    glcm_start, glcm_end = FEATURE_GROUPS['glcm']
    indices.extend(range(glcm_start, glcm_end))

    # ========== UNIVERSAL SHAPE FEATURES (54 features) ==========
    hu_start, hu_end = FEATURE_GROUPS['hu_moments']
    indices.extend(range(hu_start, hu_end))

    shape_start, shape_end = FEATURE_GROUPS['shape']
    indices.extend(range(shape_start, shape_end))

    edge_start, edge_end = FEATURE_GROUPS['edge']
    indices.extend(range(edge_start, edge_end))

    # ========== UNIVERSAL REGION FEATURES (12 features) ==========
    superpixel_start, superpixel_end = FEATURE_GROUPS['superpixel']
    indices.extend(range(superpixel_start, superpixel_end))

    # ========== STUMP-SPECIFIC FEATURES (33 features) ==========
    # Morphological Features: All 8 features (white peak detection)
    morphological_start, morphological_end = FEATURE_GROUPS['morphological']
    indices.extend(range(morphological_start, morphological_end))

    # Haar-like Features: All 10 features (vertical patterns)
    haar_start, haar_end = FEATURE_GROUPS['haar']
    indices.extend(range(haar_start, haar_end))

    # Hough Features: Last 3 features (line detection)
    hough_start, hough_end = FEATURE_GROUPS['hough']
    indices.extend(range(hough_end - 3, hough_end))

    # Stump Color Features: All 10 features
    # stump_color_start, stump_color_end = FEATURE_GROUPS['stump_color']
    # indices.extend(range(stump_color_start, stump_color_end))

    # ========== TOTAL: 1012 features ==========
    indices = sorted(set(indices))

    print(f"Stump model: Selected {len(indices)} features out of 2343")

    return indices


def get_feature_subsets() -> Dict[str, List[int]]:
    """
    Get all feature subsets for binary models.

    Returns:
        Dictionary mapping model names to feature indices
    """
    return {
        'ball': get_ball_feature_indices(),
        'bat': get_bat_feature_indices(),
        'stump': get_stump_feature_indices()
    }


def validate_feature_indices():
    """
    Validate that all feature indices are within bounds and don't overlap incorrectly.
    """
    all_names = get_all_feature_names()
    total_features = len(all_names)

    subsets = get_feature_subsets()

    print("\n" + "="*70)
    print("FEATURE SELECTION VALIDATION")
    print("="*70)

    for model_name, indices in subsets.items():
        # Check all indices are valid
        assert all(0 <= idx < total_features for idx in indices), \
            f"{model_name}: Some indices out of bounds [0, {total_features})"

        # Check no duplicates
        assert len(indices) == len(set(indices)), \
            f"{model_name}: Duplicate indices found"

        # Print summary
        print(f"\n{model_name.upper()} MODEL:")
        print(f"  Total features selected: {len(indices)}")
        print(f"  Reduction: {total_features} -> {len(indices)} " +
              f"({100 * len(indices) / total_features:.1f}%)")
        print(f"  Feature-to-sample ratio: {len(indices)}/100 = " +
              f"{len(indices)/100:.1f}:1 (ideal <10:1)")

        # Show feature name samples
        print(f"  Sample features:")
        sample_indices = indices[:5] + indices[-5:]
        for idx in sample_indices[:10]:
            print(f"    [{idx:4d}] {all_names[idx]}")

    print("\n" + "="*70)
    print("VALIDATION PASSED")
    print("="*70 + "\n")


if __name__ == '__main__':
    # Test feature selection
    print("Testing feature selection module...")

    # Validate all feature selections
    validate_feature_indices()

    # Print detailed breakdown
    all_names = get_all_feature_names()

    print("\nFEATURE BREAKDOWN BY MODEL:")
    print("="*70)

    subsets = get_feature_subsets()

    for model_name, indices in subsets.items():
        print(f"\n{model_name.upper()} MODEL ({len(indices)} features):")
        print("-" * 70)

        # Group features by type
        for group_name, (start, end) in FEATURE_GROUPS.items():
            group_indices = [idx for idx in indices if start <= idx < end]
            if group_indices:
                total_in_group = end - start
                selected_in_group = len(group_indices)
                pct = 100 * selected_in_group / total_in_group if total_in_group > 0 else 0
                print(f"  {group_name:25s}: {selected_in_group:3d}/{total_in_group:3d} " +
                      f"({pct:5.1f}%)")
