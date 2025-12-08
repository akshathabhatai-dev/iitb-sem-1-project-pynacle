"""
Morphological Features for Cricket Object Detection

This module extracts 8 H-Maxima and H-Minima features for detecting regional
intensity extrema in the image.

Features breakdown:
- H-Maxima (4): Detect bright peaks - critical for ball (red peak), stump (white peak)
- H-Minima (4): Detect dark valleys - shadows, background

Critical for:
- Ball: Red channel has bright peak (high intensity red region)
- Stump: Grayscale has bright peak (white wickets)
- Background discrimination: Uniform regions have few extrema

Author: Pynacle Team
Date: 2025
"""

import numpy as np
import cv2
from typing import List
from skimage.morphology import h_maxima, h_minima, reconstruction


def extract_morphological_features(cell_rgb: np.ndarray) -> np.ndarray:
    """
    Extract 8 morphological features (H-Maxima and H-Minima) from RGB cell.

    H-Maxima detects regional intensity peaks (bright spots):
    - Ball: Red channel should have bright red peak
    - Stump: Grayscale should have bright white peak

    H-Minima detects regional intensity valleys (dark spots):
    - Shadows, background, low-intensity regions

    Args:
        cell_rgb: RGB image cell of shape (H, W, 3) in BGR format
                 Expected size: 75x100 pixels

    Returns:
        Array of 8 float32 features
    """
    # Handle empty or invalid images
    if cell_rgb.size == 0:
        return np.zeros(8, dtype=np.float32)

    # Convert to grayscale for general H-Maxima/H-Minima
    cell_gray = cv2.cvtColor(cell_rgb, cv2.COLOR_BGR2GRAY)

    # Extract red channel for ball detection (BGR format: channel 2 is red)
    red_channel = cell_rgb[:, :, 2].astype(np.uint8)

    # H-parameter for morphological operations
    # h=20 means we detect peaks that are at least 20 intensity units above surroundings
    h_threshold = 20

    # Get total pixel count for normalization
    total_pixels = cell_gray.shape[0] * cell_gray.shape[1]

    # ========== H-MAXIMA FEATURES (4 features) ==========
    # Detect regional maxima (peaks) in grayscale and red channel

    try:
        # H-Maxima on grayscale (detects white/bright objects like stumps)
        maxima_gray = h_maxima(cell_gray, h=h_threshold)

        # H-Maxima on red channel (detects red peaks like cricket ball)
        maxima_red = h_maxima(red_channel, h=h_threshold)

        # 1. H-Maxima count (number of peak regions)
        # Ball/stump should have 1-2 peaks, background has 0-1
        num_maxima_gray = np.count_nonzero(maxima_gray)
        num_maxima_red = np.count_nonzero(maxima_red)
        h_maxima_count = float(num_maxima_gray + num_maxima_red) / 2.0

        # 2. H-Maxima mean intensity (average intensity at peak locations)
        # Ball has high red intensity, stump has high grayscale intensity
        if num_maxima_gray > 0:
            maxima_gray_intensity = np.mean(cell_gray[maxima_gray > 0])
        else:
            maxima_gray_intensity = 0.0

        if num_maxima_red > 0:
            maxima_red_intensity = np.mean(red_channel[maxima_red > 0])
        else:
            maxima_red_intensity = 0.0

        h_maxima_mean_intensity = (maxima_gray_intensity + maxima_red_intensity) / 2.0

        # 3. H-Maxima max intensity (peak of the peak)
        # Maximum intensity value at the brightest maxima
        if num_maxima_gray > 0:
            maxima_gray_max = np.max(cell_gray[maxima_gray > 0])
        else:
            maxima_gray_max = 0.0

        if num_maxima_red > 0:
            maxima_red_max = np.max(red_channel[maxima_red > 0])
        else:
            maxima_red_max = 0.0

        h_maxima_max_intensity = max(maxima_gray_max, maxima_red_max)

        # 4. H-Maxima area ratio (percentage of pixels that are maxima)
        # Ball/stump should have small focused peak area
        h_maxima_area_ratio = float(num_maxima_gray + num_maxima_red) / (2.0 * total_pixels)

    except Exception as e:
        # If H-Maxima fails (rare), return zeros
        print(f"H-Maxima extraction failed: {e}")
        h_maxima_count = 0.0
        h_maxima_mean_intensity = 0.0
        h_maxima_max_intensity = 0.0
        h_maxima_area_ratio = 0.0

    # ========== H-MINIMA FEATURES (4 features) ==========
    # Detect regional minima (valleys/dark spots)

    try:
        # H-Minima on grayscale (detects dark regions, shadows)
        minima_gray = h_minima(cell_gray, h=h_threshold)

        # H-Minima on red channel (detects non-red regions)
        minima_red = h_minima(red_channel, h=h_threshold)

        # 5. H-Minima count (number of valley regions)
        # Background/grass may have multiple minima, objects have fewer
        num_minima_gray = np.count_nonzero(minima_gray)
        num_minima_red = np.count_nonzero(minima_red)
        h_minima_count = float(num_minima_gray + num_minima_red) / 2.0

        # 6. H-Minima mean intensity (average intensity at valley locations)
        # Dark shadows have low intensity
        if num_minima_gray > 0:
            minima_gray_intensity = np.mean(cell_gray[minima_gray > 0])
        else:
            minima_gray_intensity = 0.0

        if num_minima_red > 0:
            minima_red_intensity = np.mean(red_channel[minima_red > 0])
        else:
            minima_red_intensity = 0.0

        h_minima_mean_intensity = (minima_gray_intensity + minima_red_intensity) / 2.0

        # 7. H-Minima min intensity (darkest valley point)
        if num_minima_gray > 0:
            minima_gray_min = np.min(cell_gray[minima_gray > 0])
        else:
            minima_gray_min = 0.0

        if num_minima_red > 0:
            minima_red_min = np.min(red_channel[minima_red > 0])
        else:
            minima_red_min = 0.0

        h_minima_min_intensity = min(minima_gray_min, minima_red_min)

        # 8. H-Minima area ratio (percentage of pixels that are minima)
        h_minima_area_ratio = float(num_minima_gray + num_minima_red) / (2.0 * total_pixels)

    except Exception as e:
        # If H-Minima fails (rare), return zeros
        print(f"H-Minima extraction failed: {e}")
        h_minima_count = 0.0
        h_minima_mean_intensity = 0.0
        h_minima_min_intensity = 0.0
        h_minima_area_ratio = 0.0

    # ========== COMBINE ALL FEATURES ==========
    features = np.array([
        # H-Maxima Features (4)
        h_maxima_count,
        h_maxima_mean_intensity,
        h_maxima_max_intensity,
        h_maxima_area_ratio,

        # H-Minima Features (4)
        h_minima_count,
        h_minima_mean_intensity,
        h_minima_min_intensity,
        h_minima_area_ratio
    ], dtype=np.float32)

    # Verify we have exactly 8 features
    assert len(features) == 8, f"Expected 8 features, got {len(features)}"

    # Replace any NaN or Inf values with 0.0 (defensive programming)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def get_morphological_feature_names() -> List[str]:
    """
    Get names of all 8 morphological features.

    Returns:
        List of 8 feature names
    """
    names = [
        # H-Maxima Features (4)
        'h_maxima_count',
        'h_maxima_mean_intensity',
        'h_maxima_max_intensity',
        'h_maxima_area_ratio',

        # H-Minima Features (4)
        'h_minima_count',
        'h_minima_mean_intensity',
        'h_minima_min_intensity',
        'h_minima_area_ratio'
    ]

    assert len(names) == 8, f"Expected 8 feature names, got {len(names)}"

    return names


if __name__ == '__main__':
    # Test feature extraction
    print("Testing Morphological Features (H-Maxima/H-Minima)...")

    # Create a dummy cell with bright red spot (simulating ball)
    dummy_cell_ball = np.ones((75, 100, 3), dtype=np.uint8) * 100  # Medium gray background
    # Add bright red circular spot in center
    cv2.circle(dummy_cell_ball, (50, 37), 15, (100, 100, 255), -1)  # BGR: bright red

    # Create a dummy cell with bright white vertical stripe (simulating stump)
    dummy_cell_stump = np.ones((75, 100, 3), dtype=np.uint8) * 100  # Medium gray background
    cv2.rectangle(dummy_cell_stump, (45, 10), (55, 65), (255, 255, 255), -1)  # White vertical stripe

    # Create a dummy cell with uniform color (simulating grass background)
    dummy_cell_grass = np.ones((75, 100, 3), dtype=np.uint8) * 120  # Uniform medium intensity

    # Extract features
    features_ball = extract_morphological_features(dummy_cell_ball)
    features_stump = extract_morphological_features(dummy_cell_stump)
    features_grass = extract_morphological_features(dummy_cell_grass)

    print(f"\nExtracted {len(features_ball)} features")
    print(f"\nMorphological Feature Comparison:")
    print(f"{'Feature Name':<30} {'Ball':<12} {'Stump':<12} {'Grass':<12}")
    print("=" * 70)

    names = get_morphological_feature_names()
    for i, name in enumerate(names):
        print(f"{name:<30} {features_ball[i]:<12.4f} {features_stump[i]:<12.4f} {features_grass[i]:<12.4f}")

    print("\n✓ Test passed!")
    print("  Ball: Should have high h_maxima_count and h_maxima_max_intensity (red peak)")
    print("  Stump: Should have high h_maxima_count (white peak)")
    print("  Grass: Should have low h_maxima_count (uniform, no peaks)")
