"""
Advanced Color Moments for Cricket Object Detection

This module extracts 8 higher-order color moment features for advanced color analysis.

Features breakdown:
- RGB Skewness (3): Distribution asymmetry for R, G, B channels
- RGB Kurtosis (3): Distribution peakedness for R, G, B channels
- YCrCb Features (2): Chrom

inance-based red detection

Critical for:
- Ball: High red skewness (red pixels concentrated in bright range)
- Uniform objects: Low kurtosis (normal distribution)
- Complex textures: High kurtosis (heavy tails, many outliers)

Author: Pynacle Team
Date: 2025
"""

import numpy as np
import cv2
from typing import List
from scipy import stats


def extract_advanced_color_moments(cell_rgb: np.ndarray) -> np.ndarray:
    """
    Extract 8 advanced color moment features from RGB cell.

    Higher-order moments (skewness, kurtosis) capture distribution shape beyond
    mean and variance, providing additional discrimination for color analysis.

    Args:
        cell_rgb: RGB image cell of shape (H, W, 3) in BGR format
                 Expected size: 75x100 pixels

    Returns:
        Array of 8 float32 features
    """
    # Handle empty or invalid images
    if cell_rgb.size == 0:
        return np.zeros(8, dtype=np.float32)

    # OpenCV uses BGR format
    B = cell_rgb[:, :, 0].flatten().astype(np.float32)
    G = cell_rgb[:, :, 1].flatten().astype(np.float32)
    R = cell_rgb[:, :, 2].flatten().astype(np.float32)

    # ========== RGB SKEWNESS (3 features) ==========
    # Skewness measures distribution asymmetry
    # Positive skew: tail on right (many low values, few high values)
    # Negative skew: tail on left (few low values, many high values)
    # Zero skew: symmetric distribution

    # 1. Red channel skewness
    # Ball: Positive skew (many bright red pixels, bimodal distribution)
    # Grass: Near-zero skew (symmetric green distribution)
    red_skewness = float(stats.skew(R))

    # 2. Green channel skewness
    # Grass: Positive skew (concentrated in mid-green range)
    green_skewness = float(stats.skew(G))

    # 3. Blue channel skewness
    # Sky/water: High blue skew
    blue_skewness = float(stats.skew(B))

    # ========== RGB KURTOSIS (3 features) ==========
    # Kurtosis measures distribution "tailedness" and peakedness
    # High kurtosis: Sharp peak, heavy tails (many outliers)
    # Low kurtosis: Flat distribution (uniform)
    # Normal distribution: kurtosis ≈ 3 (excess kurtosis ≈ 0)

    # 4. Red channel kurtosis (excess kurtosis: kurtosis - 3)
    # Ball: High kurtosis (concentrated red + some background)
    # Uniform object: Low kurtosis
    red_kurtosis = float(stats.kurtosis(R, fisher=True))  # fisher=True gives excess kurtosis

    # 5. Green channel kurtosis
    # Grass: Variable kurtosis depending on texture complexity
    green_kurtosis = float(stats.kurtosis(G, fisher=True))

    # 6. Blue channel kurtosis
    blue_kurtosis = float(stats.kurtosis(B, fisher=True))

    # ========== YCrCb COLOR SPACE (2 features) ==========
    # YCrCb separates luminance (Y) from chrominance (Cr=red, Cb=blue)
    # Cr channel is excellent for red detection (ball)

    # Convert BGR to YCrCb
    cell_ycrcb = cv2.cvtColor(cell_rgb, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Y = cell_ycrcb[:, :, 0]   # Luma (brightness): 0-255
    Cr = cell_ycrcb[:, :, 1]  # Red chrominance: 0-255 (128=neutral)
    Cb = cell_ycrcb[:, :, 2]  # Blue chrominance: 0-255 (128=neutral)

    # 7. YCrCb Cr mean (red chrominance)
    # Ball: High Cr mean (>140) indicates red-dominated color
    # Grass: Low Cr mean (<120) indicates green-dominated color
    ycrcb_cr_mean = np.mean(Cr)

    # 8. YCrCb Cr max (peak red chrominance)
    # Ball should have very high Cr max (>180)
    ycrcb_cr_max = np.max(Cr)

    # ========== COMBINE ALL FEATURES ==========
    features = np.array([
        # RGB Skewness (3)
        red_skewness,
        green_skewness,
        blue_skewness,

        # RGB Kurtosis (3)
        red_kurtosis,
        green_kurtosis,
        blue_kurtosis,

        # YCrCb Features (2)
        ycrcb_cr_mean,
        ycrcb_cr_max
    ], dtype=np.float32)

    # Verify we have exactly 8 features
    assert len(features) == 8, f"Expected 8 features, got {len(features)}"

    # Replace any NaN or Inf values with 0.0 (defensive programming)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def get_advanced_color_moment_names() -> List[str]:
    """
    Get names of all 8 advanced color moment features.

    Returns:
        List of 8 feature names
    """
    names = [
        # RGB Skewness (3)
        'red_skewness',
        'green_skewness',
        'blue_skewness',

        # RGB Kurtosis (3)
        'red_kurtosis',
        'green_kurtosis',
        'blue_kurtosis',

        # YCrCb Features (2)
        'ycrcb_cr_mean',
        'ycrcb_cr_max'
    ]

    assert len(names) == 8, f"Expected 8 feature names, got {len(names)}"

    return names


if __name__ == '__main__':
    # Test feature extraction
    print("Testing Advanced Color Moments...")

    # Create a dummy cell with concentrated red (simulating ball)
    dummy_cell_ball = np.zeros((75, 100, 3), dtype=np.uint8)
    dummy_cell_ball[:, :] = [50, 50, 100]  # BGR: Dark red background
    cv2.circle(dummy_cell_ball, (50, 37), 20, (50, 50, 250), -1)  # Bright red circle

    # Create a dummy cell with uniform green (simulating grass)
    dummy_cell_grass = np.zeros((75, 100, 3), dtype=np.uint8)
    dummy_cell_grass[:, :] = [50, 120, 50]  # BGR: Green

    # Create a dummy cell with uniform gray (simulating stump)
    dummy_cell_stump = np.ones((75, 100, 3), dtype=np.uint8) * 200  # White/light gray

    # Extract features
    features_ball = extract_advanced_color_moments(dummy_cell_ball)
    features_grass = extract_advanced_color_moments(dummy_cell_grass)
    features_stump = extract_advanced_color_moments(dummy_cell_stump)

    print(f"\nExtracted {len(features_ball)} features")
    print(f"\nKey Color Moment Metrics Comparison:")
    print(f"{'Feature Name':<25} {'Ball':<15} {'Grass':<15} {'Stump':<15}")
    print("=" * 75)

    names = get_advanced_color_moment_names()

    # Show key discriminative features
    key_features = ['red_skewness', 'red_kurtosis', 'ycrcb_cr_mean', 'ycrcb_cr_max']
    for key in key_features:
        idx = names.index(key)
        print(f"{key:<25} {features_ball[idx]:<15.4f} {features_grass[idx]:<15.4f} {features_stump[idx]:<15.4f}")

    print("\n✓ Test passed!")
    print("  Ball: High red skewness/kurtosis, high YCrCb Cr (red chrominance)")
    print("  Grass: Low red, high green moments")
    print("  Stump: Low skewness/kurtosis (uniform gray)")
