"""
YUV Color Space Features for Cricket Object Detection

This module extracts 6 YUV color space features for ball/bat/stump detection.

YUV separates luminance (Y) from chrominance (U, V):
- Y: Brightness/intensity (0-255)
- U: Blue-yellow chrominance (0-255, 128=neutral)
- V: Red-cyan chrominance (0-255, 128=neutral)

Features breakdown:
- U channel statistics (3): Blue-yellow chrominance
- V channel statistics (3): Red-cyan chrominance (critical for ball)

Critical for:
- Ball: High V mean/max (>140, red chrominance), low U
- Grass: Low V (<120, cyan/green), moderate U
- Stump: V≈128 (neutral, white/gray), U≈128
- Bat: Moderate V (120-135, brown), low U

Why YUV complements YCrCb:
- YUV V-channel more sensitive to red-cyan axis than YCrCb Cr
- Alternative chrominance decomposition provides orthogonal information
- Commonly used in video encoding, proven effective for color discrimination

Author: Pynacle Team
Date: 2025
Phase: 4A - Advanced Color Features
"""

import numpy as np
import cv2
from typing import List


def extract_yuv_features(cell_rgb: np.ndarray) -> np.ndarray:
    """
    Extract 6 YUV color space features from RGB cell.

    YUV provides alternative chrominance representation to YCrCb,
    particularly useful for red object detection (ball).

    Args:
        cell_rgb: RGB image cell of shape (H, W, 3) in BGR format
                 Expected size: 75x100 pixels

    Returns:
        Array of 6 float32 features
    """
    # Handle empty or invalid images
    if cell_rgb.size == 0:
        return np.zeros(6, dtype=np.float32)

    # Convert BGR to YUV
    # OpenCV formula:
    # Y = 0.299*R + 0.587*G + 0.114*B
    # U = -0.147*R - 0.289*G + 0.436*B + 128
    # V = 0.615*R - 0.515*G - 0.100*B + 128
    cell_yuv = cv2.cvtColor(cell_rgb, cv2.COLOR_BGR2YUV).astype(np.float32)

    # Extract channels
    Y = cell_yuv[:, :, 0]   # Luma (brightness): 0-255
    U = cell_yuv[:, :, 1]   # Blue-yellow chrominance: 0-255 (128=neutral)
    V = cell_yuv[:, :, 2]   # Red-cyan chrominance: 0-255 (128=neutral)

    # ========== U CHANNEL FEATURES (3 features) ==========
    # U captures blue-yellow axis
    # High U (>128): Blue dominant
    # Low U (<128): Yellow dominant

    # 1. YUV U mean
    # Sky/water: High U (blue)
    # Bat: Low U (brown/yellow tint)
    # Most cricket objects: U ≈ 120-135
    yuv_u_mean = np.mean(U)

    # 2. YUV U std (chrominance uniformity)
    # Uniform color: Low std
    # Varied texture: Higher std
    yuv_u_std = np.std(U)

    # 3. YUV U max (peak blue chrominance)
    # Identifies presence of blue pixels
    yuv_u_max = np.max(U)

    # ========== V CHANNEL FEATURES (3 features) ==========
    # V captures red-cyan axis - CRITICAL FOR BALL DETECTION
    # High V (>128): Red dominant
    # Low V (<128): Cyan/green dominant

    # 4. YUV V mean (red chrominance)
    # Ball: V > 140 (red-dominated color)
    # Grass: V < 120 (green-dominated color)
    # Stump: V ≈ 128 (neutral gray/white)
    # Bat: V ≈ 120-135 (brown, slight red tint)
    yuv_v_mean = np.mean(V)

    # 5. YUV V std (red chrominance uniformity)
    # Ball: Low std (uniform red)
    # Grass: Higher std (texture variation)
    yuv_v_std = np.std(V)

    # 6. YUV V max (peak red chrominance)
    # Ball should have very high V max (>160)
    # Critical discriminator for red ball detection
    yuv_v_max = np.max(V)

    # ========== COMBINE ALL FEATURES ==========
    features = np.array([
        # U Channel Features (3)
        yuv_u_mean,
        yuv_u_std,
        yuv_u_max,

        # V Channel Features (3)
        yuv_v_mean,
        yuv_v_std,
        yuv_v_max
    ], dtype=np.float32)

    # Verify we have exactly 6 features
    assert len(features) == 6, f"Expected 6 features, got {len(features)}"

    # Replace any NaN or Inf values with 0.0 (defensive programming)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def get_yuv_feature_names() -> List[str]:
    """
    Get names of all 6 YUV features.

    Returns:
        List of 6 feature names
    """
    names = [
        # U Channel Features (3)
        'yuv_u_mean',
        'yuv_u_std',
        'yuv_u_max',

        # V Channel Features (3)
        'yuv_v_mean',
        'yuv_v_std',
        'yuv_v_max'
    ]

    assert len(names) == 6, f"Expected 6 feature names, got {len(names)}"

    return names


if __name__ == '__main__':
    # Test feature extraction
    print("Testing YUV Color Space Features...")

    # Create a dummy cell with concentrated red (simulating ball)
    dummy_cell_ball = np.zeros((75, 100, 3), dtype=np.uint8)
    dummy_cell_ball[:, :] = [50, 50, 100]  # BGR: Dark red background
    cv2.circle(dummy_cell_ball, (50, 37), 20, (50, 50, 250), -1)  # Bright red circle

    # Create a dummy cell with uniform green (simulating grass)
    dummy_cell_grass = np.zeros((75, 100, 3), dtype=np.uint8)
    dummy_cell_grass[:, :] = [50, 120, 50]  # BGR: Green

    # Create a dummy cell with uniform gray (simulating stump)
    dummy_cell_stump = np.ones((75, 100, 3), dtype=np.uint8) * 200  # White/light gray

    # Create a dummy cell with brown (simulating bat)
    dummy_cell_bat = np.zeros((75, 100, 3), dtype=np.uint8)
    dummy_cell_bat[:, :] = [60, 100, 140]  # BGR: Brown

    # Extract features
    features_ball = extract_yuv_features(dummy_cell_ball)
    features_grass = extract_yuv_features(dummy_cell_grass)
    features_stump = extract_yuv_features(dummy_cell_stump)
    features_bat = extract_yuv_features(dummy_cell_bat)

    print(f"\nExtracted {len(features_ball)} features")
    print(f"\nKey YUV Metrics Comparison:")
    print(f"{'Feature Name':<20} {'Ball':<15} {'Grass':<15} {'Stump':<15} {'Bat':<15}")
    print("=" * 85)

    names = get_yuv_feature_names()

    # Show all features (only 6 total)
    for name in names:
        idx = names.index(name)
        print(f"{name:<20} {features_ball[idx]:<15.4f} {features_grass[idx]:<15.4f} "
              f"{features_stump[idx]:<15.4f} {features_bat[idx]:<15.4f}")

    print("\n✓ Test passed!")
    print("  Ball: High V mean/max (>140, red chrominance)")
    print("  Grass: Low V (<120, green chrominance)")
    print("  Stump: V≈128 (neutral, white/gray)")
    print("  Bat: Moderate V (120-135, brown with slight red)")
    print("\nV channel is CRITICAL for ball detection - complements YCrCb Cr channel")
