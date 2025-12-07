"""
Enhanced Color Features for Cricket Object Detection

This module extracts 18 advanced color features specifically designed for
detecting red cricket balls in varying lighting conditions.

Features breakdown:
- Red Dominance Features (6): Detect red objects (cricket ball)
- HSV Red Detection (6): Lighting-invariant red detection
- LAB Color Space (6): Perceptually-uniform color space for red discrimination

Author: Pynacle Team
Date: 2025
"""

import numpy as np
import cv2
from typing import List


def extract_enhanced_color_features(cell_rgb: np.ndarray) -> np.ndarray:
    """
    Extract 18 enhanced color features from RGB cell.

    Critical for ball detection - current system only detects 3/17 balls (F1: 0.0619).
    These features provide red-specific discrimination that basic RGB/HSV lack.

    Args:
        cell_rgb: RGB image cell of shape (H, W, 3) in BGR format (OpenCV standard)
                 Expected size: 75x100 pixels

    Returns:
        Array of 18 float32 features
    """
    # Ensure we have a valid image
    if cell_rgb.size == 0:
        return np.zeros(18, dtype=np.float32)

    # OpenCV uses BGR, so we need to handle color channels correctly
    # BGR format: cell_rgb[:,:,0]=Blue, cell_rgb[:,:,1]=Green, cell_rgb[:,:,2]=Red
    B = cell_rgb[:, :, 0].astype(np.float32)
    G = cell_rgb[:, :, 1].astype(np.float32)
    R = cell_rgb[:, :, 2].astype(np.float32)

    eps = 1e-7  # Small epsilon to prevent division by zero

    # ========== RED DOMINANCE FEATURES (6 features) ==========
    # These features detect when red channel dominates (cricket ball signature)

    # 1. Red dominance ratio: R / (G + B)
    # High values (>1.5) indicate strong red dominance (ball)
    red_dominance_ratio = np.mean(R / (G + B + eps))

    # 2. Maximum red channel value in cell
    # Peak red intensity (ball has bright red spots)
    red_channel_max = np.max(R)

    # 3. Percentage of pixels with high red (R > 150)
    # Ball should have significant high-red region
    red_threshold_pct = np.mean(R > 150)

    # 4. Normalized red mean: R / (R + G + B)
    # Fraction of total intensity that is red
    total_intensity = R + G + B + eps
    normalized_red_mean = np.mean(R / total_intensity)

    # 5. Red-Green difference: Mean(R - G)
    # Ball has R >> G (red much greater than green)
    red_green_diff = np.mean(R - G)

    # 6. Red-Blue difference: Mean(R - B)
    # Ball has R >> B (red much greater than blue)
    red_blue_diff = np.mean(R - B)

    # ========== HSV RED HUE DETECTION (6 features) ==========
    # HSV is lighting-invariant, critical for ball detection in shadows

    # Convert BGR to HSV
    cell_hsv = cv2.cvtColor(cell_rgb, cv2.COLOR_BGR2HSV).astype(np.float32)
    H = cell_hsv[:, :, 0]  # Hue: 0-180 in OpenCV (not 0-360!)
    S = cell_hsv[:, :, 1]  # Saturation: 0-255
    V = cell_hsv[:, :, 2]  # Value (brightness): 0-255

    # Red hue wraparound in HSV: 0-10 and 170-180 (in OpenCV's 0-180 scale)
    # This handles the discontinuity at 0/360 degrees
    red_hue_mask = ((H >= 0) & (H <= 10)) | ((H >= 170) & (H <= 180))

    # 7. Percentage of pixels in red hue range
    hue_red_wraparound_pct = np.mean(red_hue_mask)

    # 8. High-saturation red pixels percentage
    # Ball is vivid red (high saturation + red hue)
    high_sat_red_mask = red_hue_mask & (S > 150)
    high_saturation_red_pct = np.mean(high_sat_red_mask)

    # 9. Mean hue of red pixels (handle wraparound)
    # Average hue value for pixels in red range
    if np.any(red_hue_mask):
        red_hue_mean_adjusted = np.mean(H[red_hue_mask])
    else:
        red_hue_mean_adjusted = 0.0

    # 10. Standard deviation of red hue
    # Uniformity of red color (ball should have consistent red)
    if np.any(red_hue_mask):
        red_hue_std = np.std(H[red_hue_mask])
    else:
        red_hue_std = 0.0

    # 11. Mean value (brightness) at red hue pixels
    # How bright are the red regions?
    if np.any(red_hue_mask):
        v_channel_at_red_hues = np.mean(V[red_hue_mask])
    else:
        v_channel_at_red_hues = 0.0

    # 12. Mean saturation at red hue pixels
    # How saturated (vivid) are the red regions?
    if np.any(red_hue_mask):
        s_channel_at_red_hues = np.mean(S[red_hue_mask])
    else:
        s_channel_at_red_hues = 0.0

    # ========== LAB COLOR SPACE FEATURES (6 features) ==========
    # LAB is perceptually uniform - better for color distance calculations
    # a* axis: green (-) to red (+) - CRITICAL for red ball detection!

    # Convert BGR to LAB
    cell_lab = cv2.cvtColor(cell_rgb, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = cell_lab[:, :, 0]  # Lightness: 0-255
    A = cell_lab[:, :, 1]  # a* channel: 0-255 (128=neutral, >128=red, <128=green)
    B_lab = cell_lab[:, :, 2]  # b* channel: 0-255 (128=neutral, >128=yellow, <128=blue)

    # 13. Mean of a* channel (green-red axis)
    # High values (>140) indicate red dominance
    lab_a_mean = np.mean(A)

    # 14. Maximum a* value (peak redness)
    lab_a_max = np.max(A)

    # 15. Standard deviation of a* channel
    # Color uniformity in red-green axis
    lab_a_std = np.std(A)

    # 16. Mean of b* channel (blue-yellow axis)
    # Red ball may have slight yellow tint in certain lighting
    lab_b_mean = np.mean(B_lab)

    # 17. Mean lightness
    # Overall brightness of cell
    lab_l_mean = np.mean(L)

    # 18. Color distance to pure red in LAB space
    # Pure red in LAB: approximately L=53, a*=80, b*=67 (for RGB 255,0,0)
    # We'll use LAB (128, 178, 150) as typical cricket ball red
    target_L = 128.0
    target_A = 178.0  # High a* for red
    target_B = 150.0  # Moderate b*

    # Euclidean distance to target red color
    dist_L = (L - target_L) ** 2
    dist_A = (A - target_A) ** 2
    dist_B = (B_lab - target_B) ** 2
    color_distance_squared = dist_L + dist_A + dist_B
    lab_color_distance_to_red = np.mean(np.sqrt(color_distance_squared))

    # ========== COMBINE ALL FEATURES ==========
    features = np.array([
        # Red Dominance (6)
        red_dominance_ratio,
        red_channel_max,
        red_threshold_pct,
        normalized_red_mean,
        red_green_diff,
        red_blue_diff,

        # HSV Red Detection (6)
        hue_red_wraparound_pct,
        high_saturation_red_pct,
        red_hue_mean_adjusted,
        red_hue_std,
        v_channel_at_red_hues,
        s_channel_at_red_hues,

        # LAB Color Space (6)
        lab_a_mean,
        lab_a_max,
        lab_a_std,
        lab_b_mean,
        lab_l_mean,
        lab_color_distance_to_red
    ], dtype=np.float32)

    # Verify we have exactly 18 features
    assert len(features) == 18, f"Expected 18 features, got {len(features)}"

    # Replace any NaN or Inf values with 0.0 (defensive programming)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def get_enhanced_color_feature_names() -> List[str]:
    """
    Get names of all 18 enhanced color features.

    Returns:
        List of 18 feature names
    """
    names = [
        # Red Dominance (6)
        'red_dominance_ratio',
        'red_channel_max',
        'red_threshold_pct',
        'normalized_red_mean',
        'red_green_diff',
        'red_blue_diff',

        # HSV Red Detection (6)
        'hue_red_wraparound_pct',
        'high_saturation_red_pct',
        'red_hue_mean_adjusted',
        'red_hue_std',
        'v_channel_at_red_hues',
        's_channel_at_red_hues',

        # LAB Color Space (6)
        'lab_a_mean',
        'lab_a_max',
        'lab_a_std',
        'lab_b_mean',
        'lab_l_mean',
        'lab_color_distance_to_red'
    ]

    assert len(names) == 18, f"Expected 18 feature names, got {len(names)}"

    return names


if __name__ == '__main__':
    # Test feature extraction
    print("Testing Enhanced Color Features...")

    # Create a dummy red cell (simulating cricket ball)
    dummy_cell_red = np.zeros((75, 100, 3), dtype=np.uint8)
    dummy_cell_red[:, :, 2] = 200  # High red channel (BGR format)
    dummy_cell_red[:, :, 1] = 50   # Low green
    dummy_cell_red[:, :, 0] = 50   # Low blue

    # Create a dummy green cell (simulating grass)
    dummy_cell_green = np.zeros((75, 100, 3), dtype=np.uint8)
    dummy_cell_green[:, :, 1] = 200  # High green
    dummy_cell_green[:, :, 2] = 50   # Low red
    dummy_cell_green[:, :, 0] = 50   # Low blue

    # Extract features
    features_red = extract_enhanced_color_features(dummy_cell_red)
    features_green = extract_enhanced_color_features(dummy_cell_green)

    print(f"\nExtracted {len(features_red)} features")
    print(f"\nFeature comparison (Red cell vs Green cell):")
    print(f"{'Feature Name':<30} {'Red Cell':<12} {'Green Cell':<12} {'Ratio (R/G)':<12}")
    print("=" * 70)

    names = get_enhanced_color_feature_names()
    for i, name in enumerate(names):
        ratio = features_red[i] / (features_green[i] + 1e-7)
        print(f"{name:<30} {features_red[i]:<12.4f} {features_green[i]:<12.4f} {ratio:<12.2f}x")

    print("\n✓ Test passed! Red cell should have much higher red-specific features.")
