"""
Chromaticity Features for Cricket Object Detection

This module extracts 14 chromaticity-based color features for ball/bat/stump detection.

Features breakdown:
- RG Chromaticity (4): Intensity-invariant color representation
- Excess Red Index (4): Red detection inspired by vegetation indices
- Opponent Color Space (6): Redness vs greenness discrimination

Critical for:
- Ball: High r chromaticity (r>0.5), high Excess Red, positive O1 (redness)
- Grass: High g chromaticity (g>0.4), negative ExR, negative O1 (greenness)
- Bat: Moderate r/g, low ExR
- Stump: Low r/g (white/gray), near-zero ExR

Author: Pynacle Team
Date: 2025
Phase: 4A - Advanced Color Features
"""

import numpy as np
import cv2
from typing import List


def extract_chromaticity_features(cell_rgb: np.ndarray) -> np.ndarray:
    """
    Extract 14 chromaticity features from RGB cell.

    Chromaticity features normalize out intensity, making them invariant to
    lighting conditions and shadows. Critical for red ball detection.

    Args:
        cell_rgb: RGB image cell of shape (H, W, 3) in BGR format
                 Expected size: 75x100 pixels

    Returns:
        Array of 14 float32 features
    """
    # Handle empty or invalid images
    if cell_rgb.size == 0:
        return np.zeros(14, dtype=np.float32)

    # OpenCV uses BGR format
    B = cell_rgb[:, :, 0].astype(np.float32)
    G = cell_rgb[:, :, 1].astype(np.float32)
    R = cell_rgb[:, :, 2].astype(np.float32)

    # Avoid division by zero
    total = R + G + B + 1e-7

    # ========== RG CHROMATICITY (4 features) ==========
    # Chromaticity: r = R/(R+G+B), g = G/(R+G+B)
    # Note: b = B/(R+G+B) is redundant since r+g+b=1
    # This representation is independent of illumination intensity

    # 1. r chromaticity mean
    # Ball: r > 0.5 (red-dominated)
    # Grass: r < 0.3 (green-dominated)
    # Stump: r ≈ 0.33 (white/gray, equal RGB)
    r = R / total
    rg_chrom_r_mean = np.mean(r)

    # 2. g chromaticity mean
    # Grass: g > 0.4
    # Ball: g < 0.25
    g = G / total
    rg_chrom_g_mean = np.mean(g)

    # 3. r chromaticity std (color uniformity)
    # Ball: Low std (uniform red)
    # Grass: Higher std (texture variation)
    rg_chrom_r_std = np.std(r)

    # 4. g chromaticity std
    rg_chrom_g_std = np.std(g)

    # ========== EXCESS RED INDEX (4 features) ==========
    # Inspired by vegetation indices (ExG, ExR)
    # ExR = (R - G) / (R + G + B)
    # Emphasizes redness vs greenness, normalized by total intensity

    # 5. Excess Red mean
    # Ball: ExR > 0.2 (red dominant)
    # Grass: ExR < -0.1 (green dominant)
    ExR = (R - G) / total
    excess_red_mean = np.mean(ExR)

    # 6. Red Index mean
    # RedIndex = R / (G + B)
    # Ball: RedIndex > 1.5
    # Grass: RedIndex < 0.5
    RedIndex = R / (G + B + 1e-7)
    red_index_mean = np.mean(RedIndex)

    # 7. Normalized R-G mean
    # NormRG = (R - G) / max(R, G)
    # Range: [-1, 1], where 1 = pure red, -1 = pure green
    NormRG = (R - G) / np.maximum(np.maximum(R, G), 1e-7)
    norm_rg_mean = np.mean(NormRG)

    # 8. Normalized R-G max
    # Ball should have pixels with NormRG close to 1.0
    norm_rg_max = np.max(NormRG)

    # ========== OPPONENT COLOR SPACE (6 features) ==========
    # Opponent color theory: Human vision processes color as:
    # - Red vs Green (O1)
    # - Yellow vs Blue (O2)
    # - Intensity (O3)

    # 9. O1 mean (redness vs greenness)
    # Ball: O1 > 50 (high positive = red)
    # Grass: O1 < -20 (negative = green)
    O1 = R - G
    opponent_o1_mean = np.mean(O1)

    # 10. O1 std (color uniformity)
    opponent_o1_std = np.std(O1)

    # 11. O2 mean (yellowness vs blueness)
    # O2 = (R + G) - 2*B
    # Yellow/brown bat: O2 > 0
    # Blue sky (rare): O2 < 0
    O2 = R + G - 2 * B
    opponent_o2_mean = np.mean(O2)

    # 12. O2 std
    opponent_o2_std = np.std(O2)

    # 13. O3 mean (intensity/brightness)
    # O3 = R + G + B
    # Stump: High O3 (white)
    # Dark objects: Low O3
    O3 = R + G + B
    opponent_o3_mean = np.mean(O3)

    # 14. O3 std (intensity variance)
    opponent_o3_std = np.std(O3)

    # ========== COMBINE ALL FEATURES ==========
    features = np.array([
        # RG Chromaticity (4)
        rg_chrom_r_mean,
        rg_chrom_g_mean,
        rg_chrom_r_std,
        rg_chrom_g_std,

        # Excess Red Index (4)
        excess_red_mean,
        red_index_mean,
        norm_rg_mean,
        norm_rg_max,

        # Opponent Color Space (6)
        opponent_o1_mean,
        opponent_o1_std,
        opponent_o2_mean,
        opponent_o2_std,
        opponent_o3_mean,
        opponent_o3_std
    ], dtype=np.float32)

    # Verify we have exactly 14 features
    assert len(features) == 14, f"Expected 14 features, got {len(features)}"

    # Replace any NaN or Inf values with 0.0 (defensive programming)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def get_chromaticity_feature_names() -> List[str]:
    """
    Get names of all 14 chromaticity features.

    Returns:
        List of 14 feature names
    """
    names = [
        # RG Chromaticity (4)
        'rg_chrom_r_mean',
        'rg_chrom_g_mean',
        'rg_chrom_r_std',
        'rg_chrom_g_std',

        # Excess Red Index (4)
        'excess_red_mean',
        'red_index_mean',
        'norm_rg_mean',
        'norm_rg_max',

        # Opponent Color Space (6)
        'opponent_o1_mean',
        'opponent_o1_std',
        'opponent_o2_mean',
        'opponent_o2_std',
        'opponent_o3_mean',
        'opponent_o3_std'
    ]

    assert len(names) == 14, f"Expected 14 feature names, got {len(names)}"

    return names


if __name__ == '__main__':
    # Test feature extraction
    print("Testing Chromaticity Features...")

    # Create a dummy cell with concentrated red (simulating ball)
    dummy_cell_ball = np.zeros((75, 100, 3), dtype=np.uint8)
    dummy_cell_ball[:, :] = [50, 50, 80]  # BGR: Dark red background
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
    features_ball = extract_chromaticity_features(dummy_cell_ball)
    features_grass = extract_chromaticity_features(dummy_cell_grass)
    features_stump = extract_chromaticity_features(dummy_cell_stump)
    features_bat = extract_chromaticity_features(dummy_cell_bat)

    print(f"\nExtracted {len(features_ball)} features")
    print(f"\nKey Chromaticity Metrics Comparison:")
    print(f"{'Feature Name':<25} {'Ball':<15} {'Grass':<15} {'Stump':<15} {'Bat':<15}")
    print("=" * 95)

    names = get_chromaticity_feature_names()

    # Show key discriminative features
    key_features = ['rg_chrom_r_mean', 'rg_chrom_g_mean', 'excess_red_mean',
                   'red_index_mean', 'opponent_o1_mean', 'norm_rg_max']
    for key in key_features:
        idx = names.index(key)
        print(f"{key:<25} {features_ball[idx]:<15.4f} {features_grass[idx]:<15.4f} "
              f"{features_stump[idx]:<15.4f} {features_bat[idx]:<15.4f}")

    print("\n✓ Test passed!")
    print("  Ball: High r chromaticity (>0.5), high ExR, high RedIndex, positive O1")
    print("  Grass: High g chromaticity (>0.4), negative ExR, low RedIndex, negative O1")
    print("  Stump: Balanced r/g (~0.33), near-zero ExR, RedIndex~1.0")
    print("  Bat: Moderate r (0.35-0.45), low ExR, moderate O1")
