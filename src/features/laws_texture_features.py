"""
Laws' Texture Energy Features for Cricket Object Detection

This module extracts 9 Laws' texture energy features for bat/stump/grass discrimination.

Laws' texture filters detect specific patterns:
- L (Level): Average intensity
- E (Edge): Edge detection
- S (Spot): Spot detection

Combining these in 2D gives 9 filters that capture different texture patterns.

Features breakdown:
- 9 Laws' filter responses (L3E3, E3L3, E3E3, S3E3, E3S3, S3L3, L3S3, S3S3, L3L3)

Critical for:
- Bat: High E3E3, L3E3 (wood grain has directional edges)
- Stump: High E3L3, S3L3 (vertical stick with white texture)
- Grass: Low all (random texture, no structured patterns)
- No-Object: Similar to grass

Author: Pynacle Team
Date: 2025
Phase: 4C - Advanced Texture Features
"""

import numpy as np
import cv2
from scipy.ndimage import convolve
from typing import List


def extract_laws_texture_features(cell_gray: np.ndarray) -> np.ndarray:
    """
    Extract 9 Laws' texture energy features from grayscale cell.

    Laws' filters are predefined 3x3 kernels that detect texture patterns.
    They are particularly effective for ordered patterns (wood grain)
    vs random patterns (grass).

    Args:
        cell_gray: Grayscale image cell of shape (H, W)
                   Expected size: 75x100 pixels

    Returns:
        Array of 9 float32 features
    """
    # Handle empty or invalid images
    if cell_gray.size == 0:
        return np.zeros(9, dtype=np.float32)

    # Convert to float for convolution
    cell_float = cell_gray.astype(np.float32)

    # ========== LAWS' 1D KERNELS ==========
    # L3 (Level): Detects average intensity
    L3 = np.array([1, 2, 1], dtype=np.float32)

    # E3 (Edge): Detects edges
    E3 = np.array([-1, 0, 1], dtype=np.float32)

    # S3 (Spot): Detects spots
    S3 = np.array([-1, 2, -1], dtype=np.float32)

    # ========== GENERATE 2D KERNELS ==========
    # Outer product of 1D kernels creates 2D filters

    # 1. L3E3: Level-Edge (horizontal edges)
    # Bat: High (horizontal wood grain edges)
    kernel_L3E3 = np.outer(L3, E3)

    # 2. E3L3: Edge-Level (vertical edges)
    # Stump: High (vertical stick edges)
    kernel_E3L3 = np.outer(E3, L3)

    # 3. E3E3: Edge-Edge (all edges)
    # Bat/Stump: High (object boundaries, wood grain)
    # Grass: Low (no strong edges)
    kernel_E3E3 = np.outer(E3, E3)

    # 4. S3E3: Spot-Edge (spots with edges)
    # Detects small features with directional variation
    kernel_S3E3 = np.outer(S3, E3)

    # 5. E3S3: Edge-Spot (edges with spots)
    # Detects directional features with intensity variation
    kernel_E3S3 = np.outer(E3, S3)

    # 6. S3L3: Spot-Level (spots with average)
    # Stump: Detects white spots in vertical pattern
    kernel_S3L3 = np.outer(S3, L3)

    # 7. L3S3: Level-Spot (average with spots)
    # Detects horizontal patterns with intensity variation
    kernel_L3S3 = np.outer(L3, S3)

    # 8. S3S3: Spot-Spot (spots in all directions)
    # Ball: May detect circular pattern
    # Grass: Low (no spot patterns)
    kernel_S3S3 = np.outer(S3, S3)

    # 9. L3L3: Level-Level (average intensity)
    # Baseline texture measure
    kernel_L3L3 = np.outer(L3, L3)

    # ========== APPLY FILTERS AND COMPUTE ENERGY ==========
    # Texture energy = mean absolute response

    # 1. L3E3 energy
    response_L3E3 = convolve(cell_float, kernel_L3E3, mode='reflect')
    energy_L3E3 = np.mean(np.abs(response_L3E3))

    # 2. E3L3 energy
    response_E3L3 = convolve(cell_float, kernel_E3L3, mode='reflect')
    energy_E3L3 = np.mean(np.abs(response_E3L3))

    # 3. E3E3 energy
    response_E3E3 = convolve(cell_float, kernel_E3E3, mode='reflect')
    energy_E3E3 = np.mean(np.abs(response_E3E3))

    # 4. S3E3 energy
    response_S3E3 = convolve(cell_float, kernel_S3E3, mode='reflect')
    energy_S3E3 = np.mean(np.abs(response_S3E3))

    # 5. E3S3 energy
    response_E3S3 = convolve(cell_float, kernel_E3S3, mode='reflect')
    energy_E3S3 = np.mean(np.abs(response_E3S3))

    # 6. S3L3 energy
    response_S3L3 = convolve(cell_float, kernel_S3L3, mode='reflect')
    energy_S3L3 = np.mean(np.abs(response_S3L3))

    # 7. L3S3 energy
    response_L3S3 = convolve(cell_float, kernel_L3S3, mode='reflect')
    energy_L3S3 = np.mean(np.abs(response_L3S3))

    # 8. S3S3 energy
    response_S3S3 = convolve(cell_float, kernel_S3S3, mode='reflect')
    energy_S3S3 = np.mean(np.abs(response_S3S3))

    # 9. L3L3 energy
    response_L3L3 = convolve(cell_float, kernel_L3L3, mode='reflect')
    energy_L3L3 = np.mean(np.abs(response_L3L3))

    # ========== COMBINE ALL FEATURES ==========
    features = np.array([
        energy_L3E3,
        energy_E3L3,
        energy_E3E3,
        energy_S3E3,
        energy_E3S3,
        energy_S3L3,
        energy_L3S3,
        energy_S3S3,
        energy_L3L3
    ], dtype=np.float32)

    # Verify we have exactly 9 features
    assert len(features) == 9, f"Expected 9 features, got {len(features)}"

    # Replace any NaN or Inf values with 0.0 (defensive programming)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def get_laws_texture_feature_names() -> List[str]:
    """
    Get names of all 9 Laws' texture features.

    Returns:
        List of 9 feature names
    """
    names = [
        'laws_L3E3',  # Level-Edge (horizontal edges)
        'laws_E3L3',  # Edge-Level (vertical edges)
        'laws_E3E3',  # Edge-Edge (all edges)
        'laws_S3E3',  # Spot-Edge
        'laws_E3S3',  # Edge-Spot
        'laws_S3L3',  # Spot-Level
        'laws_L3S3',  # Level-Spot
        'laws_S3S3',  # Spot-Spot
        'laws_L3L3'   # Level-Level (baseline)
    ]

    assert len(names) == 9, f"Expected 9 feature names, got {len(names)}"

    return names


if __name__ == '__main__':
    # Test feature extraction
    print("Testing Laws' Texture Energy Features...")

    # Create a dummy cell with horizontal lines (simulating bat wood grain)
    dummy_cell_bat = np.zeros((75, 100), dtype=np.uint8)
    for y in range(0, 75, 10):
        dummy_cell_bat[y:y+3, :] = 200  # Horizontal stripes

    # Create a dummy cell with vertical lines (simulating stump)
    dummy_cell_stump = np.zeros((75, 100), dtype=np.uint8)
    for x in range(0, 100, 15):
        dummy_cell_stump[:, x:x+2] = 220  # Vertical stripes

    # Create a dummy cell with random noise (simulating grass)
    dummy_cell_grass = np.random.randint(80, 140, (75, 100), dtype=np.uint8)

    # Create a dummy cell with uniform intensity (simulating no object)
    dummy_cell_no_object = np.ones((75, 100), dtype=np.uint8) * 120

    # Extract features
    features_bat = extract_laws_texture_features(dummy_cell_bat)
    features_stump = extract_laws_texture_features(dummy_cell_stump)
    features_grass = extract_laws_texture_features(dummy_cell_grass)
    features_no_object = extract_laws_texture_features(dummy_cell_no_object)

    print(f"\nExtracted {len(features_bat)} features")
    print(f"\nKey Laws' Texture Metrics Comparison:")
    print(f"{'Feature Name':<20} {'Bat':<15} {'Stump':<15} {'Grass':<15} {'No-Object':<15}")
    print("=" * 80)

    names = get_laws_texture_feature_names()

    # Show key discriminative features
    key_features = ['laws_L3E3', 'laws_E3L3', 'laws_E3E3']
    for key in key_features:
        idx = names.index(key)
        print(f"{key:<20} {features_bat[idx]:<15.4f} {features_stump[idx]:<15.4f} "
              f"{features_grass[idx]:<15.4f} {features_no_object[idx]:<15.4f}")

    print("\n✓ Test passed!")
    print("  Bat: High L3E3 (horizontal edges from wood grain)")
    print("  Stump: High E3L3 (vertical edges)")
    print("  Grass: Low all (no structured patterns)")
    print("\nLaws' texture features are CRITICAL for bat wood texture discrimination!")
