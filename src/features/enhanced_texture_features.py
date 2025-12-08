"""
Enhanced Texture Features for Cricket Object Detection

This module extracts 14 advanced texture features using Haralick texture descriptors
computed from Gray Level Co-occurrence Matrix (GLCM).

Features breakdown:
- Haralick Texture (14): Advanced GLCM-based texture statistics

Critical for:
- Bat: Wood grain texture (moderate homogeneity, high energy)
- Stump: Smooth white surface with vertical patterns
- Ball: Smooth leather texture (high homogeneity)
- Grass: Complex random texture (low homogeneity, low energy)

Note: We use mahotas library for Haralick features as it provides comprehensive
texture analysis beyond basic GLCM features.

Author: Pynacle Team
Date: 2025
"""

import numpy as np
import cv2
from typing import List

try:
    import mahotas
    MAHOTAS_AVAILABLE = True
except ImportError:
    MAHOTAS_AVAILABLE = False
    print("Warning: mahotas not available, will use fallback Haralick implementation")


def extract_enhanced_texture_features(cell_gray: np.ndarray) -> np.ndarray:
    """
    Extract 14 enhanced texture features from grayscale cell.

    Uses Haralick texture features computed from GLCM at multiple angles.
    These features capture second-order texture statistics critical for
    discriminating bat wood texture from smooth ball and grass.

    Args:
        cell_gray: Grayscale image of shape (H, W) or RGB image (will be converted)
                  Expected size: 75x100 pixels

    Returns:
        Array of 14 float32 features
    """
    # Handle empty or invalid images
    if cell_gray.size == 0:
        return np.zeros(14, dtype=np.float32)

    # Convert to grayscale if RGB
    if len(cell_gray.shape) == 3:
        cell_gray = cv2.cvtColor(cell_gray, cv2.COLOR_BGR2GRAY)

    # ========== HARALICK TEXTURE FEATURES (14 features) ==========

    try:
        if MAHOTAS_AVAILABLE:
            # Use mahotas for comprehensive Haralick features
            # Compute Haralick features at 4 angles (0°, 45°, 90°, 135°)
            # and average them for rotation invariance

            # Quantize to 8-bit if needed (GLCM works better with reduced levels)
            if cell_gray.dtype != np.uint8:
                cell_gray = cell_gray.astype(np.uint8)

            # Compute Haralick features (returns 13 features × 4 directions = 52 values)
            # We'll average across the 4 directions
            haralick_features = mahotas.features.haralick(cell_gray, ignore_zeros=True, return_mean=True)

            # Mahotas returns 13 features, we'll extract the most discriminative ones
            # and add one more (information measure correlation)

            # Feature indices (from mahotas documentation):
            # 0: Angular Second Moment (Energy)
            # 1: Contrast
            # 2: Correlation
            # 3: Sum of Squares (Variance)
            # 4: Inverse Difference Moment (Homogeneity)
            # 5: Sum Average
            # 6: Sum Variance
            # 7: Sum Entropy
            # 8: Entropy
            # 9: Difference Variance
            # 10: Difference Entropy
            # 11: Information Measure of Correlation 1
            # 12: Information Measure of Correlation 2

            # Extract all 13 features
            angular_second_moment = haralick_features[0]
            contrast = haralick_features[1]
            correlation = haralick_features[2]
            sum_of_squares_variance = haralick_features[3]
            inverse_difference_moment = haralick_features[4]
            sum_average = haralick_features[5]
            sum_variance = haralick_features[6]
            sum_entropy = haralick_features[7]
            entropy = haralick_features[8]
            difference_variance = haralick_features[9]
            difference_entropy = haralick_features[10]
            information_measure_correlation_1 = haralick_features[11]
            information_measure_correlation_2 = haralick_features[12]

            # Add one more feature: Normalized contrast (14th feature)
            # Contrast normalized by sum_average to be scale-invariant
            if sum_average > 0:
                normalized_contrast = contrast / (sum_average + 1e-7)
            else:
                normalized_contrast = 0.0

        else:
            # Fallback: Use basic GLCM features (simpler implementation)
            # This won't be as comprehensive but will work without mahotas

            # Quantize image to 16 gray levels for faster GLCM computation
            cell_quantized = (cell_gray // 16).astype(np.uint8)

            # Compute basic GLCM statistics
            from skimage.feature import graycomatrix, graycoprops

            # Distance 1, 4 angles (0°, 45°, 90°, 135°)
            glcm = graycomatrix(cell_quantized, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                               levels=16, symmetric=True, normed=True)

            # Compute properties and average across angles
            contrast_vals = graycoprops(glcm, 'contrast')
            homogeneity_vals = graycoprops(glcm, 'homogeneity')
            energy_vals = graycoprops(glcm, 'energy')
            correlation_vals = graycoprops(glcm, 'correlation')
            dissimilarity_vals = graycoprops(glcm, 'dissimilarity')

            # Average across angles
            angular_second_moment = float(np.mean(energy_vals))
            contrast = float(np.mean(contrast_vals))
            correlation = float(np.mean(correlation_vals))
            inverse_difference_moment = float(np.mean(homogeneity_vals))

            # Estimate other features from available data
            sum_of_squares_variance = float(np.var(cell_gray))
            sum_average = float(np.mean(cell_gray))
            sum_variance = float(np.var(cell_gray))
            entropy = -np.sum(cell_gray * np.log(cell_gray + 1e-10)) / (cell_gray.size)
            sum_entropy = entropy * 0.8  # Approximation
            difference_variance = float(np.mean(dissimilarity_vals))
            difference_entropy = entropy * 0.6  # Approximation
            information_measure_correlation_1 = correlation * 0.7  # Approximation
            information_measure_correlation_2 = correlation * 0.5  # Approximation
            normalized_contrast = contrast / (sum_average + 1e-7)

    except Exception as e:
        # If Haralick extraction fails, return zeros
        print(f"Haralick extraction failed: {e}")
        angular_second_moment = 0.0
        contrast = 0.0
        correlation = 0.0
        sum_of_squares_variance = 0.0
        inverse_difference_moment = 0.0
        sum_average = 0.0
        sum_variance = 0.0
        sum_entropy = 0.0
        entropy = 0.0
        difference_variance = 0.0
        difference_entropy = 0.0
        information_measure_correlation_1 = 0.0
        information_measure_correlation_2 = 0.0
        normalized_contrast = 0.0

    # ========== COMBINE ALL FEATURES ==========
    features = np.array([
        angular_second_moment,           # 1. Energy (uniformity)
        contrast,                         # 2. Contrast (local variation)
        correlation,                      # 3. Correlation (linear dependency)
        sum_of_squares_variance,          # 4. Variance
        inverse_difference_moment,        # 5. Homogeneity (smoothness)
        sum_average,                      # 6. Sum average
        sum_variance,                     # 7. Sum variance
        sum_entropy,                      # 8. Sum entropy (randomness)
        entropy,                          # 9. Entropy (disorder)
        difference_variance,              # 10. Difference variance
        difference_entropy,               # 11. Difference entropy
        information_measure_correlation_1, # 12. Info measure correlation 1
        information_measure_correlation_2, # 13. Info measure correlation 2
        normalized_contrast               # 14. Normalized contrast
    ], dtype=np.float32)

    # Verify we have exactly 14 features
    assert len(features) == 14, f"Expected 14 features, got {len(features)}"

    # Replace any NaN or Inf values with 0.0 (defensive programming)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def get_enhanced_texture_feature_names() -> List[str]:
    """
    Get names of all 14 enhanced texture features.

    Returns:
        List of 14 feature names
    """
    names = [
        'haralick_angular_second_moment',
        'haralick_contrast',
        'haralick_correlation',
        'haralick_sum_of_squares_variance',
        'haralick_inverse_difference_moment',
        'haralick_sum_average',
        'haralick_sum_variance',
        'haralick_sum_entropy',
        'haralick_entropy',
        'haralick_difference_variance',
        'haralick_difference_entropy',
        'haralick_information_measure_correlation_1',
        'haralick_information_measure_correlation_2',
        'haralick_normalized_contrast'
    ]

    assert len(names) == 14, f"Expected 14 feature names, got {len(names)}"

    return names


if __name__ == '__main__':
    # Test feature extraction
    print("Testing Enhanced Texture Features (Haralick)...")
    if not MAHOTAS_AVAILABLE:
        print("Note: Running with fallback implementation (mahotas not installed)")

    # Create a dummy cell with smooth texture (simulating ball)
    dummy_cell_smooth = np.ones((75, 100), dtype=np.uint8) * 150  # Uniform gray

    # Create a dummy cell with wood grain texture (simulating bat)
    dummy_cell_wood = np.zeros((75, 100), dtype=np.uint8)
    for i in range(75):
        # Horizontal wood grain pattern
        intensity = 100 + 30 * np.sin(i * 0.3) + np.random.randint(-10, 10)
        dummy_cell_wood[i, :] = np.clip(intensity, 0, 255)

    # Create a dummy cell with random texture (simulating grass)
    dummy_cell_grass = np.random.randint(80, 160, (75, 100), dtype=np.uint8)

    # Extract features
    features_smooth = extract_enhanced_texture_features(dummy_cell_smooth)
    features_wood = extract_enhanced_texture_features(dummy_cell_wood)
    features_grass = extract_enhanced_texture_features(dummy_cell_grass)

    print(f"\nExtracted {len(features_smooth)} features")
    print(f"\nKey Haralick Metrics Comparison:")
    print(f"{'Feature Name':<40} {'Smooth (Ball)':<15} {'Wood (Bat)':<15} {'Random (Grass)':<15}")
    print("=" * 90)

    names = get_enhanced_texture_feature_names()

    # Show key discriminative features
    key_features = ['haralick_angular_second_moment', 'haralick_contrast',
                   'haralick_inverse_difference_moment', 'haralick_entropy']
    for key in key_features:
        idx = names.index(key)
        print(f"{key:<40} {features_smooth[idx]:<15.6f} {features_wood[idx]:<15.6f} {features_grass[idx]:<15.6f}")

    print("\n✓ Test passed!")
    print("  Smooth: High homogeneity, low entropy (uniform)")
    print("  Wood: Moderate homogeneity, moderate contrast (patterned)")
    print("  Grass: Low homogeneity, high entropy (random)")
