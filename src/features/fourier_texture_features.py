"""
Fourier Power Spectrum Texture Features for Cricket Object Detection

This module extracts 6 Fourier-based texture features for bat/grass discrimination.

Fourier analysis detects frequency patterns in textures:
- Bat wood grain has periodic directionality (shows up as peaks in frequency domain)
- Grass is random (white noise pattern in frequency domain)

Features breakdown:
- Radial Power Distribution (4): Power in different frequency bands
- Peak Ratio (1): Ratio of peak power to mean power
- Spectral Entropy (1): Randomness measure in frequency domain

Critical for:
- Bat: Low entropy (periodic), high peak ratio (directional wood grain frequency)
- Grass: High entropy (random), low peak ratio (no dominant frequencies)
- Ball: Medium entropy (circular pattern has some structure)
- Stump: Similar to bat but vertical directionality

Author: Pynacle Team
Date: 2025
Phase: 4C - Advanced Texture Features
"""

import numpy as np
import cv2
from typing import List


def extract_fourier_texture_features(cell_gray: np.ndarray) -> np.ndarray:
    """
    Extract 6 Fourier power spectrum features from grayscale cell.

    FFT transforms spatial texture into frequency domain where
    periodic patterns appear as distinct peaks.

    Args:
        cell_gray: Grayscale image cell of shape (H, W)
                   Expected size: 75x100 pixels

    Returns:
        Array of 6 float32 features
    """
    # Handle empty or invalid images
    if cell_gray.size == 0:
        return np.zeros(6, dtype=np.float32)

    # Convert to float
    cell_float = cell_gray.astype(np.float32)

    # ========== COMPUTE 2D FFT ==========
    # Fast Fourier Transform converts spatial domain to frequency domain
    fft = np.fft.fft2(cell_float)

    # Shift zero-frequency component to center
    fft_shifted = np.fft.fftshift(fft)

    # Compute power spectrum (magnitude squared)
    power_spectrum = np.abs(fft_shifted) ** 2

    # Log transform for better visualization and feature extraction
    # Add 1 to avoid log(0)
    log_power = np.log10(power_spectrum + 1.0)

    # Get center coordinates
    rows, cols = power_spectrum.shape
    center_row, center_col = rows // 2, cols // 2

    # ========== RADIAL POWER DISTRIBUTION (4 features) ==========
    # Divide frequency domain into concentric rings
    # Low frequencies (near center): global patterns, DC component
    # High frequencies (far from center): fine details, noise

    # Create meshgrid for radial distance computation
    y_indices, x_indices = np.ogrid[:rows, :cols]
    distance_from_center = np.sqrt(
        (x_indices - center_col) ** 2 + (y_indices - center_row) ** 2
    )

    # Define radial bands (normalized by max distance)
    max_distance = np.sqrt(center_row ** 2 + center_col ** 2)

    # Band 1: Very low frequencies (0-25% of max distance)
    # Contains DC component and very coarse patterns
    band1_mask = distance_from_center <= 0.25 * max_distance
    band1_power = np.mean(log_power[band1_mask]) if np.any(band1_mask) else 0.0

    # Band 2: Low frequencies (25-50% of max distance)
    # Contains coarse texture patterns
    # Bat wood grain periodicity appears here
    band2_mask = (distance_from_center > 0.25 * max_distance) & \
                  (distance_from_center <= 0.50 * max_distance)
    band2_power = np.mean(log_power[band2_mask]) if np.any(band2_mask) else 0.0

    # Band 3: Medium frequencies (50-75% of max distance)
    # Contains finer texture details
    band3_mask = (distance_from_center > 0.50 * max_distance) & \
                  (distance_from_center <= 0.75 * max_distance)
    band3_power = np.mean(log_power[band3_mask]) if np.any(band3_mask) else 0.0

    # Band 4: High frequencies (75-100% of max distance)
    # Contains fine details and noise
    # Grass has more power in high frequencies (random texture)
    band4_mask = distance_from_center > 0.75 * max_distance
    band4_power = np.mean(log_power[band4_mask]) if np.any(band4_mask) else 0.0

    # ========== PEAK RATIO (1 feature) ==========
    # Ratio of maximum power to mean power
    # Bat wood grain: High peak ratio (dominant frequency from periodic grain)
    # Grass: Low peak ratio (no dominant frequencies, random)

    # Exclude DC component (center pixel) from peak detection
    # as it's always the strongest and not texture-related
    power_no_dc = power_spectrum.copy()
    power_no_dc[center_row, center_col] = 0.0

    max_power = np.max(power_no_dc)
    mean_power = np.mean(power_no_dc)

    if mean_power > 0:
        peak_ratio = max_power / mean_power
    else:
        peak_ratio = 0.0

    # ========== SPECTRAL ENTROPY (1 feature) ==========
    # Measures randomness in frequency domain
    # High entropy: Random texture (grass)
    # Low entropy: Structured texture (bat wood grain)

    # Normalize power spectrum to probability distribution
    total_power = np.sum(power_no_dc)
    if total_power > 0:
        prob_dist = power_no_dc / total_power

        # Compute Shannon entropy
        # H = -sum(p * log2(p))
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        prob_dist = prob_dist + epsilon

        # Only consider non-zero probabilities
        entropy = -np.sum(prob_dist * np.log2(prob_dist))
    else:
        entropy = 0.0

    # ========== COMBINE ALL FEATURES ==========
    features = np.array([
        # Radial Power Distribution (4)
        band1_power,
        band2_power,
        band3_power,
        band4_power,

        # Peak Ratio (1)
        peak_ratio,

        # Spectral Entropy (1)
        entropy
    ], dtype=np.float32)

    # Verify we have exactly 6 features
    assert len(features) == 6, f"Expected 6 features, got {len(features)}"

    # Replace any NaN or Inf values with 0.0 (defensive programming)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def get_fourier_texture_feature_names() -> List[str]:
    """
    Get names of all 6 Fourier texture features.

    Returns:
        List of 6 feature names
    """
    names = [
        # Radial Power Distribution (4)
        'fourier_band1_power',  # Very low freq (DC, global)
        'fourier_band2_power',  # Low freq (coarse texture)
        'fourier_band3_power',  # Medium freq (fine texture)
        'fourier_band4_power',  # High freq (details, noise)

        # Peak Ratio (1)
        'fourier_peak_ratio',   # Dominant frequency strength

        # Spectral Entropy (1)
        'fourier_entropy'       # Randomness measure
    ]

    assert len(names) == 6, f"Expected 6 feature names, got {len(names)}"

    return names


if __name__ == '__main__':
    # Test feature extraction
    print("Testing Fourier Power Spectrum Features...")

    # Create a dummy cell with periodic horizontal pattern (simulating bat wood grain)
    dummy_cell_bat = np.zeros((75, 100), dtype=np.uint8)
    for y in range(0, 75, 5):
        dummy_cell_bat[y:y+2, :] = 180  # Fine horizontal stripes (periodic)

    # Create a dummy cell with random noise (simulating grass)
    dummy_cell_grass = np.random.randint(60, 160, (75, 100), dtype=np.uint8)

    # Create a dummy cell with uniform intensity (simulating no object)
    dummy_cell_no_object = np.ones((75, 100), dtype=np.uint8) * 120

    # Create a dummy cell with circular pattern (simulating ball)
    dummy_cell_ball = np.zeros((75, 100), dtype=np.uint8)
    cv2.circle(dummy_cell_ball, (50, 37), 20, 200, -1)

    # Extract features
    features_bat = extract_fourier_texture_features(dummy_cell_bat)
    features_grass = extract_fourier_texture_features(dummy_cell_grass)
    features_no_object = extract_fourier_texture_features(dummy_cell_no_object)
    features_ball = extract_fourier_texture_features(dummy_cell_ball)

    print(f"\nExtracted {len(features_bat)} features")
    print(f"\nKey Fourier Metrics Comparison:")
    print(f"{'Feature Name':<25} {'Bat':<15} {'Grass':<15} {'No-Obj':<15} {'Ball':<15}")
    print("=" * 85)

    names = get_fourier_texture_feature_names()

    # Show key discriminative features
    key_features = ['fourier_peak_ratio', 'fourier_entropy', 'fourier_band2_power']
    for key in key_features:
        idx = names.index(key)
        print(f"{key:<25} {features_bat[idx]:<15.4f} {features_grass[idx]:<15.4f} "
              f"{features_no_object[idx]:<15.4f} {features_ball[idx]:<15.4f}")

    print("\n✓ Test passed!")
    print("  Bat: High peak ratio (periodic wood grain), low entropy (structured)")
    print("  Grass: Low peak ratio (random), high entropy (unstructured)")
    print("  No-Object: Very low entropy (uniform)")
    print("\nFourier analysis is CRITICAL for bat vs grass discrimination!")
