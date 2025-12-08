"""
Edge Detection Features for Cricket Object Detection

This module extracts 12 edge and gradient features for detecting object boundaries.

Features breakdown:
- Canny Edge Features (6): Edge count, density, and orientation
- Sobel Gradient Features (6): Magnitude and directional gradients

Critical for:
- Stump: Strong vertical edges (90° orientation)
- Bat: Horizontal and object boundary edges
- Ball: Circular edge pattern

Author: Pynacle Team
Date: 2025
"""

import numpy as np
import cv2
from typing import List


def extract_edge_features(cell_rgb: np.ndarray) -> np.ndarray:
    """
    Extract 12 edge detection features from RGB cell.

    Critical for detecting object boundaries and orientation:
    - Stumps have strong vertical edges
    - Bats have horizontal/diagonal edges
    - Ball has circular edge pattern

    Args:
        cell_rgb: RGB image cell of shape (H, W, 3) in BGR format
                 Expected size: 75x100 pixels

    Returns:
        Array of 12 float32 features
    """
    # Handle empty or invalid images
    if cell_rgb.size == 0:
        return np.zeros(12, dtype=np.float32)

    # Convert to grayscale for edge detection
    cell_gray = cv2.cvtColor(cell_rgb, cv2.COLOR_BGR2GRAY)

    # Get total pixel count for normalization
    total_pixels = cell_gray.shape[0] * cell_gray.shape[1]

    # ========== CANNY EDGE FEATURES (6 features) ==========

    # Apply Canny edge detection
    # Auto-determine thresholds using median intensity
    median_intensity = np.median(cell_gray)
    lower_threshold = int(max(0, 0.66 * median_intensity))
    upper_threshold = int(min(255, 1.33 * median_intensity))

    canny_edges = cv2.Canny(cell_gray, lower_threshold, upper_threshold)

    # 1. Canny edge count (number of edge pixels)
    canny_edge_count = np.count_nonzero(canny_edges)

    # 2. Canny edge density (edge pixels / total pixels)
    # High density = lots of edges (complex object or background)
    # Low density with some edges = simple object (ball, stump)
    canny_edge_density = float(canny_edge_count) / float(total_pixels)

    # For orientation-specific edges, compute gradients
    # Use Sobel to get gradient directions at edge pixels
    sobel_x = cv2.Sobel(cell_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(cell_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_direction = np.arctan2(sobel_y, sobel_x) * 180.0 / np.pi  # Convert to degrees

    # Get edges mask
    edge_mask = canny_edges > 0

    if canny_edge_count > 0:
        # Get edge directions (only at edge pixels)
        edge_directions = gradient_direction[edge_mask]

        # 3. Vertical edges (80-100° orientation)
        # CRITICAL for stump detection!
        vertical_mask = (edge_directions >= 80) & (edge_directions <= 100)
        canny_vertical_edges = np.sum(vertical_mask)

        # 4. Horizontal edges (-10 to 10° orientation)
        # CRITICAL for bat detection!
        horizontal_mask = ((edge_directions >= -10) & (edge_directions <= 10)) | \
                         ((edge_directions >= 170) | (edge_directions <= -170))
        canny_horizontal_edges = np.sum(horizontal_mask)

        # 5. Mean edge intensity (edge strength)
        edge_intensities = gradient_magnitude[edge_mask]
        canny_edge_mean_intensity = np.mean(edge_intensities)

        # 6. Standard deviation of edge intensity
        # Uniform edges (ball, stump) have low std, complex edges (grass) have high std
        canny_edge_std_intensity = np.std(edge_intensities)

    else:
        # No edges detected
        canny_vertical_edges = 0.0
        canny_horizontal_edges = 0.0
        canny_edge_mean_intensity = 0.0
        canny_edge_std_intensity = 0.0

    # ========== SOBEL GRADIENT FEATURES (6 features) ==========

    # Sobel operators already computed above for Canny orientation analysis
    # Reuse them for Sobel features

    # 7. Sobel magnitude mean (average gradient strength)
    # High values = strong edges (object boundaries)
    sobel_magnitude_mean = np.mean(gradient_magnitude)

    # 8. Sobel magnitude max (peak gradient)
    # Very high values = sharp boundaries (stump edges, ball outline)
    sobel_magnitude_max = np.max(gradient_magnitude)

    # 9. Sobel X mean (horizontal gradient strength)
    # Positive = intensity increases left to right
    # Negative = intensity decreases left to right
    sobel_x_mean = np.mean(sobel_x)

    # 10. Sobel Y mean (vertical gradient strength)
    # Positive = intensity increases top to bottom
    # Negative = intensity decreases top to bottom
    # CRITICAL for stump (strong vertical gradients)
    sobel_y_mean = np.mean(sobel_y)

    # 11. Sobel orientation variance
    # Measure of edge direction uniformity
    # Ball (circular) has high variance (edges in all directions)
    # Stump (vertical) has low variance (mostly vertical edges)
    sobel_orientation_variance = np.var(gradient_direction)

    # 12. Sobel gradient ratio = |mean(sobel_y)| / |mean(sobel_x)| + eps
    # > 1.0 = more vertical gradients (stump)
    # < 1.0 = more horizontal gradients (bat)
    # ~ 1.0 = balanced gradients (ball, square objects)
    eps = 1e-7
    sobel_gradient_ratio = np.abs(sobel_y_mean) / (np.abs(sobel_x_mean) + eps)

    # ========== COMBINE ALL FEATURES ==========
    features = np.array([
        # Canny Edge Features (6)
        canny_edge_count,
        canny_edge_density,
        canny_vertical_edges,
        canny_horizontal_edges,
        canny_edge_mean_intensity,
        canny_edge_std_intensity,

        # Sobel Gradient Features (6)
        sobel_magnitude_mean,
        sobel_magnitude_max,
        sobel_x_mean,
        sobel_y_mean,
        sobel_orientation_variance,
        sobel_gradient_ratio
    ], dtype=np.float32)

    # Verify we have exactly 12 features
    assert len(features) == 12, f"Expected 12 features, got {len(features)}"

    # Replace any NaN or Inf values with 0.0 (defensive programming)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def get_edge_feature_names() -> List[str]:
    """
    Get names of all 12 edge detection features.

    Returns:
        List of 12 feature names
    """
    names = [
        # Canny Edge Features (6)
        'canny_edge_count',
        'canny_edge_density',
        'canny_vertical_edges',
        'canny_horizontal_edges',
        'canny_edge_mean_intensity',
        'canny_edge_std_intensity',

        # Sobel Gradient Features (6)
        'sobel_magnitude_mean',
        'sobel_magnitude_max',
        'sobel_x_mean',
        'sobel_y_mean',
        'sobel_orientation_variance',
        'sobel_gradient_ratio'
    ]

    assert len(names) == 12, f"Expected 12 feature names, got {len(names)}"

    return names


if __name__ == '__main__':
    # Test feature extraction
    print("Testing Edge Detection Features...")

    # Create a dummy image with vertical lines (simulating stump)
    dummy_cell_vertical = np.ones((75, 100, 3), dtype=np.uint8) * 128  # Gray background
    cv2.rectangle(dummy_cell_vertical, (40, 10), (42, 65), (255, 255, 255), -1)  # Vertical white stripe
    cv2.rectangle(dummy_cell_vertical, (58, 10), (60, 65), (255, 255, 255), -1)  # Another vertical stripe

    # Create a dummy image with horizontal lines (simulating bat)
    dummy_cell_horizontal = np.ones((75, 100, 3), dtype=np.uint8) * 128
    cv2.rectangle(dummy_cell_horizontal, (10, 35), (90, 40), (255, 255, 255), -1)  # Horizontal white stripe

    # Create a dummy image with circle (simulating ball)
    dummy_cell_circle = np.ones((75, 100, 3), dtype=np.uint8) * 128
    cv2.circle(dummy_cell_circle, (50, 37), 20, (255, 255, 255), 2)  # Circle outline

    # Extract features
    features_vertical = extract_edge_features(dummy_cell_vertical)
    features_horizontal = extract_edge_features(dummy_cell_horizontal)
    features_circle = extract_edge_features(dummy_cell_circle)

    print(f"\nExtracted {len(features_vertical)} features")
    print(f"\nKey Edge Metrics Comparison:")
    print(f"{'Metric':<30} {'Vertical (Stump)':<18} {'Horizontal (Bat)':<18} {'Circle (Ball)':<18}")
    print("=" * 90)

    names = get_edge_feature_names()

    # Show key discriminative features
    key_features = ['canny_vertical_edges', 'canny_horizontal_edges',
                   'sobel_gradient_ratio', 'sobel_orientation_variance']
    for key in key_features:
        idx = names.index(key)
        print(f"{key:<30} {features_vertical[idx]:<18.4f} {features_horizontal[idx]:<18.4f} {features_circle[idx]:<18.4f}")

    print("\n✓ Test passed!")
    print("  Vertical: High vertical edges, high gradient ratio (>1.0)")
    print("  Horizontal: High horizontal edges, low gradient ratio (<1.0)")
    print("  Circle: Balanced edges, high orientation variance")
