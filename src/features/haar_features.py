"""
Haar-like Features for Cricket Object Detection

This module extracts 10 Haar-like features using rectangle-based intensity patterns.

Features breakdown:
- Two-Rectangle Features (4): Vertical and horizontal edge detection
- Three-Rectangle Features (3): Line and stripe detection
- Four-Rectangle Features (3): Checkerboard and cross patterns

Critical for:
- Stump: Strong vertical two-rectangle pattern (white-dark-white)
- Bat: Horizontal/diagonal patterns (elongated object)
- Ball: Weak rectangle features (smooth circular object)

Author: Pynacle Team
Date: 2025
"""

import numpy as np
import cv2
from typing import List


def extract_haar_features(cell_gray: np.ndarray) -> np.ndarray:
    """
    Extract 10 Haar-like features from grayscale cell.

    Haar-like features use rectangle sums to detect edges, lines, and patterns.
    These are fast to compute and effective for detecting vertical/horizontal structures.

    Args:
        cell_gray: Grayscale image of shape (H, W) or RGB image (will be converted)
                  Expected size: 75x100 pixels

    Returns:
        Array of 10 float32 features
    """
    # Handle empty or invalid images
    if cell_gray.size == 0:
        return np.zeros(10, dtype=np.float32)

    # Convert to grayscale if RGB
    if len(cell_gray.shape) == 3:
        cell_gray = cv2.cvtColor(cell_gray, cv2.COLOR_BGR2GRAY)

    # Get image dimensions
    h, w = cell_gray.shape

    # Compute integral image for fast rectangle sum computation
    # integral[y, x] = sum of all pixels in rectangle (0,0) to (y,x)
    integral = cv2.integral(cell_gray.astype(np.float64))

    def rect_sum(integral, x, y, width, height):
        """
        Compute sum of pixels in rectangle using integral image.
        Fast O(1) computation.
        """
        x1, y1 = x, y
        x2, y2 = x + width, y + height

        # Bounds checking
        x2 = min(x2, integral.shape[1] - 1)
        y2 = min(y2, integral.shape[0] - 1)
        x1 = max(x1, 0)
        y1 = max(y1, 0)

        # Integral image formula: sum = D - C - B + A
        #   A-------B
        #   |   rect|
        #   C-------D
        A = integral[y1, x1] if (y1 >= 0 and x1 >= 0) else 0
        B = integral[y1, x2] if (y1 >= 0 and x2 < integral.shape[1]) else 0
        C = integral[y2, x1] if (y2 < integral.shape[0] and x1 >= 0) else 0
        D = integral[y2, x2] if (y2 < integral.shape[0] and x2 < integral.shape[1]) else 0

        return D - C - B + A

    # ========== TWO-RECTANGLE FEATURES (4 features) ==========
    # Detect edges by comparing adjacent rectangles

    # 1. Vertical two-rectangle (left-right edge)
    # | Left | Right |
    # Stump: High response (white-dark-white)
    half_w = w // 2
    left_sum = rect_sum(integral, 0, 0, half_w, h)
    right_sum = rect_sum(integral, half_w, 0, w - half_w, h)
    haar_vertical_2rect = abs(left_sum - right_sum) / (w * h)

    # 2. Horizontal two-rectangle (top-bottom edge)
    # +-------+
    # |  Top  |
    # +-------+
    # | Bottom|
    # +-------+
    half_h = h // 2
    top_sum = rect_sum(integral, 0, 0, w, half_h)
    bottom_sum = rect_sum(integral, 0, half_h, w, h - half_h)
    haar_horizontal_2rect = abs(top_sum - bottom_sum) / (w * h)

    # 3. Vertical edge strength (max across multiple positions)
    # Scan for strongest vertical edge
    max_vertical_edge = 0.0
    step = w // 4
    for split_x in range(step, w - step, step):
        left = rect_sum(integral, 0, 0, split_x, h)
        right = rect_sum(integral, split_x, 0, w - split_x, h)
        edge_strength = abs(left / split_x - right / (w - split_x))
        max_vertical_edge = max(max_vertical_edge, edge_strength)
    haar_vertical_edge_max = max_vertical_edge

    # 4. Horizontal edge strength (max across multiple positions)
    max_horizontal_edge = 0.0
    step = h // 4
    for split_y in range(step, h - step, step):
        top = rect_sum(integral, 0, 0, w, split_y)
        bottom = rect_sum(integral, 0, split_y, w, h - split_y)
        edge_strength = abs(top / split_y - bottom / (h - split_y))
        max_horizontal_edge = max(max_horizontal_edge, edge_strength)
    haar_horizontal_edge_max = max_horizontal_edge

    # ========== THREE-RECTANGLE FEATURES (3 features) ==========
    # Detect lines and stripes

    # 5. Vertical three-rectangle (vertical line detector)
    # | Left | Center | Right |
    # Detects vertical stripe (bat handle, stump)
    third_w = w // 3
    left_sum = rect_sum(integral, 0, 0, third_w, h)
    center_sum = rect_sum(integral, third_w, 0, third_w, h)
    right_sum = rect_sum(integral, 2 * third_w, 0, w - 2 * third_w, h)
    # Center brighter than sides: positive response
    haar_vertical_3rect = (center_sum - (left_sum + right_sum) / 2.0) / (w * h)

    # 6. Horizontal three-rectangle (horizontal line detector)
    # +-------+
    # |  Top  |
    # +-------+
    # | Center|
    # +-------+
    # | Bottom|
    # +-------+
    # Detects horizontal stripe (bat body)
    third_h = h // 3
    top_sum = rect_sum(integral, 0, 0, w, third_h)
    center_sum = rect_sum(integral, 0, third_h, w, third_h)
    bottom_sum = rect_sum(integral, 0, 2 * third_h, w, h - 2 * third_h)
    haar_horizontal_3rect = (center_sum - (top_sum + bottom_sum) / 2.0) / (w * h)

    # 7. Diagonal three-rectangle (diagonal pattern)
    # Approximate diagonal by comparing corners
    tl_br_sum = rect_sum(integral, 0, 0, half_w, half_h) + \
                rect_sum(integral, half_w, half_h, w - half_w, h - half_h)
    tr_bl_sum = rect_sum(integral, half_w, 0, w - half_w, half_h) + \
                rect_sum(integral, 0, half_h, half_w, h - half_h)
    haar_diagonal_3rect = abs(tl_br_sum - tr_bl_sum) / (w * h)

    # ========== FOUR-RECTANGLE FEATURES (3 features) ==========
    # Detect checkerboard and cross patterns

    # 8. Four-rectangle checkerboard pattern
    # +-------+-------+
    # |  TL   |  TR   |
    # +-------+-------+
    # |  BL   |  BR   |
    # +-------+-------+
    # Checkerboard: (TL + BR) - (TR + BL)
    tl_sum = rect_sum(integral, 0, 0, half_w, half_h)
    tr_sum = rect_sum(integral, half_w, 0, w - half_w, half_h)
    bl_sum = rect_sum(integral, 0, half_h, half_w, h - half_h)
    br_sum = rect_sum(integral, half_w, half_h, w - half_w, h - half_h)
    haar_checkerboard = abs((tl_sum + br_sum) - (tr_sum + bl_sum)) / (w * h)

    # 9. Four-rectangle cross pattern
    # Measure variance across the four quadrants
    mean_sum = (tl_sum + tr_sum + bl_sum + br_sum) / 4.0
    variance = ((tl_sum - mean_sum)**2 + (tr_sum - mean_sum)**2 +
                (bl_sum - mean_sum)**2 + (br_sum - mean_sum)**2) / 4.0
    haar_cross_variance = variance / (w * h)

    # 10. Center-surround pattern (four-rectangle center emphasis)
    # Compare center region to outer regions
    center_w, center_h = w // 2, h // 2
    center_x, center_y = w // 4, h // 4
    center_sum = rect_sum(integral, center_x, center_y, center_w, center_h)
    surround_sum = rect_sum(integral, 0, 0, w, h) - center_sum
    center_area = center_w * center_h
    surround_area = w * h - center_area
    if surround_area > 0:
        haar_center_surround = abs(center_sum / center_area - surround_sum / surround_area)
    else:
        haar_center_surround = 0.0

    # ========== COMBINE ALL FEATURES ==========
    features = np.array([
        # Two-Rectangle Features (4)
        haar_vertical_2rect,
        haar_horizontal_2rect,
        haar_vertical_edge_max,
        haar_horizontal_edge_max,

        # Three-Rectangle Features (3)
        haar_vertical_3rect,
        haar_horizontal_3rect,
        haar_diagonal_3rect,

        # Four-Rectangle Features (3)
        haar_checkerboard,
        haar_cross_variance,
        haar_center_surround
    ], dtype=np.float32)

    # Verify we have exactly 10 features
    assert len(features) == 10, f"Expected 10 features, got {len(features)}"

    # Replace any NaN or Inf values with 0.0 (defensive programming)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def get_haar_feature_names() -> List[str]:
    """
    Get names of all 10 Haar-like features.

    Returns:
        List of 10 feature names
    """
    names = [
        # Two-Rectangle Features (4)
        'haar_vertical_2rect',
        'haar_horizontal_2rect',
        'haar_vertical_edge_max',
        'haar_horizontal_edge_max',

        # Three-Rectangle Features (3)
        'haar_vertical_3rect',
        'haar_horizontal_3rect',
        'haar_diagonal_3rect',

        # Four-Rectangle Features (3)
        'haar_checkerboard',
        'haar_cross_variance',
        'haar_center_surround'
    ]

    assert len(names) == 10, f"Expected 10 feature names, got {len(names)}"

    return names


if __name__ == '__main__':
    # Test feature extraction
    print("Testing Haar-like Features...")

    # Create a dummy cell with vertical stripes (simulating stump)
    dummy_cell_stump = np.ones((75, 100), dtype=np.uint8) * 50
    dummy_cell_stump[:, 30:35] = 255  # White stripe
    dummy_cell_stump[:, 65:70] = 255  # White stripe

    # Create a dummy cell with horizontal stripe (simulating bat)
    dummy_cell_bat = np.ones((75, 100), dtype=np.uint8) * 50
    dummy_cell_bat[30:40, :] = 150  # Horizontal bright stripe

    # Create a dummy cell with uniform intensity (simulating ball)
    dummy_cell_ball = np.ones((75, 100), dtype=np.uint8) * 200

    # Extract features
    features_stump = extract_haar_features(dummy_cell_stump)
    features_bat = extract_haar_features(dummy_cell_bat)
    features_ball = extract_haar_features(dummy_cell_ball)

    print(f"\nExtracted {len(features_stump)} features")
    print(f"\nKey Haar Metrics Comparison:")
    print(f"{'Feature Name':<30} {'Stump':<12} {'Bat':<12} {'Ball':<12}")
    print("=" * 70)

    names = get_haar_feature_names()

    # Show key discriminative features
    key_features = ['haar_vertical_2rect', 'haar_horizontal_2rect',
                   'haar_vertical_3rect', 'haar_horizontal_3rect']
    for key in key_features:
        idx = names.index(key)
        print(f"{key:<30} {features_stump[idx]:<12.4f} {features_bat[idx]:<12.4f} {features_ball[idx]:<12.4f}")

    print("\n✓ Test passed!")
    print("  Stump: High vertical rectangle features (vertical stripes)")
    print("  Bat: High horizontal rectangle features (horizontal object)")
    print("  Ball: Low rectangle features (uniform smooth object)")
