"""
Shape-Based Features for Cricket Object Detection

This module extracts 24 shape and contour features specifically designed for
discriminating between ball (circular), bat (elongated), and stump (vertical).

Features breakdown:
- Contour Analysis (12): Shape properties from largest contour
- Hu Moments (7): Scale/rotation-invariant shape descriptors
- Bounding Box (5): Aspect ratio and spatial properties

Author: Pynacle Team
Date: 2025
"""

import numpy as np
import cv2
from typing import List


def extract_shape_features(cell_rgb: np.ndarray) -> np.ndarray:
    """
    Extract 24 shape-based features from RGB cell.

    Critical for object discrimination:
    - Ball: Circularity ~ 1.0, Aspect ratio ~ 1.0
    - Bat: Low circularity, Aspect ratio > 2.0 (elongated horizontally)
    - Stump: Low circularity, Aspect ratio < 0.5 (elongated vertically)

    Args:
        cell_rgb: RGB image cell of shape (H, W, 3) in BGR format
                 Expected size: 75x100 pixels

    Returns:
        Array of 24 float32 features
    """
    # Handle empty or invalid images
    if cell_rgb.size == 0:
        return np.zeros(24, dtype=np.float32)

    # Convert to grayscale for contour detection
    cell_gray = cv2.cvtColor(cell_rgb, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding to detect objects
    # Use Otsu's method for automatic threshold determination
    _, binary = cv2.threshold(cell_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ========== CONTOUR ANALYSIS FEATURES (12 features) ==========

    if len(contours) > 0:
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_contour = contours[0]

        # 1. Number of contours (object count in cell)
        num_contours = len(contours)

        # 2. Largest contour area
        largest_contour_area = cv2.contourArea(largest_contour)

        # 3. Largest contour perimeter
        largest_contour_perimeter = cv2.arcLength(largest_contour, True)

        # 4. Circularity = 4π × area / perimeter²
        # Perfect circle = 1.0, elongated shapes < 0.5
        # CRITICAL for ball detection!
        if largest_contour_perimeter > 0:
            circularity = (4.0 * np.pi * largest_contour_area) / (largest_contour_perimeter ** 2)
            circularity = min(circularity, 1.0)  # Cap at 1.0 due to discretization
        else:
            circularity = 0.0

        # 5. Aspect ratio = width / height
        # Ball ~1.0, Horizontal bat >2.0, Vertical stump <0.5
        # CRITICAL for bat/stump discrimination!
        x, y, w, h = cv2.boundingRect(largest_contour)
        if h > 0:
            aspect_ratio = float(w) / float(h)
        else:
            aspect_ratio = 0.0

        # 6. Extent = contour_area / bounding_box_area
        # How much of bounding box is filled (compactness)
        bbox_area = w * h
        if bbox_area > 0:
            extent = largest_contour_area / bbox_area
        else:
            extent = 0.0

        # 7. Solidity = contour_area / convex_hull_area
        # Measures concavity (ball is convex, solidity~1.0)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = largest_contour_area / hull_area
        else:
            solidity = 0.0

        # 8. Contour compactness (another circularity measure)
        # = perimeter² / area (inverse of circularity, normalized)
        if largest_contour_area > 0:
            compactness = (largest_contour_perimeter ** 2) / (4.0 * np.pi * largest_contour_area)
        else:
            compactness = 0.0

        # 9. Rectangularity = contour_area / bounding_rectangle_area
        # Same as extent but computed differently
        min_rect = cv2.minAreaRect(largest_contour)
        min_rect_area = min_rect[1][0] * min_rect[1][1]
        if min_rect_area > 0:
            rectangularity = largest_contour_area / min_rect_area
        else:
            rectangularity = 0.0

        # 10. Eccentricity (elongation measure from ellipse fit)
        # 0 = circle, approaching 1 = line
        if len(largest_contour) >= 5:  # Need at least 5 points to fit ellipse
            try:
                ellipse = cv2.fitEllipse(largest_contour)
                major_axis = max(ellipse[1])
                minor_axis = min(ellipse[1])
                if major_axis > 0:
                    eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
                else:
                    eccentricity = 0.0
            except:
                eccentricity = 0.0
        else:
            eccentricity = 0.0

        # 11. Convex hull vertices count
        # More vertices = more complex shape
        convex_hull_vertices_count = len(hull)

        # 12. Contour orientation (angle of principal axis)
        # 0° = horizontal, 90° = vertical
        # CRITICAL for bat (horizontal) vs stump (vertical) discrimination!
        if len(largest_contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(largest_contour)
                contour_orientation = ellipse[2]  # Angle in degrees
            except:
                contour_orientation = 0.0
        else:
            contour_orientation = 0.0

    else:
        # No contours found - all features = 0
        num_contours = 0
        largest_contour_area = 0.0
        largest_contour_perimeter = 0.0
        circularity = 0.0
        aspect_ratio = 0.0
        extent = 0.0
        solidity = 0.0
        compactness = 0.0
        rectangularity = 0.0
        eccentricity = 0.0
        convex_hull_vertices_count = 0
        contour_orientation = 0.0

    # ========== HU MOMENTS (7 features) ==========
    # Scale, rotation, and translation invariant shape descriptors
    # Excellent for distinguishing ball vs bat vs stump shapes

    moments = cv2.moments(cell_gray)
    hu_moments = cv2.HuMoments(moments).flatten()

    # Hu moments can have very large/small values, apply log transform for stability
    # Use signed log to preserve sign information
    hu_moments_log = np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

    # Extract 7 Hu moment features
    hu_1 = hu_moments_log[0]
    hu_2 = hu_moments_log[1]
    hu_3 = hu_moments_log[2]
    hu_4 = hu_moments_log[3]
    hu_5 = hu_moments_log[4]
    hu_6 = hu_moments_log[5]
    hu_7 = hu_moments_log[6]

    # ========== BOUNDING BOX FEATURES (5 features) ==========

    if len(contours) > 0:
        # Use largest contour bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)

        # 20. Bounding box width
        bbox_width = float(w)

        # 21. Bounding box height
        bbox_height = float(h)

        # 22. Bounding box aspect ratio (same as #5 but separate for clarity)
        if h > 0:
            bbox_aspect_ratio = float(w) / float(h)
        else:
            bbox_aspect_ratio = 0.0

        # 23. Bounding box area
        bbox_area = float(w * h)

        # 24. Bounding box fill ratio = contour_area / bbox_area
        # Same as extent (#6) - how much of bbox is filled
        if bbox_area > 0:
            bbox_fill_ratio = largest_contour_area / bbox_area
        else:
            bbox_fill_ratio = 0.0

    else:
        # No contours - default to cell dimensions
        bbox_width = 100.0  # Cell width
        bbox_height = 75.0  # Cell height
        bbox_aspect_ratio = 100.0 / 75.0
        bbox_area = 100.0 * 75.0
        bbox_fill_ratio = 0.0

    # ========== COMBINE ALL FEATURES ==========
    features = np.array([
        # Contour Analysis (12)
        num_contours,
        largest_contour_area,
        largest_contour_perimeter,
        circularity,
        aspect_ratio,
        extent,
        solidity,
        compactness,
        rectangularity,
        eccentricity,
        convex_hull_vertices_count,
        contour_orientation,

        # Hu Moments (7)
        hu_1, hu_2, hu_3, hu_4, hu_5, hu_6, hu_7,

        # Bounding Box (5)
        bbox_width,
        bbox_height,
        bbox_aspect_ratio,
        bbox_area,
        bbox_fill_ratio
    ], dtype=np.float32)

    # Verify we have exactly 24 features
    assert len(features) == 24, f"Expected 24 features, got {len(features)}"

    # Replace any NaN or Inf values with 0.0 (defensive programming)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def get_shape_feature_names() -> List[str]:
    """
    Get names of all 24 shape features.

    Returns:
        List of 24 feature names
    """
    names = [
        # Contour Analysis (12)
        'num_contours',
        'largest_contour_area',
        'largest_contour_perimeter',
        'circularity',
        'aspect_ratio',
        'extent',
        'solidity',
        'compactness',
        'rectangularity',
        'eccentricity',
        'convex_hull_vertices_count',
        'contour_orientation',

        # Hu Moments (7)
        'hu_moment_1',
        'hu_moment_2',
        'hu_moment_3',
        'hu_moment_4',
        'hu_moment_5',
        'hu_moment_6',
        'hu_moment_7',

        # Bounding Box (5)
        'bbox_width',
        'bbox_height',
        'bbox_aspect_ratio',
        'bbox_area',
        'bbox_fill_ratio'
    ]

    assert len(names) == 24, f"Expected 24 feature names, got {len(names)}"

    return names


if __name__ == '__main__':
    # Test feature extraction
    print("Testing Shape Features...")

    # Create a dummy circular object (simulating cricket ball)
    dummy_cell_circle = np.zeros((75, 100, 3), dtype=np.uint8)
    cv2.circle(dummy_cell_circle, (50, 37), 20, (255, 255, 255), -1)

    # Create a dummy horizontal elongated object (simulating bat)
    dummy_cell_horizontal = np.zeros((75, 100, 3), dtype=np.uint8)
    cv2.rectangle(dummy_cell_horizontal, (10, 30), (90, 45), (255, 255, 255), -1)

    # Create a dummy vertical elongated object (simulating stump)
    dummy_cell_vertical = np.zeros((75, 100, 3), dtype=np.uint8)
    cv2.rectangle(dummy_cell_vertical, (40, 10), (60, 65), (255, 255, 255), -1)

    # Extract features
    features_circle = extract_shape_features(dummy_cell_circle)
    features_horizontal = extract_shape_features(dummy_cell_horizontal)
    features_vertical = extract_shape_features(dummy_cell_vertical)

    print(f"\nExtracted {len(features_circle)} features")
    print(f"\nKey Shape Metrics Comparison:")
    print(f"{'Metric':<25} {'Circle (Ball)':<15} {'Horizontal (Bat)':<18} {'Vertical (Stump)':<18}")
    print("=" * 80)

    names = get_shape_feature_names()

    # Show key discriminative features
    key_features = ['circularity', 'aspect_ratio', 'eccentricity', 'contour_orientation']
    for key in key_features:
        idx = names.index(key)
        print(f"{key:<25} {features_circle[idx]:<15.4f} {features_horizontal[idx]:<18.4f} {features_vertical[idx]:<18.4f}")

    print("\n✓ Test passed!")
    print("  Circle: High circularity (~1.0), aspect ratio ~1.0")
    print("  Horizontal: Low circularity, aspect ratio >2.0 (wide)")
    print("  Vertical: Low circularity, aspect ratio <0.5 (tall)")
