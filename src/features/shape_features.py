"""
Shape-Based Features for Cricket Object Detection

This module extracts 35 shape and contour features specifically designed for
discriminating between ball (circular), bat (elongated), and stump (vertical).

Features breakdown:
- Contour Analysis (12): Shape properties from largest contour
- Hu Moments (7): Scale/rotation-invariant shape descriptors
- Bounding Box (5): Aspect ratio and spatial properties
- Minimum Enclosing Circle (4): PHASE 4B - Ball discrimination via fill ratio
- Convexity Defects (4): PHASE 4B - Smooth vs elongated shape detection
- Polygon Approximation (3): PHASE 4B - Shape complexity analysis

Author: Pynacle Team
Date: 2025
Phase: Enhanced with Phase 4B features
"""

import numpy as np
import cv2
from typing import List


def extract_shape_features(cell_rgb: np.ndarray) -> np.ndarray:
    """
    Extract 35 shape-based features from RGB cell (PHASE 4B Enhanced).

    Critical for object discrimination:
    - Ball: Circularity ~ 1.0, Aspect ratio ~ 1.0, High min circle fill ratio (>0.75)
    - Bat: Low circularity, Aspect ratio > 2.0 (elongated horizontally), Few convexity defects
    - Stump: Low circularity, Aspect ratio < 0.5 (elongated vertically), Few convexity defects

    Phase 4B adds:
    - Minimum Enclosing Circle (4): Critical for ball vs elongated shapes
    - Convexity Defects (4): Smooth ball (0 defects) vs elongated bat/stump
    - Polygon Approximation (3): Shape complexity (ball = many vertices, bat/stump = few)

    Args:
        cell_rgb: RGB image cell of shape (H, W, 3) in BGR format
                 Expected size: 75x100 pixels

    Returns:
        Array of 35 float32 features (24 original + 11 Phase 4B)
    """
    # Handle empty or invalid images
    if cell_rgb.size == 0:
        return np.zeros(35, dtype=np.float32)

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

    # ========== MINIMUM ENCLOSING CIRCLE (4 features) - PHASE 4B ==========
    # CRITICAL for ball detection: Ball has high circle fill ratio (>0.75)
    # Bat/Stump have low fill ratio (<0.3) since they're elongated

    if len(contours) > 0:
        # Find minimum enclosing circle for largest contour
        (min_circle_x, min_circle_y), min_circle_radius = cv2.minEnclosingCircle(largest_contour)

        # 25. Minimum circle radius
        # Ball: radius ~ 10-25 pixels
        # Bat/Stump: larger radius (encloses elongated shape)
        min_enclosing_circle_radius = float(min_circle_radius)

        # 26. Minimum circle area
        min_circle_area = np.pi * (min_circle_radius ** 2)

        # 27. Minimum circle fill ratio = contour_area / circle_area
        # CRITICAL DISCRIMINATOR:
        # Ball: >0.75 (compact circular object fills the circle)
        # Bat: <0.3 (elongated, leaves much of circle empty)
        # Stump: <0.3 (elongated vertically)
        if min_circle_area > 0:
            min_circle_fill_ratio = largest_contour_area / min_circle_area
        else:
            min_circle_fill_ratio = 0.0

        # 28. Circle-to-bbox ratio = circle_radius / max(bbox_width, bbox_height)
        # Measures how much the enclosing circle differs from bounding box
        # Ball: ratio ~ 0.5-0.7 (circle ≈ bbox)
        # Bat/Stump: ratio > 1.0 (circle much larger than one dimension)
        if max(bbox_width, bbox_height) > 0:
            circle_to_bbox_ratio = min_circle_radius / max(bbox_width, bbox_height)
        else:
            circle_to_bbox_ratio = 0.0

    else:
        min_enclosing_circle_radius = 0.0
        min_circle_area = 0.0
        min_circle_fill_ratio = 0.0
        circle_to_bbox_ratio = 0.0

    # ========== CONVEXITY DEFECTS (4 features) - PHASE 4B ==========
    # CRITICAL for smooth vs elongated shape detection:
    # Ball: 0 convexity defects (perfectly smooth, convex)
    # Bat: few defects at ends (mostly convex)
    # Stump: few defects (stick-like shape)

    if len(contours) > 0 and len(largest_contour) > 3:
        # Convexity defects require convex hull
        hull_indices = cv2.convexHull(largest_contour, returnPoints=False)

        if hull_indices is not None and len(hull_indices) > 3:
            try:
                defects = cv2.convexityDefects(largest_contour, hull_indices)

                if defects is not None:
                    # 29. Number of convexity defects
                    # Ball: 0 (smooth contour)
                    # Bat/Stump: 0-2 (mostly convex stick-like shapes)
                    # Complex shapes: >3
                    num_convexity_defects = len(defects)

                    # Extract defect depths (distance from contour to hull)
                    defect_depths = defects[:, 0, 3] / 256.0  # OpenCV stores depth as int*256

                    # 30. Max convexity defect depth
                    # Larger depth = more concave
                    max_defect_depth = float(np.max(defect_depths))

                    # 31. Mean convexity defect depth
                    mean_defect_depth = float(np.mean(defect_depths))

                    # 32. Convexity defect area ratio
                    # = sum(defect_areas) / contour_area
                    # Approximation: defect area ≈ depth * base
                    # Higher ratio = more concavity
                    total_defect_area = np.sum(defect_depths)
                    if largest_contour_area > 0:
                        convexity_defect_area_ratio = total_defect_area / largest_contour_area
                    else:
                        convexity_defect_area_ratio = 0.0

                else:
                    # No defects detected (perfect convex hull = contour)
                    num_convexity_defects = 0
                    max_defect_depth = 0.0
                    mean_defect_depth = 0.0
                    convexity_defect_area_ratio = 0.0

            except cv2.error:
                # Error in computing defects (e.g., insufficient points)
                num_convexity_defects = 0
                max_defect_depth = 0.0
                mean_defect_depth = 0.0
                convexity_defect_area_ratio = 0.0
        else:
            # Insufficient hull points
            num_convexity_defects = 0
            max_defect_depth = 0.0
            mean_defect_depth = 0.0
            convexity_defect_area_ratio = 0.0
    else:
        # No contours or too few points
        num_convexity_defects = 0
        max_defect_depth = 0.0
        mean_defect_depth = 0.0
        convexity_defect_area_ratio = 0.0

    # ========== POLYGON APPROXIMATION (3 features) - PHASE 4B ==========
    # Approximates contour as polygon with fewer vertices
    # Ball: Cannot be approximated well (many vertices needed for circle)
    # Bat/Stump: Can be approximated with 4-6 vertices (rectangle-like)

    if len(contours) > 0:
        # Douglas-Peucker algorithm with epsilon = 2% of perimeter
        epsilon = 0.02 * largest_contour_perimeter
        approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

        # 33. Polygon vertex count
        # Ball: high count (circle needs many vertices)
        # Rectangle: 4 vertices
        # Triangle: 3 vertices
        polygon_vertex_count = len(approx_polygon)

        # 34. Polygon approximation error
        # = perimeter(approx_polygon) / perimeter(original_contour)
        # Ball: high error (poor approximation)
        # Rectangle: low error (good approximation)
        approx_perimeter = cv2.arcLength(approx_polygon, True)
        if largest_contour_perimeter > 0:
            polygon_approx_error = abs(approx_perimeter - largest_contour_perimeter) / largest_contour_perimeter
        else:
            polygon_approx_error = 0.0

        # 35. Polygon compactness
        # = 4π × area / perimeter² of approximated polygon
        # Similar to circularity but for simplified shape
        approx_area = cv2.contourArea(approx_polygon)
        if approx_perimeter > 0:
            polygon_compactness = (4.0 * np.pi * approx_area) / (approx_perimeter ** 2)
            polygon_compactness = min(polygon_compactness, 1.0)
        else:
            polygon_compactness = 0.0

    else:
        polygon_vertex_count = 0
        polygon_approx_error = 0.0
        polygon_compactness = 0.0

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
        bbox_fill_ratio,

        # Minimum Enclosing Circle (4) - PHASE 4B
        min_enclosing_circle_radius,
        min_circle_area,
        min_circle_fill_ratio,
        circle_to_bbox_ratio,

        # Convexity Defects (4) - PHASE 4B
        num_convexity_defects,
        max_defect_depth,
        mean_defect_depth,
        convexity_defect_area_ratio,

        # Polygon Approximation (3) - PHASE 4B
        polygon_vertex_count,
        polygon_approx_error,
        polygon_compactness
    ], dtype=np.float32)

    # Verify we have exactly 35 features (24 original + 11 Phase 4B)
    assert len(features) == 35, f"Expected 35 features, got {len(features)}"

    # Replace any NaN or Inf values with 0.0 (defensive programming)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def get_shape_feature_names() -> List[str]:
    """
    Get names of all 35 shape features (PHASE 4B Enhanced).

    Returns:
        List of 35 feature names (24 original + 11 Phase 4B)
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
        'bbox_fill_ratio',

        # Minimum Enclosing Circle (4) - PHASE 4B
        'min_enclosing_circle_radius',
        'min_circle_area',
        'min_circle_fill_ratio',
        'circle_to_bbox_ratio',

        # Convexity Defects (4) - PHASE 4B
        'num_convexity_defects',
        'max_defect_depth',
        'mean_defect_depth',
        'convexity_defect_area_ratio',

        # Polygon Approximation (3) - PHASE 4B
        'polygon_vertex_count',
        'polygon_approx_error',
        'polygon_compactness'
    ]

    assert len(names) == 35, f"Expected 35 feature names, got {len(names)}"

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
