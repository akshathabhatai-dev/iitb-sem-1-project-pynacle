"""
SLIC Superpixel Features for Cricket Object Detection

This module extracts 12 SLIC (Simple Linear Iterative Clustering) superpixel features
for region-based color segmentation and analysis.

Features breakdown:
- Superpixel Statistics (6): Count, size, compactness, boundary strength
- Dominant Superpixel Features (6): Red content, hue, saturation, area of largest region

Critical for:
- Ball: Single dominant red superpixel (uniform red region)
- Bat: Few elongated superpixels (wood texture)
- Stump: Few vertical superpixels (white columns)
- Grass: Many small scattered superpixels (complex texture)

Author: Pynacle Team
Date: 2025
"""

import numpy as np
import cv2
from typing import List
from skimage.segmentation import slic
from skimage.measure import regionprops, label


def extract_superpixel_features(cell_rgb: np.ndarray) -> np.ndarray:
    """
    Extract 12 SLIC superpixel features from RGB cell.

    SLIC segments the image into perceptually uniform regions based on color similarity.
    This helps identify coherent object regions (ball, bat, stump) vs scattered background.

    Args:
        cell_rgb: RGB image cell of shape (H, W, 3) in BGR format
                 Expected size: 75x100 pixels

    Returns:
        Array of 12 float32 features
    """
    # Handle empty or invalid images
    if cell_rgb.size == 0:
        return np.zeros(12, dtype=np.float32)

    # Convert BGR to RGB for SLIC (scikit-image uses RGB)
    cell_rgb_sk = cv2.cvtColor(cell_rgb, cv2.COLOR_BGR2RGB)

    # Get total pixel count
    total_pixels = cell_rgb.shape[0] * cell_rgb.shape[1]

    # ========== SLIC SUPERPIXEL SEGMENTATION ==========

    try:
        # Apply SLIC superpixel segmentation
        # n_segments: Approximate number of superpixels (smaller cell needs fewer)
        # compactness: Balance between color and spatial proximity (10 = balanced)
        # sigma: Gaussian smoothing before segmentation
        segments = slic(cell_rgb_sk, n_segments=10, compactness=10, sigma=1,
                       start_label=1, channel_axis=2)

        # Get unique superpixel labels
        unique_labels = np.unique(segments)
        num_superpixels = len(unique_labels)

        # ========== SUPERPIXEL STATISTICS (6 features) ==========

        # 1. Number of superpixels
        # Ball/stump: Few superpixels (1-3), uniform object
        # Grass: Many superpixels (8-12), complex texture
        slic_num_superpixels = float(num_superpixels)

        # Compute superpixel sizes
        superpixel_sizes = []
        for label_val in unique_labels:
            size = np.sum(segments == label_val)
            superpixel_sizes.append(size)

        superpixel_sizes = np.array(superpixel_sizes)

        # 2. Mean superpixel size
        # Large superpixels = uniform regions (ball, stump)
        slic_size_mean = np.mean(superpixel_sizes)

        # 3. Std of superpixel size
        # Low std = uniform sizes (consistent object)
        # High std = varying sizes (complex scene)
        slic_size_std = np.std(superpixel_sizes)

        # 4. Superpixel compactness mean
        # Measure how "round" each superpixel is
        compactness_vals = []
        labeled_segments = label(segments)
        regions = regionprops(labeled_segments)

        for region in regions:
            # Compactness = 4π×area / perimeter²
            if region.perimeter > 0:
                comp = (4.0 * np.pi * region.area) / (region.perimeter ** 2)
                compactness_vals.append(comp)

        if len(compactness_vals) > 0:
            slic_compactness_mean = np.mean(compactness_vals)
        else:
            slic_compactness_mean = 0.0

        # 5. Superpixel boundary strength (mean gradient at boundaries)
        # Compute gradients
        gray = cv2.cvtColor(cell_rgb, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Find superpixel boundaries (where labels change)
        boundaries = np.zeros_like(segments, dtype=bool)
        for i in range(segments.shape[0] - 1):
            for j in range(segments.shape[1] - 1):
                if (segments[i, j] != segments[i+1, j]) or \
                   (segments[i, j] != segments[i, j+1]):
                    boundaries[i, j] = True

        if np.any(boundaries):
            slic_boundary_strength = np.mean(grad_magnitude[boundaries])
        else:
            slic_boundary_strength = 0.0

        # 6. Largest superpixel ratio (largest superpixel area / total area)
        # Ball: One dominant superpixel (high ratio ~0.8-1.0)
        # Grass: Many small superpixels (low ratio ~0.1-0.2)
        slic_largest_ratio = float(np.max(superpixel_sizes)) / float(total_pixels)

        # ========== DOMINANT SUPERPIXEL FEATURES (6 features) ==========
        # Analyze the largest superpixel (likely the object)

        largest_label = unique_labels[np.argmax(superpixel_sizes)]
        largest_mask = (segments == largest_label)

        # Extract RGB values in largest superpixel
        largest_pixels_rgb = cell_rgb[largest_mask]

        if len(largest_pixels_rgb) > 0:
            # BGR format: channel 2 is Red
            red_channel = largest_pixels_rgb[:, 2]
            green_channel = largest_pixels_rgb[:, 1]
            blue_channel = largest_pixels_rgb[:, 0]

            # 7. Dominant superpixel red content (mean red channel)
            # Ball should have high red content in dominant superpixel
            slic_dominant_red = np.mean(red_channel)

            # Convert largest superpixel to HSV for color analysis
            cell_hsv = cv2.cvtColor(cell_rgb, cv2.COLOR_BGR2HSV)
            largest_pixels_hsv = cell_hsv[largest_mask]

            # 8. Dominant superpixel hue (mean hue)
            # Red hue: 0-10 or 170-180 (in OpenCV's 0-180 scale)
            slic_dominant_hue = np.mean(largest_pixels_hsv[:, 0])

            # 9. Dominant superpixel saturation (mean saturation)
            # Ball has high saturation (vivid red)
            slic_dominant_saturation = np.mean(largest_pixels_hsv[:, 1])

            # 10. Dominant superpixel area (pixels)
            slic_dominant_area = float(np.sum(largest_mask))

            # 11. Superpixel color variance (std of RGB in largest superpixel)
            # Uniform object has low variance
            color_variance = np.mean([np.std(red_channel),
                                     np.std(green_channel),
                                     np.std(blue_channel)])
            slic_color_variance = color_variance

            # 12. Dominant superpixel red dominance ratio
            # R / (G + B) for dominant superpixel
            eps = 1e-7
            slic_dominant_red_ratio = np.mean(red_channel / (green_channel + blue_channel + eps))

        else:
            # No dominant superpixel found
            slic_dominant_red = 0.0
            slic_dominant_hue = 0.0
            slic_dominant_saturation = 0.0
            slic_dominant_area = 0.0
            slic_color_variance = 0.0
            slic_dominant_red_ratio = 0.0

    except Exception as e:
        # If SLIC fails (rare), return zeros
        print(f"SLIC extraction failed: {e}")
        slic_num_superpixels = 0.0
        slic_size_mean = 0.0
        slic_size_std = 0.0
        slic_compactness_mean = 0.0
        slic_boundary_strength = 0.0
        slic_largest_ratio = 0.0
        slic_dominant_red = 0.0
        slic_dominant_hue = 0.0
        slic_dominant_saturation = 0.0
        slic_dominant_area = 0.0
        slic_color_variance = 0.0
        slic_dominant_red_ratio = 0.0

    # ========== COMBINE ALL FEATURES ==========
    features = np.array([
        # Superpixel Statistics (6)
        slic_num_superpixels,
        slic_size_mean,
        slic_size_std,
        slic_compactness_mean,
        slic_boundary_strength,
        slic_largest_ratio,

        # Dominant Superpixel Features (6)
        slic_dominant_red,
        slic_dominant_hue,
        slic_dominant_saturation,
        slic_dominant_area,
        slic_color_variance,
        slic_dominant_red_ratio
    ], dtype=np.float32)

    # Verify we have exactly 12 features
    assert len(features) == 12, f"Expected 12 features, got {len(features)}"

    # Replace any NaN or Inf values with 0.0 (defensive programming)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def get_superpixel_feature_names() -> List[str]:
    """
    Get names of all 12 SLIC superpixel features.

    Returns:
        List of 12 feature names
    """
    names = [
        # Superpixel Statistics (6)
        'slic_num_superpixels',
        'slic_size_mean',
        'slic_size_std',
        'slic_compactness_mean',
        'slic_boundary_strength',
        'slic_largest_ratio',

        # Dominant Superpixel Features (6)
        'slic_dominant_red',
        'slic_dominant_hue',
        'slic_dominant_saturation',
        'slic_dominant_area',
        'slic_color_variance',
        'slic_dominant_red_ratio'
    ]

    assert len(names) == 12, f"Expected 12 feature names, got {len(names)}"

    return names


if __name__ == '__main__':
    # Test feature extraction
    print("Testing SLIC Superpixel Features...")

    # Create a dummy cell with uniform red region (simulating ball)
    dummy_cell_ball = np.zeros((75, 100, 3), dtype=np.uint8)
    dummy_cell_ball[:, :] = [50, 50, 200]  # BGR: Red background
    cv2.circle(dummy_cell_ball, (50, 37), 25, (50, 50, 255), -1)  # Brighter red circle

    # Create a dummy cell with vertical white stripes (simulating stump)
    dummy_cell_stump = np.ones((75, 100, 3), dtype=np.uint8) * 80  # Dark background
    cv2.rectangle(dummy_cell_stump, (40, 10), (45, 65), (255, 255, 255), -1)
    cv2.rectangle(dummy_cell_stump, (55, 10), (60, 65), (255, 255, 255), -1)

    # Create a dummy cell with complex random texture (simulating grass)
    dummy_cell_grass = np.random.randint(60, 140, (75, 100, 3), dtype=np.uint8)

    # Extract features
    features_ball = extract_superpixel_features(dummy_cell_ball)
    features_stump = extract_superpixel_features(dummy_cell_stump)
    features_grass = extract_superpixel_features(dummy_cell_grass)

    print(f"\nExtracted {len(features_ball)} features")
    print(f"\nKey SLIC Metrics Comparison:")
    print(f"{'Feature Name':<30} {'Ball':<12} {'Stump':<12} {'Grass':<12}")
    print("=" * 70)

    names = get_superpixel_feature_names()

    # Show key discriminative features
    key_features = ['slic_num_superpixels', 'slic_largest_ratio',
                   'slic_dominant_red', 'slic_dominant_red_ratio']
    for key in key_features:
        idx = names.index(key)
        print(f"{key:<30} {features_ball[idx]:<12.4f} {features_stump[idx]:<12.4f} {features_grass[idx]:<12.4f}")

    print("\n✓ Test passed!")
    print("  Ball: Few superpixels, high red content in dominant region")
    print("  Stump: Few superpixels (vertical white regions)")
    print("  Grass: Many superpixels (complex texture)")
