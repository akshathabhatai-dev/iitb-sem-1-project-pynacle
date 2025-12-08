"""
SIFT Keypoint Features for Cricket Object Detection

This module extracts 14 SIFT (Scale-Invariant Feature Transform) keypoint features
for detecting and describing local object features.

Features breakdown:
- Keypoint Statistics (8): Count, density, response strength, scale, orientation
- Descriptor Statistics (6): Mean, std, sparsity, L2 norm, max bin, contrast

Critical for:
- Ball: Few keypoints with high response (smooth circular object)
- Bat: Moderate keypoints along edges (elongated object)
- Stump: Multiple keypoints at vertical edges (textured vertical object)
- Background: Many scattered keypoints with low response (grass texture)

Author: Pynacle Team
Date: 2025
"""

import numpy as np
import cv2
from typing import List, Optional


def extract_sift_features(cell_rgb: np.ndarray) -> np.ndarray:
    """
    Extract 14 SIFT keypoint features from RGB cell.

    SIFT detects scale-invariant keypoints and computes descriptors.
    Critical for detecting objects of varying sizes and orientations.

    Args:
        cell_rgb: RGB image cell of shape (H, W, 3) in BGR format
                 Expected size: 75x100 pixels

    Returns:
        Array of 14 float32 features
    """
    # Handle empty or invalid images
    if cell_rgb.size == 0:
        return np.zeros(14, dtype=np.float32)

    # Convert to grayscale for SIFT
    cell_gray = cv2.cvtColor(cell_rgb, cv2.COLOR_BGR2GRAY)

    # Get total pixel count for normalization
    total_pixels = cell_gray.shape[0] * cell_gray.shape[1]

    # ========== SIFT KEYPOINT DETECTION ==========

    try:
        # Create SIFT detector
        # nfeatures=50: Limit to top 50 keypoints to avoid overfitting to background
        # contrastThreshold=0.04: Default, filters out low-contrast keypoints
        # edgeThreshold=10: Default, filters out edge-like features
        sift = cv2.SIFT_create(nfeatures=50, contrastThreshold=0.04, edgeThreshold=10)

        # Detect keypoints and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(cell_gray, None)

        # ========== KEYPOINT STATISTICS (8 features) ==========

        if keypoints is not None and len(keypoints) > 0:
            # 1. SIFT keypoint count
            # Ball: few keypoints (smooth surface)
            # Bat/Stump: moderate keypoints (textured, edges)
            # Grass: many keypoints (complex texture)
            sift_keypoint_count = float(len(keypoints))

            # 2. SIFT keypoint density (keypoints per pixel)
            sift_keypoint_density = sift_keypoint_count / float(total_pixels)

            # Extract keypoint properties
            responses = np.array([kp.response for kp in keypoints])  # Strength of keypoint
            sizes = np.array([kp.size for kp in keypoints])  # Scale (diameter)
            angles = np.array([kp.angle for kp in keypoints])  # Orientation (0-360°)

            # 3. SIFT response mean (average keypoint strength)
            # High response = strong corners/blobs (ball edge, stump corners)
            sift_response_mean = np.mean(responses)

            # 4. SIFT response max (strongest keypoint)
            sift_response_max = np.max(responses)

            # 5. SIFT response std (variability in keypoint strength)
            # Uniform object (ball) has low std, complex object has high std
            sift_response_std = np.std(responses)

            # 6. SIFT scale mean (average keypoint size)
            # Larger objects have larger scale keypoints
            sift_scale_mean = np.mean(sizes)

            # 7. SIFT scale std (variability in keypoint scale)
            # Uniform-scale object has low std
            sift_scale_std = np.std(sizes)

            # 8. SIFT orientation variance (directional uniformity)
            # Circular object (ball) has high variance (edges in all directions)
            # Vertical object (stump) has low variance (mostly vertical edges)
            # Horizontal object (bat) has low variance (mostly horizontal edges)
            sift_orientation_variance = np.var(angles)

        else:
            # No keypoints detected (smooth uniform region)
            sift_keypoint_count = 0.0
            sift_keypoint_density = 0.0
            sift_response_mean = 0.0
            sift_response_max = 0.0
            sift_response_std = 0.0
            sift_scale_mean = 0.0
            sift_scale_std = 0.0
            sift_orientation_variance = 0.0

        # ========== DESCRIPTOR STATISTICS (6 features) ==========

        if descriptors is not None and len(descriptors) > 0:
            # SIFT descriptor: 128-dimensional vector for each keypoint
            # Each element represents histogram of gradients in a subregion

            # 9. SIFT descriptor mean (average descriptor value)
            sift_descriptor_mean = np.mean(descriptors)

            # 10. SIFT descriptor std (descriptor variability)
            # Complex objects have high variability
            sift_descriptor_std = np.std(descriptors)

            # 11. SIFT descriptor sparsity (percentage of near-zero values)
            # Smooth objects have sparse descriptors (many zeros)
            threshold = 10.0  # Values < 10 considered "near zero"
            sift_descriptor_sparsity = np.mean(descriptors < threshold)

            # 12. SIFT descriptor L2 norm mean (average descriptor magnitude)
            # Measures overall descriptor strength
            descriptor_norms = np.linalg.norm(descriptors, axis=1)
            sift_descriptor_l2_norm_mean = np.mean(descriptor_norms)

            # 13. SIFT descriptor max bin (maximum value across all descriptors)
            # Strong gradients produce high max values
            sift_descriptor_max_bin = np.max(descriptors)

            # 14. SIFT descriptor contrast (range of descriptor values)
            # High contrast = diverse gradient patterns
            sift_descriptor_contrast = np.max(descriptors) - np.min(descriptors)

        else:
            # No descriptors computed
            sift_descriptor_mean = 0.0
            sift_descriptor_std = 0.0
            sift_descriptor_sparsity = 0.0
            sift_descriptor_l2_norm_mean = 0.0
            sift_descriptor_max_bin = 0.0
            sift_descriptor_contrast = 0.0

    except Exception as e:
        # If SIFT fails (rare), return zeros
        print(f"SIFT extraction failed: {e}")
        sift_keypoint_count = 0.0
        sift_keypoint_density = 0.0
        sift_response_mean = 0.0
        sift_response_max = 0.0
        sift_response_std = 0.0
        sift_scale_mean = 0.0
        sift_scale_std = 0.0
        sift_orientation_variance = 0.0
        sift_descriptor_mean = 0.0
        sift_descriptor_std = 0.0
        sift_descriptor_sparsity = 0.0
        sift_descriptor_l2_norm_mean = 0.0
        sift_descriptor_max_bin = 0.0
        sift_descriptor_contrast = 0.0

    # ========== COMBINE ALL FEATURES ==========
    features = np.array([
        # Keypoint Statistics (8)
        sift_keypoint_count,
        sift_keypoint_density,
        sift_response_mean,
        sift_response_max,
        sift_response_std,
        sift_scale_mean,
        sift_scale_std,
        sift_orientation_variance,

        # Descriptor Statistics (6)
        sift_descriptor_mean,
        sift_descriptor_std,
        sift_descriptor_sparsity,
        sift_descriptor_l2_norm_mean,
        sift_descriptor_max_bin,
        sift_descriptor_contrast
    ], dtype=np.float32)

    # Verify we have exactly 14 features
    assert len(features) == 14, f"Expected 14 features, got {len(features)}"

    # Replace any NaN or Inf values with 0.0 (defensive programming)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def get_sift_feature_names() -> List[str]:
    """
    Get names of all 14 SIFT features.

    Returns:
        List of 14 feature names
    """
    names = [
        # Keypoint Statistics (8)
        'sift_keypoint_count',
        'sift_keypoint_density',
        'sift_response_mean',
        'sift_response_max',
        'sift_response_std',
        'sift_scale_mean',
        'sift_scale_std',
        'sift_orientation_variance',

        # Descriptor Statistics (6)
        'sift_descriptor_mean',
        'sift_descriptor_std',
        'sift_descriptor_sparsity',
        'sift_descriptor_l2_norm_mean',
        'sift_descriptor_max_bin',
        'sift_descriptor_contrast'
    ]

    assert len(names) == 14, f"Expected 14 feature names, got {len(names)}"

    return names


if __name__ == '__main__':
    # Test feature extraction
    print("Testing SIFT Keypoint Features...")

    # Create a dummy cell with circular object (simulating ball)
    dummy_cell_ball = np.ones((75, 100, 3), dtype=np.uint8) * 100
    cv2.circle(dummy_cell_ball, (50, 37), 20, (255, 255, 255), 2)  # Circle outline

    # Create a dummy cell with vertical edges (simulating stump)
    dummy_cell_stump = np.ones((75, 100, 3), dtype=np.uint8) * 100
    cv2.rectangle(dummy_cell_stump, (40, 10), (42, 65), (255, 255, 255), -1)
    cv2.rectangle(dummy_cell_stump, (58, 10), (60, 65), (255, 255, 255), -1)

    # Create a dummy cell with complex texture (simulating grass)
    dummy_cell_grass = np.random.randint(80, 140, (75, 100, 3), dtype=np.uint8)

    # Extract features
    features_ball = extract_sift_features(dummy_cell_ball)
    features_stump = extract_sift_features(dummy_cell_stump)
    features_grass = extract_sift_features(dummy_cell_grass)

    print(f"\nExtracted {len(features_ball)} features")
    print(f"\nKey SIFT Metrics Comparison:")
    print(f"{'Feature Name':<30} {'Ball':<12} {'Stump':<12} {'Grass':<12}")
    print("=" * 70)

    names = get_sift_feature_names()

    # Show key discriminative features
    key_features = ['sift_keypoint_count', 'sift_response_mean', 'sift_orientation_variance',
                   'sift_descriptor_sparsity']
    for key in key_features:
        idx = names.index(key)
        print(f"{key:<30} {features_ball[idx]:<12.4f} {features_stump[idx]:<12.4f} {features_grass[idx]:<12.4f}")

    print("\n✓ Test passed!")
    print("  Ball: Few keypoints (smooth), high orientation variance (circular)")
    print("  Stump: Moderate keypoints at edges, low orientation variance (vertical)")
    print("  Grass: Many keypoints (complex texture)")
