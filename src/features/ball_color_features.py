"""
Ball Color Features Module

Extracts 12 hand-crafted features optimized for red cricket ball detection.
Cricket balls are typically bright red/pink, making color a strong discriminator.

Features (12 total):
    1. Red hue dominance (percentage of red hue pixels)
    2. Red hue mean
    3. Red saturation mean
    4. Red value mean (HSV)
    5. R-G ratio mean (red channel / green channel)
    6. R-B ratio mean (red channel / blue channel)
    7. Red pixel percentage (R > G and R > B)
    8. Red intensity mean (red channel raw value)
    9. LAB a* channel mean (red-green axis)
    10. LAB a* channel variance
    11. Red cluster compactness (spatial concentration)
    12. Maximum red intensity
"""

import cv2
import numpy as np


def extract_ball_color_features(cell_image):
    """
    Extract 12 red ball detection features from cell image.

    Args:
        cell_image: RGB cell image (100x75 pixels typically)

    Returns:
        np.ndarray: 12 ball-specific color features
    """
    features = []

    # Ensure image is valid
    if cell_image is None or cell_image.size == 0:
        return np.zeros(12)

    # Convert to different color spaces
    hsv = cv2.cvtColor(cell_image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(cell_image, cv2.COLOR_RGB2LAB)

    # Extract channels
    R = cell_image[:, :, 0].astype(np.float32)
    G = cell_image[:, :, 1].astype(np.float32)
    B = cell_image[:, :, 2].astype(np.float32)

    H = hsv[:, :, 0].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)

    L = lab[:, :, 0].astype(np.float32)
    a_star = lab[:, :, 1].astype(np.float32)  # Red-green axis
    b_star = lab[:, :, 2].astype(np.float32)  # Yellow-blue axis

    # Feature 1: Red hue dominance
    # Red hue is in ranges [0-10] and [160-180] in OpenCV (0-180 scale)
    red_hue_mask1 = (H <= 10)
    red_hue_mask2 = (H >= 160)
    red_hue_mask = red_hue_mask1 | red_hue_mask2
    red_hue_percentage = np.sum(red_hue_mask) / (H.shape[0] * H.shape[1])
    features.append(red_hue_percentage)

    # Feature 2: Red hue mean (normalized)
    # Normalize red hues to 0-1 range (considering wraparound)
    red_hue_pixels = H[red_hue_mask]
    if len(red_hue_pixels) > 0:
        red_hue_mean = np.mean(red_hue_pixels) / 180.0
    else:
        red_hue_mean = 0
    features.append(red_hue_mean)

    # Feature 3: Red saturation mean
    # High saturation indicates vivid red color
    red_pixels_saturation = S[red_hue_mask]
    if len(red_pixels_saturation) > 0:
        red_saturation_mean = np.mean(red_pixels_saturation) / 255.0
    else:
        red_saturation_mean = 0
    features.append(red_saturation_mean)

    # Feature 4: Red value mean (HSV)
    # Brightness of red pixels
    red_pixels_value = V[red_hue_mask]
    if len(red_pixels_value) > 0:
        red_value_mean = np.mean(red_pixels_value) / 255.0
    else:
        red_value_mean = 0
    features.append(red_value_mean)

    # Feature 5: R-G ratio mean
    # Red channel dominance over green
    with np.errstate(divide='ignore', invalid='ignore'):
        rg_ratio = np.where(G > 0, R / G, 0)
        rg_ratio_mean = np.mean(rg_ratio)
        # Clip to reasonable range
        rg_ratio_mean = np.clip(rg_ratio_mean, 0, 5)
    features.append(rg_ratio_mean)

    # Feature 6: R-B ratio mean
    # Red channel dominance over blue
    with np.errstate(divide='ignore', invalid='ignore'):
        rb_ratio = np.where(B > 0, R / B, 0)
        rb_ratio_mean = np.mean(rb_ratio)
        # Clip to reasonable range
        rb_ratio_mean = np.clip(rb_ratio_mean, 0, 5)
    features.append(rb_ratio_mean)

    # Feature 7: Red pixel percentage
    # Percentage of pixels where R > G and R > B
    red_pixel_mask = (R > G) & (R > B) & (R > 100)  # Threshold to avoid noise
    red_pixel_percentage = np.sum(red_pixel_mask) / (R.shape[0] * R.shape[1])
    features.append(red_pixel_percentage)

    # Feature 8: Red intensity mean
    # Raw red channel mean value
    red_intensity_mean = np.mean(R) / 255.0
    features.append(red_intensity_mean)

    # Feature 9: LAB a* channel mean (red-green axis)
    # Positive values indicate red, negative indicate green
    # LAB a* ranges from 0-255 in OpenCV (128 = neutral)
    a_star_mean = (np.mean(a_star) - 128) / 128.0  # Normalize to [-1, 1]
    features.append(a_star_mean)

    # Feature 10: LAB a* channel variance
    # High variance may indicate red object presence
    a_star_variance = np.var(a_star) / (128.0 ** 2)  # Normalized variance
    features.append(a_star_variance)

    # Feature 11: Red cluster compactness
    # Measure spatial concentration of red pixels (ball is typically compact)
    if np.sum(red_pixel_mask) > 0:
        # Find center of mass of red pixels
        y_coords, x_coords = np.where(red_pixel_mask)
        center_y = np.mean(y_coords)
        center_x = np.mean(x_coords)

        # Calculate average distance from center
        distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        avg_distance = np.mean(distances)

        # Normalize by image diagonal
        diagonal = np.sqrt(cell_image.shape[0]**2 + cell_image.shape[1]**2)
        compactness = 1 - (avg_distance / diagonal)  # Higher = more compact
        compactness = np.clip(compactness, 0, 1)
    else:
        compactness = 0

    features.append(compactness)

    # Feature 12: Maximum red intensity
    # Peak red value in the cell
    max_red_intensity = np.max(R) / 255.0
    features.append(max_red_intensity)

    return np.array(features, dtype=np.float32)


def get_ball_color_feature_names():
    """
    Get names of all ball color features.

    Returns:
        list: 12 feature names
    """
    return [
        'ball_red_hue_percentage',
        'ball_red_hue_mean',
        'ball_red_saturation_mean',
        'ball_red_value_mean',
        'ball_rg_ratio_mean',
        'ball_rb_ratio_mean',
        'ball_red_pixel_percentage',
        'ball_red_intensity_mean',
        'ball_lab_a_star_mean',
        'ball_lab_a_star_variance',
        'ball_red_cluster_compactness',
        'ball_max_red_intensity'
    ]


# Test function
if __name__ == '__main__':
    # Create a test image with red region
    test_image = np.zeros((75, 100, 3), dtype=np.uint8)
    test_image[:, :, 0] = 200  # High red channel
    test_image[:, :, 1] = 50   # Low green channel
    test_image[:, :, 2] = 50   # Low blue channel

    # Add a bright red circular region (simulating ball)
    center = (50, 37)
    radius = 15
    for y in range(test_image.shape[0]):
        for x in range(test_image.shape[1]):
            if (x - center[0])**2 + (y - center[1])**2 < radius**2:
                test_image[y, x] = [255, 0, 0]  # Bright red

    # Extract features
    features = extract_ball_color_features(test_image)
    feature_names = get_ball_color_feature_names()

    print("Ball Color Features Test:")
    print("="*60)
    for name, value in zip(feature_names, features):
        print(f"{name:40s}: {value:.4f}")
    print("="*60)
    print(f"Total features: {len(features)}")

    # Verify feature count
    assert len(features) == 12, f"Expected 12 features, got {len(features)}"
    print("\n✓ Feature count validation passed")
