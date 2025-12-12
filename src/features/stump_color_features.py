"""
Stump Color Features Module

Extracts 10 hand-crafted features optimized for white/yellow cricket stump detection.
Cricket stumps are typically white or pale yellow, with high brightness and low saturation.

Features (10 total):
    1. Whiteness intensity (high R+G+B, low variance)
    2. White pixel percentage (bright + low saturation)
    3. HSV value mean (brightness channel)
    4. HSV saturation mean (low = white/pale)
    5. Yellow hue percentage (wooden stumps)
    6. Grayscale intensity mean
    7. Grayscale intensity variance (uniformity)
    8. Lightness mean (LAB L* channel)
    9. Color variance across channels (low = white)
    10. Maximum brightness
"""

import cv2
import numpy as np


def extract_stump_color_features(cell_image):
    """
    Extract 10 white/yellow stump detection features from cell image.

    Args:
        cell_image: RGB cell image (100x75 pixels typically)

    Returns:
        np.ndarray: 10 stump-specific color features
    """
    features = []

    # Ensure image is valid
    if cell_image is None or cell_image.size == 0:
        return np.zeros(10)

    # Convert to different color spaces
    hsv = cv2.cvtColor(cell_image, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(cell_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lab = cv2.cvtColor(cell_image, cv2.COLOR_RGB2LAB)

    # Extract channels
    R = cell_image[:, :, 0].astype(np.float32)
    G = cell_image[:, :, 1].astype(np.float32)
    B = cell_image[:, :, 2].astype(np.float32)

    H = hsv[:, :, 0].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)

    L = lab[:, :, 0].astype(np.float32)

    # Feature 1: Whiteness intensity
    # White = high R+G+B with low channel variance
    rgb_sum = R + G + B
    whiteness_intensity = np.mean(rgb_sum) / (3 * 255.0)
    features.append(whiteness_intensity)

    # Feature 2: White pixel percentage
    # White pixels: high brightness (V > 200) and low saturation (S < 50)
    white_pixel_mask = (V > 200) & (S < 50)
    white_pixel_percentage = np.sum(white_pixel_mask) / (V.shape[0] * V.shape[1])
    features.append(white_pixel_percentage)

    # Feature 3: HSV value mean (brightness)
    # Stumps are bright
    value_mean = np.mean(V) / 255.0
    features.append(value_mean)

    # Feature 4: HSV saturation mean
    # Low saturation = white/desaturated colors
    saturation_mean = np.mean(S) / 255.0
    features.append(saturation_mean)

    # Feature 5: Yellow hue percentage
    # Some stumps are yellow/cream colored
    # Yellow hue is around 30-60 degrees (15-30 in OpenCV 0-180 scale)
    yellow_hue_mask = (H >= 15) & (H <= 35)
    yellow_hue_percentage = np.sum(yellow_hue_mask) / (H.shape[0] * H.shape[1])
    features.append(yellow_hue_percentage)

    # Feature 6: Grayscale intensity mean
    # Overall brightness
    gray_mean = np.mean(gray) / 255.0
    features.append(gray_mean)

    # Feature 7: Grayscale intensity variance
    # Low variance = uniform color (characteristic of stumps)
    gray_variance = np.var(gray) / (255.0 ** 2)
    features.append(gray_variance)

    # Feature 8: Lightness mean (LAB L* channel)
    # High lightness = bright/white
    # LAB L* ranges from 0-255 in OpenCV (0=black, 255=white)
    lightness_mean = np.mean(L) / 255.0
    features.append(lightness_mean)

    # Feature 9: Color variance across RGB channels
    # White has R ≈ G ≈ B (low variance across channels)
    # For each pixel, calculate variance across R, G, B channels
    rgb_stack = np.stack([R, G, B], axis=2)
    channel_variance = np.var(rgb_stack, axis=2)  # Variance across RGB for each pixel
    mean_channel_variance = np.mean(channel_variance) / (255.0 ** 2)
    features.append(mean_channel_variance)

    # Feature 10: Maximum brightness
    # Peak brightness in the cell
    max_brightness = np.max(V) / 255.0
    features.append(max_brightness)

    return np.array(features, dtype=np.float32)


def get_stump_color_feature_names():
    """
    Get names of all stump color features.

    Returns:
        list: 10 feature names
    """
    return [
        'stump_whiteness_intensity',
        'stump_white_pixel_percentage',
        'stump_value_mean',
        'stump_saturation_mean',
        'stump_yellow_hue_percentage',
        'stump_gray_intensity_mean',
        'stump_gray_variance',
        'stump_lightness_mean',
        'stump_color_channel_variance',
        'stump_max_brightness'
    ]


# Test function
if __name__ == '__main__':
    # Create a test image with white region
    test_image = np.zeros((75, 100, 3), dtype=np.uint8)

    # Add a white vertical region (simulating stump)
    for y in range(10, 65):
        for x in range(40, 60):
            # White/cream color with slight variation
            noise = np.random.randint(-10, 10)
            test_image[y, x] = [240 + noise, 240 + noise, 240 + noise]

    # Clip to valid range
    test_image = np.clip(test_image, 0, 255)

    # Extract features
    features = extract_stump_color_features(test_image)
    feature_names = get_stump_color_feature_names()

    print("Stump Color Features Test:")
    print("="*60)
    for name, value in zip(feature_names, features):
        print(f"{name:40s}: {value:.4f}")
    print("="*60)
    print(f"Total features: {len(features)}")

    # Verify feature count
    assert len(features) == 10, f"Expected 10 features, got {len(features)}"
    print("\n✓ Feature count validation passed")

    # Test with yellow stump
    print("\n" + "="*60)
    print("Testing with yellow stump:")
    print("="*60)

    # Create yellow stump
    test_image_yellow = np.zeros((75, 100, 3), dtype=np.uint8)
    for y in range(10, 65):
        for x in range(40, 60):
            # Yellow/cream color
            noise = np.random.randint(-10, 10)
            test_image_yellow[y, x] = [255 + noise, 255 + noise, 200 + noise]

    test_image_yellow = np.clip(test_image_yellow, 0, 255)

    features_yellow = extract_stump_color_features(test_image_yellow)

    for name, value in zip(feature_names, features_yellow):
        print(f"{name:40s}: {value:.4f}")
    print("="*60)
