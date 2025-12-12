"""
Bat Color Features Module

Extracts 12 hand-crafted features optimized for brown/wood cricket bat detection.
Cricket bats are typically brown/tan/wood-colored, making brown hue detection important.

Features (12 total):
    1. Brown hue dominance (percentage of brown hue pixels)
    2. Brown hue mean
    3. Brown saturation mean
    4. Brown value mean (HSV)
    5. R-G ratio mean (distinguishes brown from green grass)
    6. G-B ratio mean (brown has more green than blue)
    7. Brown pixel percentage (brown hue range + moderate saturation)
    8. LAB b* channel mean (yellow-blue axis, brown is yellowish)
    9. LAB b* channel variance
    10. Wood texture indicator (local intensity variation)
    11. Brown cluster elongation (bat is elongated shape)
    12. Maximum brown intensity
"""

import cv2
import numpy as np


def extract_bat_color_features(cell_image):
    """
    Extract 12 brown bat detection features from cell image.

    Args:
        cell_image: RGB cell image (100x75 pixels typically)

    Returns:
        np.ndarray: 12 bat-specific color features
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
    a_star = lab[:, :, 1].astype(np.float32)
    b_star = lab[:, :, 2].astype(np.float32)  # Yellow-blue axis

    # Feature 1: Brown hue dominance
    # Brown/wood hue is typically in range [10-30] degrees (orange-brown range)
    # In OpenCV: 0-180 scale, so [5-15] for brown
    brown_hue_mask = (H >= 5) & (H <= 30)  # Broader range to capture wood tones
    brown_hue_percentage = np.sum(brown_hue_mask) / (H.shape[0] * H.shape[1])
    features.append(brown_hue_percentage)

    # Feature 2: Brown hue mean
    brown_hue_pixels = H[brown_hue_mask]
    if len(brown_hue_pixels) > 0:
        brown_hue_mean = np.mean(brown_hue_pixels) / 180.0
    else:
        brown_hue_mean = 0
    features.append(brown_hue_mean)

    # Feature 3: Brown saturation mean
    # Moderate saturation (not too vivid, not too dull)
    brown_pixels_saturation = S[brown_hue_mask]
    if len(brown_pixels_saturation) > 0:
        brown_saturation_mean = np.mean(brown_pixels_saturation) / 255.0
    else:
        brown_saturation_mean = 0
    features.append(brown_saturation_mean)

    # Feature 4: Brown value mean (HSV)
    brown_pixels_value = V[brown_hue_mask]
    if len(brown_pixels_value) > 0:
        brown_value_mean = np.mean(brown_pixels_value) / 255.0
    else:
        brown_value_mean = 0
    features.append(brown_value_mean)

    # Feature 5: R-G ratio mean
    # Brown has R > G, but not as extreme as red
    # Helps distinguish brown from green grass
    with np.errstate(divide='ignore', invalid='ignore'):
        rg_ratio = np.where(G > 0, R / G, 0)
        rg_ratio_mean = np.mean(rg_ratio)
        rg_ratio_mean = np.clip(rg_ratio_mean, 0, 3)  # Reasonable range for brown
    features.append(rg_ratio_mean)

    # Feature 6: G-B ratio mean
    # Brown has G > B (yellowish component)
    with np.errstate(divide='ignore', invalid='ignore'):
        gb_ratio = np.where(B > 0, G / B, 0)
        gb_ratio_mean = np.mean(gb_ratio)
        gb_ratio_mean = np.clip(gb_ratio_mean, 0, 3)
    features.append(gb_ratio_mean)

    # Feature 7: Brown pixel percentage
    # Pixels that match brown criteria: brown hue + moderate saturation + brightness
    brown_pixel_mask = brown_hue_mask & (S > 30) & (S < 180) & (V > 50)
    brown_pixel_percentage = np.sum(brown_pixel_mask) / (R.shape[0] * R.shape[1])
    features.append(brown_pixel_percentage)

    # Feature 8: LAB b* channel mean (yellow-blue axis)
    # Positive values = yellow/brown, negative = blue
    # LAB b* ranges from 0-255 in OpenCV (128 = neutral)
    b_star_mean = (np.mean(b_star) - 128) / 128.0  # Normalize to [-1, 1]
    features.append(b_star_mean)

    # Feature 9: LAB b* channel variance
    b_star_variance = np.var(b_star) / (128.0 ** 2)
    features.append(b_star_variance)

    # Feature 10: Wood texture indicator
    # Wood has grain/texture - measure local intensity variation
    gray = cv2.cvtColor(cell_image, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Calculate local standard deviation using 3x3 window
    kernel_size = 3
    gray_squared = gray ** 2
    mean_gray = cv2.blur(gray, (kernel_size, kernel_size))
    mean_gray_squared = cv2.blur(gray_squared, (kernel_size, kernel_size))
    local_variance = mean_gray_squared - (mean_gray ** 2)
    local_variance = np.maximum(local_variance, 0)  # Avoid negative due to numerical issues

    wood_texture = np.mean(local_variance) / (255.0 ** 2)  # Normalized variance
    features.append(wood_texture)

    # Feature 11: Brown cluster elongation
    # Bat is elongated shape - measure aspect ratio of brown regions
    if np.sum(brown_pixel_mask) > 10:  # Need enough pixels
        y_coords, x_coords = np.where(brown_pixel_mask)

        # Calculate moments to get orientation and elongation
        if len(y_coords) > 0:
            # Center coordinates
            y_center = np.mean(y_coords)
            x_center = np.mean(x_coords)

            # Centered coordinates
            y_c = y_coords - y_center
            x_c = x_coords - x_center

            # Covariance matrix elements
            cov_xx = np.mean(x_c ** 2)
            cov_yy = np.mean(y_c ** 2)
            cov_xy = np.mean(x_c * y_c)

            # Eigenvalues of covariance matrix
            trace = cov_xx + cov_yy
            det = cov_xx * cov_yy - cov_xy ** 2

            if det > 0 and trace > 0:
                lambda1 = trace / 2 + np.sqrt((trace / 2) ** 2 - det)
                lambda2 = trace / 2 - np.sqrt((trace / 2) ** 2 - det)

                if lambda2 > 0:
                    elongation = np.sqrt(lambda1 / lambda2)
                    # Normalize: higher values = more elongated (bat-like)
                    elongation = np.clip(elongation - 1, 0, 5) / 5  # Scale to [0, 1]
                else:
                    elongation = 0
            else:
                elongation = 0
        else:
            elongation = 0
    else:
        elongation = 0

    features.append(elongation)

    # Feature 12: Maximum brown intensity
    # Peak brightness in brown regions
    if np.sum(brown_pixel_mask) > 0:
        brown_intensities = V[brown_pixel_mask]
        max_brown_intensity = np.max(brown_intensities) / 255.0
    else:
        max_brown_intensity = 0

    features.append(max_brown_intensity)

    return np.array(features, dtype=np.float32)


def get_bat_color_feature_names():
    """
    Get names of all bat color features.

    Returns:
        list: 12 feature names
    """
    return [
        'bat_brown_hue_percentage',
        'bat_brown_hue_mean',
        'bat_brown_saturation_mean',
        'bat_brown_value_mean',
        'bat_rg_ratio_mean',
        'bat_gb_ratio_mean',
        'bat_brown_pixel_percentage',
        'bat_lab_b_star_mean',
        'bat_lab_b_star_variance',
        'bat_wood_texture_indicator',
        'bat_brown_cluster_elongation',
        'bat_max_brown_intensity'
    ]


# Test function
if __name__ == '__main__':
    # Create a test image with brown region
    test_image = np.zeros((75, 100, 3), dtype=np.uint8)

    # Add a brown/tan elongated region (simulating bat)
    # Brown is roughly RGB(139, 69, 19) or similar
    for y in range(20, 55):
        for x in range(30, 70):
            # Add some texture variation to simulate wood grain
            noise = np.random.randint(-20, 20)
            test_image[y, x] = [139 + noise, 69 + noise, 19 + noise]

    # Make it elongated (bat shape)
    test_image[:, :, :] = np.clip(test_image, 0, 255)

    # Extract features
    features = extract_bat_color_features(test_image)
    feature_names = get_bat_color_feature_names()

    print("Bat Color Features Test:")
    print("="*60)
    for name, value in zip(feature_names, features):
        print(f"{name:40s}: {value:.4f}")
    print("="*60)
    print(f"Total features: {len(features)}")

    # Verify feature count
    assert len(features) == 12, f"Expected 12 features, got {len(features)}"
    print("\n✓ Feature count validation passed")
