import numpy as np
import cv2
from typing import List


def extract_hsv_color_features(cell_hsv: np.ndarray) -> np.ndarray:
    """
    Extract 26 HSV color features from a cell (PHASE 4A - Enhanced).

    Features:
        - 6 basic statistics: h_mean, s_mean, v_mean, h_std, s_std, v_std
        - 4 hue histogram bins (LEGACY): red_pct, green_pct, blue_pct, brown_pct
        - 12 fine hue histogram bins (NEW): hue_bin_0_30, hue_bin_30_60, ..., hue_bin_330_360
        - 1 high saturation percentage
        - 1 color purity
        - 2 max values: s_max, v_max

    Args:
        cell_hsv: HSV image cell of shape (H, W, 3)
                 In OpenCV: H in [0, 180], S in [0, 255], V in [0, 255]

    Returns:
        Array of 26 features (14 original + 12 fine hue bins)
    """
    # Extract channels
    H = cell_hsv[:, :, 0].astype(np.float32)  # Hue: 0-180
    S = cell_hsv[:, :, 1].astype(np.float32)  # Saturation: 0-255
    V = cell_hsv[:, :, 2].astype(np.float32)  # Value: 0-255

    # Basic statistics (6 features)
    h_mean = np.mean(H)
    s_mean = np.mean(S)
    v_mean = np.mean(V)
    h_std = np.std(H)
    s_std = np.std(S)
    v_std = np.std(V)

    # Hue histogram bins - percentage in color ranges (4 features)
    # Red hue: 0-10 or 170-180 (red wraps around in HSV)
    red_mask = ((H >= 0) & (H <= 10)) | ((H >= 170) & (H <= 180))
    red_pct = np.sum(red_mask) / H.size

    # Green hue: 60-80
    green_mask = (H >= 60) & (H <= 80)
    green_pct = np.sum(green_mask) / H.size

    # Blue hue: 100-130
    blue_mask = (H >= 100) & (H <= 130)
    blue_pct = np.sum(blue_mask) / H.size

    # Brown hue: 10-30
    brown_mask = (H >= 10) & (H <= 30)
    brown_pct = np.sum(brown_mask) / H.size

    # High saturation percentage (1 feature)
    high_sat_pct = np.sum(S > 150) / S.size

    # Color purity (1 feature)
    # Normalized product of saturation and value
    color_purity = np.mean((S * V) / (255.0 * 255.0))

    # Max saturation and value (2 features)
    s_max = np.max(S)
    v_max = np.max(V)

    # ========== FINE HUE HISTOGRAM (12 features) - PHASE 4A ==========
    # Divide hue circle into 12 bins of 30 degrees each
    # OpenCV hue range: 0-180 (maps to 0-360 degrees)
    # Bin edges: [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
    # Corresponds to: [0°, 30°, 60°, 90°, 120°, 150°, 180°, 210°, 240°, 270°, 300°, 330°, 360°]

    # Critical bins for cricket objects:
    # - Bin 0 (0-30°): Pure red (ball) - MOST CRITICAL
    # - Bin 1 (30-60°): Orange/yellow-red (some balls, bat edges)
    # - Bin 2 (60-90°): Yellow/yellow-green
    # - Bin 3 (90-120°): Green (grass) - MAJOR CLASS
    # - Bin 4 (120-150°): Cyan/blue-green
    # - Bins 5-11: Blue, purple, magenta ranges (rare in cricket)

    fine_hue_bins = []
    bin_width = 15  # 15 in 0-180 scale = 30 degrees in 0-360 scale

    for bin_idx in range(12):
        bin_start = bin_idx * bin_width
        bin_end = (bin_idx + 1) * bin_width

        # Handle wrap-around for last bin (165-180)
        if bin_idx == 11:
            bin_end = 180

        # Count pixels in this hue bin
        bin_mask = (H >= bin_start) & (H < bin_end)
        bin_pct = np.sum(bin_mask) / H.size
        fine_hue_bins.append(bin_pct)

    # Convert to numpy array
    fine_hue_bins = np.array(fine_hue_bins, dtype=np.float32)

    features = np.array([
        # Original 14 features
        h_mean, s_mean, v_mean,
        h_std, s_std, v_std,
        red_pct, green_pct, blue_pct, brown_pct,
        high_sat_pct,
        color_purity,
        s_max, v_max,

        # New 12 fine hue bins (Phase 4A)
        fine_hue_bins[0],   # 0-30° (pure red - CRITICAL FOR BALL)
        fine_hue_bins[1],   # 30-60° (orange/yellow-red)
        fine_hue_bins[2],   # 60-90° (yellow)
        fine_hue_bins[3],   # 90-120° (green - grass)
        fine_hue_bins[4],   # 120-150° (cyan)
        fine_hue_bins[5],   # 150-180° (blue-green)
        fine_hue_bins[6],   # 180-210° (blue)
        fine_hue_bins[7],   # 210-240° (blue-purple)
        fine_hue_bins[8],   # 240-270° (purple)
        fine_hue_bins[9],   # 270-300° (magenta)
        fine_hue_bins[10],  # 300-330° (pink-red)
        fine_hue_bins[11]   # 330-360° (red-approaching)
    ], dtype=np.float32)

    # Verify we have exactly 26 features (14 original + 12 fine hue bins)
    assert len(features) == 26, f"Expected 26 features, got {len(features)}"

    return features


def rgb_to_hsv(cell_rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB cell to HSV.

    Args:
        cell_rgb: RGB image cell (BGR in OpenCV)

    Returns:
        HSV image cell
    """
    return cv2.cvtColor(cell_rgb, cv2.COLOR_BGR2HSV)


def get_hsv_feature_names() -> List[str]:
    """
    Get the names of HSV features (PHASE 4A - Enhanced).

    Returns:
        List of 26 feature names (14 original + 12 fine hue bins)
    """
    names = [
        # Original 14 features
        'h_mean', 's_mean', 'v_mean',
        'h_std', 's_std', 'v_std',
        'red_pct', 'green_pct', 'blue_pct', 'brown_pct',
        'high_sat_pct',
        'color_purity',
        's_max', 'v_max',

        # New 12 fine hue bins (Phase 4A)
        'hue_bin_0_30',      # Pure red (ball)
        'hue_bin_30_60',     # Orange/yellow-red
        'hue_bin_60_90',     # Yellow
        'hue_bin_90_120',    # Green (grass)
        'hue_bin_120_150',   # Cyan
        'hue_bin_150_180',   # Blue-green
        'hue_bin_180_210',   # Blue
        'hue_bin_210_240',   # Blue-purple
        'hue_bin_240_270',   # Purple
        'hue_bin_270_300',   # Magenta
        'hue_bin_300_330',   # Pink-red
        'hue_bin_330_360'    # Red-approaching
    ]

    assert len(names) == 26, f"Expected 26 feature names, got {len(names)}"

    return names
