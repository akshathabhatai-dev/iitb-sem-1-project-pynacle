import cv2
import numpy as np
from typing import Tuple

def extract_hsv_hist_features(
    cell_rgb: np.ndarray,
    mode: str = "1d",                # '1d' or '3d'
    h_bins: int = 16,
    s_bins: int = 8,
    v_bins: int = 4,
    normalize: bool = True
) -> np.ndarray:
    """
    Extract HSV histogram features from a single image cell (RGB uint8 numpy array).
    Returns a 1D numpy array (dtype=float32).

    mode='1d'  -> concatenate H, S, V 1D histograms (dims = h_bins + s_bins + v_bins)
    mode='3d'  -> return flattened 3D histogram (dims = h_bins*s_bins*v_bins)

    Uses OpenCV HSV ranges: H [0,180], S [0,256], V [0,256].
    """
    if cell_rgb is None:
        return np.zeros(h_bins + s_bins + v_bins, dtype=np.float32) if mode == "1d" else \
               np.zeros(h_bins * s_bins * v_bins, dtype=np.float32)

    # convert to HSV (OpenCV expects BGR)
    cell_bgr = cv2.cvtColor(cell_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)

    if mode == "3d":
        hist = cv2.calcHist([hsv], [0,1,2], None, [h_bins, s_bins, v_bins],
                            [0,180, 0,256, 0,256])
        hist = hist.flatten().astype(np.float32)
        if normalize:
            s = hist.sum()
            if s > 0:
                hist /= s
        return hist

    # mode == '1d'
    h_hist = cv2.calcHist([hsv], [0], None, [h_bins], [0,180]).flatten().astype(np.float32)
    s_hist = cv2.calcHist([hsv], [1], None, [s_bins], [0,256]).flatten().astype(np.float32)
    v_hist = cv2.calcHist([hsv], [2], None, [v_bins], [0,256]).flatten().astype(np.float32)

    if normalize:
        # normalize each channel separately (or concat then normalize if you prefer)
        if h_hist.sum() > 0:
            h_hist /= h_hist.sum()
        if s_hist.sum() > 0:
            s_hist /= s_hist.sum()
        if v_hist.sum() > 0:
            v_hist /= v_hist.sum()

    return np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)

def get_hsv_hist_feature_names(
    h_bins: int = 16,
    s_bins: int = 8,
    v_bins: int = 4,
    mode: str = "1d"
):
    """
    Create feature names for HSV histogram features.
    
    Parameters:
        h_bins (int): Number of Hue bins.
        s_bins (int): Number of Saturation bins.
        v_bins (int): Number of Value bins.
        mode (str): '1d' for separate H,S,V histograms; 
                    '3d' for a single flattened 3D histogram.
                    
    Returns:
        List[str]: Feature names.
    """
    
    feature_names = []

    if mode == "1d":
        # Hue channels
        for i in range(h_bins):
            feature_names.append(f"hsv_H_bin_{i}")
        
        # Saturation channels
        for i in range(s_bins):
            feature_names.append(f"hsv_S_bin_{i}")

        # Value channels
        for i in range(v_bins):
            feature_names.append(f"hsv_V_bin_{i}")

        return feature_names

    elif mode == "3d":
        # 3D histogram: (H_idx, S_idx, V_idx)
        for h in range(h_bins):
            for s in range(s_bins):
                for v in range(v_bins):
                    feature_names.append(f"hsv_H{h}_S{s}_V{v}")
        
        return feature_names

    else:
        raise ValueError("mode must be '1d' or '3d'")
