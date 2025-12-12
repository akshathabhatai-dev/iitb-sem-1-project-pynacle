"""
shape_structure_features.py

Handcrafted shape + structure features for small image grid cells.
Provides:
  - extract_shape_features(cell) -> list[float]
  - get_shape_feature_names() -> list[str]

Designed to integrate into an existing feature-extraction pipeline.
"""

from typing import List, Tuple
import numpy as np
import cv2
from skimage import filters, feature, measure, morphology, color
from scipy import ndimage as ndi

from src.features.shape_features import get_shape_feature_names

# ---------------------------
# Helper utility functions
# ---------------------------
def _to_gray_uint8(img: np.ndarray) -> np.ndarray:
    """Convert input image to grayscale uint8 (0..255)."""
    if img is None:
        return None
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    # normalize to uint8 if needed
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray

def _safe_div(a, b):
    return float(a) / float(b) if b not in (0, None, 0.0) else 0.0

# ---------------------------
# Feature extractors
# ---------------------------

def _circular_hough_features(gray: np.ndarray,
                             dp=1.2,
                             min_dist=8,
                             param1=100,
                             param2=14,
                             min_radius=2,
                             max_radius=30) -> Tuple[float, float, float, float]:
    """
    Use HoughCircles to get strong circle cues.
    Returns:
      (max_accum, n_circles, mean_radius_norm, radius_std_norm)
    All normalized to 0..1 scale (approx).
    """
    try:
        # blur modestly to reduce noise
        blur = cv2.medianBlur(gray, 3)
        circles = cv2.HoughCircles(blur,
                                   cv2.HOUGH_GRADIENT,
                                   dp=dp,
                                   minDist=min_dist,
                                   param1=param1,
                                   param2=param2,
                                   minRadius=min_radius,
                                   maxRadius=max_radius)
        if circles is None:
            return 0.0, 0.0, 0.0, 0.0
        circles = np.squeeze(circles)
        if circles.ndim == 1:
            circles = circles[np.newaxis, :]
        radii = circles[:, 2]
        n = len(radii)
        max_r = np.max(radii) if n > 0 else 0
        mean_r = np.mean(radii) if n > 0 else 0
        std_r = np.std(radii) if n > 0 else 0
        # normalize by cell diagonal
        h, w = gray.shape[:2]
        diag = np.sqrt(h*h + w*w)
        return 1.0, float(n)/10.0 if n<10 else 1.0, _safe_div(mean_r, diag), _safe_div(std_r, diag)
    except Exception:
        return 0.0, 0.0, 0.0, 0.0

def _radial_symmetry_score(gray: np.ndarray) -> float:
    """
    A quick radial-symmetry score. We approximate using gradients and
    how aligned they are pointing toward a central point.
    Returns normalized score 0..1.
    """
    try:
        # small Gaussian blur then Sobel gradients
        g = cv2.GaussianBlur(gray, (3,3), 0)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        h, w = gray.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w/2.0, h/2.0
        dx = (cx - xx)
        dy = (cy - yy)
        mag = np.sqrt(dx*dx + dy*dy) + 1e-6
        dxn = dx / mag
        dyn = dy / mag
        # normalize gradient vectors
        gmag = np.sqrt(gx*gx + gy*gy) + 1e-6
        gxn = gx / gmag
        gyn = gy / gmag
        # dot product between gradient direction and radial direction
        dot = gxn * dxn + gyn * dyn
        # positive dot means gradient points toward center (or away if negative)
        score = np.maximum(dot, 0.0).sum() / (h*w)
        return float(np.clip(score*3.0, 0.0, 1.0))  # scale a bit, clamp
    except Exception:
        return 0.0

def _edge_orientation_histogram(gray: np.ndarray, nbins=8) -> List[float]:
    """
    Returns normalized histogram of gradient orientations weighted by gradient magnitude.
    nbins defaults to 8 (45-degree bins).
    """
    try:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx*gx + gy*gy)
        ang = (np.arctan2(gy, gx) + np.pi)  # range [0, 2pi]
        bin_edges = np.linspace(0.0, 2*np.pi, nbins+1)
        hist = np.zeros(nbins, dtype=float)
        for i in range(nbins):
            mask = (ang >= bin_edges[i]) & (ang < bin_edges[i+1])
            hist[i] = mag[mask].sum()
        total = hist.sum() + 1e-8
        return (hist / total).tolist()
    except Exception:
        return [0.0]*nbins

def _line_features(gray: np.ndarray,
                   canny_thresh1=50,
                   canny_thresh2=150,
                   min_line_length=8,
                   max_line_gap=3) -> Tuple[float, float, float, float]:
    """
    Uses probabilistic Hough transform to find line segments.
    Returns:
      (n_lines_norm, mean_len_norm, std_len_norm, verticality_score)
    verticality_score = fraction of line energy strongly vertical.
    """
    try:
        edges = cv2.Canny(gray, canny_thresh1, canny_thresh2)
        lines = cv2.HoughLinesP(edges,
                                rho=1,
                                theta=np.pi/180,
                                threshold=20,
                                minLineLength=min_line_length,
                                maxLineGap=max_line_gap)
        if lines is None:
            return 0.0, 0.0, 0.0, 0.0
        lines = lines.reshape(-1, 4)
        lengths = np.sqrt((lines[:,0]-lines[:,2])**2 + (lines[:,1]-lines[:,3])**2)
        mean_len = float(np.mean(lengths))
        std_len = float(np.std(lengths))
        n = len(lengths)
        # verticality: proportion of line length that is near-vertical (angle within 20 deg)
        angles = np.arctan2(lines[:,3] - lines[:,1], lines[:,2] - lines[:,0])
        angles_deg = np.abs(np.degrees(angles))
        vertical_mask = (np.abs(angles_deg - 90) <= 20)
        vertical_frac = lengths[vertical_mask].sum() / (lengths.sum() + 1e-8)
        # normalize counts and lengths by cell size
        h, w = gray.shape[:2]
        diag = np.sqrt(h*h + w*w)
        return float(np.clip(n/10.0, 0.0, 1.0)), _safe_div(mean_len, diag), _safe_div(std_len, diag), float(np.clip(vertical_frac, 0.0, 1.0))
    except Exception:
        return 0.0, 0.0, 0.0, 0.0

def _aspect_elongation_moments(gray: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute region moments to get eccentricity / elongation.
    Returns:
      (eccentricity, aspect_ratio, solidity)
    Approach:
      - threshold using Otsu to get mask
      - measure largest connected region (if present)
    """
    try:
        # apply Otsu threshold on blurred image (works for small objects)
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # ensure foreground not everything: try invert if needed
        mask = (th > 0).astype(np.uint8)
        # morphological opening to remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        if mask.sum() == 0:
            return 0.0, 0.0, 0.0
        # find largest connected component
        labels = measure.label(mask, connectivity=2)
        props = measure.regionprops(labels)
        if not props:
            return 0.0, 0.0, 0.0
        # pick largest area region
        largest = max(props, key=lambda p: p.area)
        minr, minc, maxr, maxc = largest.bbox
        height = maxr - minr
        width = maxc - minc
        aspect = _safe_div(width, height) if height>0 else 0.0
        # compute eccentricity (0..1) (skimage returns 0..1)
        ecc = float(largest.eccentricity) if hasattr(largest, 'eccentricity') else 0.0
        solidity = float(largest.solidity) if hasattr(largest, 'solidity') else 0.0
        return ecc, aspect, solidity
    except Exception:
        return 0.0, 0.0, 0.0

def _boundary_center_contrast(gray: np.ndarray) -> Tuple[float, float]:
    """
    Compute:
      - mean_center_minus_border (center intensity minus border intensity)
      - center_grad_mag (mean gradient magnitude in a small central disk)
    """
    try:
        h, w = gray.shape[:2]
        # center region (half size)
        ch, cw = int(h*0.4), int(w*0.4)
        sy = (h - ch)//2
        sx = (w - cw)//2
        center = gray[sy:sy+ch, sx:sx+cw].astype(np.float32)
        # border region: top strip + bottom + left + right (thin)
        bw = max(1, int(min(h,w)*0.12))
        border_mask = np.zeros_like(gray, dtype=bool)
        border_mask[:bw,:] = True
        border_mask[-bw:,:] = True
        border_mask[:, :bw] = True
        border_mask[:, -bw:] = True
        border_vals = gray[border_mask].astype(np.float32)
        if border_vals.size == 0 or center.size == 0:
            return 0.0, 0.0
        center_mean = float(center.mean())
        border_mean = float(border_vals.mean())
        # center gradient magnitude
        gx = cv2.Sobel(center, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(center, cv2.CV_32F, 0, 1, ksize=3)
        gmag = np.sqrt(gx*gx + gy*gy).mean()
        contrast = _safe_div(center_mean - border_mean, 255.0)
        return float(np.clip(contrast, -1.0, 1.0)), float(np.clip(gmag/50.0, 0.0, 1.0))
    except Exception:
        return 0.0, 0.0

def _contour_features(gray: np.ndarray) -> Tuple[float, float, float]:
    """
    contour area, perimeter, compactness (4*pi*area / perimeter^2)
    normalized area by cell area.
    """
    try:
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, 0.0, 0.0
        # pick largest contour
        area = 0.0
        peri = 0.0
        for c in contours:
            a = cv2.contourArea(c)
            if a > area:
                area = a
                peri = cv2.arcLength(c, True)
        h, w = gray.shape[:2]
        cell_area = float(h*w)
        area_norm = _safe_div(area, cell_area)
        compactness = _safe_div(4.0*np.pi*area, (peri*peri + 1e-8))
        return float(area_norm), float(np.clip(peri/ (np.sqrt(cell_area)+1e-8), 0.0, 2.0)), float(np.clip(compactness, 0.0, 1.0))
    except Exception:
        return 0.0, 0.0, 0.0

def _vertical_horizontal_edge_energy(gray: np.ndarray) -> Tuple[float, float]:
    """Compute energy concentrated in vertical vs horizontal Sobel responses."""
    try:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        v_energy = np.sum(np.abs(gy))
        h_energy = np.sum(np.abs(gx))
        total = (v_energy + h_energy) + 1e-8
        v_frac = float(v_energy/total)
        h_frac = float(h_energy/total)
        return v_frac, h_frac
    except Exception:
        return 0.0, 0.0

# ---------------------------
# Public API
# ---------------------------
def extract_shape_structure_features(cell: np.ndarray) -> List[float]:
    """
    Extract a set of handcrafted shape & structural features from one grid cell.
    Input:
      - cell: HxW (grayscale) or HxWx3 (BGR) numpy array
    Returns:
      list of floats (feature vector)
    """
    if cell is None:
        return [0.0]*len(get_shape_feature_names())

    gray = _to_gray_uint8(cell)
    if gray is None:
        return [0.0]*len(get_shape_feature_names())

    # 1) Hough / Circle features
    circ_flag, n_circ_norm, mean_rad_norm, std_rad_norm = _circular_hough_features(gray)

    # 2) radial symmetry
    radial_score = _radial_symmetry_score(gray)

    # 3) edge orientation histogram (8 bins)
    edge_hist = _edge_orientation_histogram(gray, nbins=8)  # 8 values

    # 4) line features
    n_lines_norm, mean_len_norm, std_len_norm, verticality = _line_features(gray)

    # 5) aspect / eccentricity / solidity
    ecc, aspect, solidity = _aspect_elongation_moments(gray)

    # 6) boundary vs center contrast & center gradient
    center_border_contrast, center_grad_norm = _boundary_center_contrast(gray)

    # 7) contour features: area_norm, perimeter_norm, compactness
    area_norm, peri_norm, compactness = _contour_features(gray)

    # 8) vertical / horizontal edge energy fractions
    v_frac, h_frac = _vertical_horizontal_edge_energy(gray)

    # 9) edge orientation dominant bin index (one-hot-ish numeric)
    dominant_bin = float(np.argmax(edge_hist)) / 7.0 if sum(edge_hist) > 0 else 0.0

    # Combine into feature vector (ordered)
    feat = [
        circ_flag,
        n_circ_norm,
        mean_rad_norm,
        std_rad_norm,
        radial_score,
    ]
    feat += edge_hist  # 8
    feat += [
        dominant_bin,
        n_lines_norm,
        mean_len_norm,
        std_len_norm,
        verticality,
        ecc,
        aspect,
        solidity,
        center_border_contrast,
        center_grad_norm,
        area_norm,
        peri_norm,
        compactness,
        v_frac,
        h_frac
    ]
    # ensure length consistency
    assert len(feat) == len(get_shape_structure_feature_names()), f"feature length mismatch {len(feat)} vs {len(get_shape_structure_feature_names())}"
    # cast to floats
    feat = [float(x) for x in feat]
    return feat

def get_shape_structure_feature_names() -> List[str]:
    """
    Return ordered names for the features generated by extract_shape_features.
    """
    names = [
        "shape_hough_circle_flag",
        "shape_hough_n_circles_norm",
        "shape_hough_mean_radius_norm",
        "shape_hough_std_radius_norm",
        "shape_radial_symmetry",
    ]
    # orientation histogram bins
    for i in range(8):
        names.append(f"shape_edge_orient_hist_{i}")
    names += [
        "shape_edge_dominant_bin_norm",
        "shape_hough_n_lines_norm",
        "shape_hough_mean_len_norm",
        "shape_hough_std_len_norm",
        "shape_line_verticality",
        "shape_eccentricity",
        "shape_aspect_ratio",
        "shape_solidity",
        "shape_center_minus_border_contrast",
        "shape_center_grad_norm",
        "shape_contour_area_norm",
        "shape_contour_perimeter_norm",
        "shape_compactness",
        "shape_vertical_edge_frac",
        "shape_horizontal_edge_frac"
    ]
    return names
