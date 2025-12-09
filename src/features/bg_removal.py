import cv2
import numpy as np

import cv2
import numpy as np

def best_cricket_mask(cell):
    H, W = cell.shape[:2]

    # ------------------------------------------------------------
    # 1. COLOR MODEL MASKS (HSV)
    # ------------------------------------------------------------
    hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)

    # Grass mask (remove)
    grass_low  = np.array([25, 40, 40])
    grass_high = np.array([95, 255, 255])
    mask_grass = cv2.inRange(hsv, grass_low, grass_high)

    # Red ball mask
    ball_red1 = cv2.inRange(hsv, np.array([0, 70, 50]),  np.array([10, 255, 255]))
    ball_red2 = cv2.inRange(hsv, np.array([160, 70, 50]), np.array([180, 255, 255]))
    mask_ball = cv2.bitwise_or(ball_red1, ball_red2)

    # White ball / white pads / kit
    mask_white = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 60, 255]))

    # Light wood (stump / bat blade)
    mask_wood = cv2.inRange(hsv, np.array([7, 20, 80]), np.array([30, 170, 255]))

    color_fg = cv2.bitwise_or(mask_ball, mask_white)
    color_fg = cv2.bitwise_or(color_fg, mask_wood)

    # ------------------------------------------------------------
    # 2. EDGE-SKELETON MAP (foreground preservation)
    # ------------------------------------------------------------
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 120)
    sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 1)
    sobel = cv2.normalize(np.abs(sobel), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Edge foreground: if edge density > 3–5%
    edge_density = cv2.blur(edges.astype(float), (7,7))
    edge_fg = (edge_density > 15).astype(np.uint8) * 255

    # ------------------------------------------------------------
    # 3. LOCAL CONTRAST (Weber contrast)
    # ------------------------------------------------------------
    blur = cv2.GaussianBlur(gray, (9,9), 0)
    contrast = np.abs(gray - blur) / (blur + 1)
    contrast = cv2.normalize(contrast, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    contrast_fg = (contrast > 35).astype(np.uint8) * 255  # threshold tuned empirically

    # ------------------------------------------------------------
    # 4. GRADIENT ENTROPY MAP
    # ------------------------------------------------------------
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy).astype(np.uint8)

    # entropy in 9×9 window
    entropy = cv2.blur(mag, (9,9))
    entropy_fg = (entropy > 30).astype(np.uint8) * 255

    # ------------------------------------------------------------
    # 5. COMBINE ALL MASKS (MAJORITY VOTE)
    # ------------------------------------------------------------
    masks = [color_fg, edge_fg, contrast_fg, entropy_fg]
    combined = sum([(m > 0).astype(np.uint8) for m in masks])
    combined = (combined >= 2).astype(np.uint8) * 255  # at least 2 sources agree

    # Remove grass
    combined = cv2.bitwise_and(combined, cv2.bitwise_not(mask_grass))

    # ------------------------------------------------------------
    # 6. MORPHOLOGICAL REFINEMENT
    # ------------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

    # ------------------------------------------------------------
    # 7. KEEP LARGEST CONNECTED COMPONENT
    # ------------------------------------------------------------
    n, labels, stats, _ = cv2.connectedComponentsWithStats(combined)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_id = 1 + np.argmax(areas)
        final = np.where(labels == max_id, 255, 0).astype(np.uint8)
    else:
        final = combined

    return final


def remove_background_hybrid(cell_bgr):
    h, w = cell_bgr.shape[:2]

    # ---------------------
    # 1. SALIENCY MAP (handcrafted)
    # ---------------------
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, sal_map = saliency.computeSaliency(cell_bgr)
    sal_map = (sal_map * 255).astype("uint8")
    _, sal_mask = cv2.threshold(sal_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ---------------------
    # 2. HSV background removal (remove grass)
    # ---------------------
    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 30, 30])
    upper_green = np.array([90, 255, 255])
    grass_mask = cv2.inRange(hsv, lower_green, upper_green)
    fg_mask_hsv = cv2.bitwise_not(grass_mask)

    # ---------------------
    # 3. Dark background removal (stadium stands)
    # ---------------------
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    _, dark_mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)

    # ---------------------
    # 4. Combine all masks
    # ---------------------
    combined = cv2.bitwise_or(sal_mask, fg_mask_hsv)
    combined = cv2.bitwise_and(combined, dark_mask)

    # ---------------------
    # 5. Use combined mask as GrabCut initialization
    # ---------------------
    mask_gc = np.where(combined > 0, cv2.GC_PR_FGD, cv2.GC_BGD).astype('uint8')

    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    try:
        cv2.grabCut(cell_bgr, mask_gc, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    except:
        # Fallback: use combined mask
        final_mask = combined
        return final_mask

    final_mask = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype('uint8')

    # ---------------------
    # 6. Morphological cleanup
    # ---------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

    return final_mask

def spectral_residual_saliency(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    # Fourier Transform
    img_fft = np.fft.fft2(gray)
    magnitude = np.abs(img_fft)
    log_magnitude = np.log(magnitude + 1e-8)
    phase = np.angle(img_fft)
    # Spectral Residual
    avg = cv2.blur(log_magnitude, (3,3))
    residual = log_magnitude - avg
    # Reconstruct
    saliency = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase)))**2
    saliency = cv2.GaussianBlur(saliency, (9,9), 2.5)
    saliency = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return saliency

def ensure_bgr(img):
    # If already BGR (3 channels)
    if len(img.shape) == 3 and img.shape[2] == 3:
        return img
    # If grayscale → convert to BGR
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    raise ValueError("Unsupported image format for ensure_bgr()")

def remove_background_cell(cell_bgr):
    cell_bgr = ensure_bgr(cell_bgr)
    h, w = cell_bgr.shape[:2]

    # 1. HSV remove grass (green)
    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    grass = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.bitwise_not(grass)   # foreground approx

    # 2. Remove very dark background (stadium stands)
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask, dark)

    # 3. Keep only largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        sizes = stats[1:, cv2.CC_STAT_AREA]
        max_label = 1 + np.argmax(sizes)
        mask = np.zeros_like(mask)
        mask[labels == max_label] = 255

    # 4. Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

import cv2
import numpy as np


def handcrafted_cricket_cell_mask(cell_bgr):
    """
    Best handcrafted background-removal mask for small cricket grid cells.
    
    Combines:
      - HSV grass removal
      - Gradient saliency (Sobel magnitude)
      - FFT spectral residual saliency (dense)
      - Morphological filtering
      - Largest connected component retention
    
    Returns:
      uint8 mask (0 or 255) of same size as cell.
    """

    H, W = cell_bgr.shape[:2]

    # ---------------------------------------------------------------------
    # 1. HSV GRASS REMOVAL  (remove background grass)
    # ---------------------------------------------------------------------
    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    lower_grass = np.array([25, 40, 40])
    upper_grass = np.array([95, 255, 255])
    grass_mask = cv2.inRange(hsv, lower_grass, upper_grass)
    fg_hsv = cv2.bitwise_not(grass_mask)  # non-grass = possible foreground

    # ---------------------------------------------------------------------
    # 2. GRADIENT SALIENCY (edges = interesting objects)
    # ---------------------------------------------------------------------
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(sobelx, sobely)

    grad_mag = cv2.GaussianBlur(grad_mag, (5, 5), 1)
    grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # adaptive threshold using very LOW percentile
    p = 65
    th_grad = np.percentile(grad_mag, p)
    grad_mask = (grad_mag >= th_grad).astype(np.uint8) * 255

    # ---------------------------------------------------------------------
    # 3. FFT SPECTRAL RESIDUAL SALIENCY  (dense foreground indicator)
    # ---------------------------------------------------------------------
    small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype('float32')

    fft = np.fft.fft2(small)
    mag = np.abs(fft)
    log_mag = np.log(mag + 1e-8)
    phase = np.angle(fft)
    avg_log = cv2.blur(log_mag, (3, 3))
    spec_residual = log_mag - avg_log
    sal = np.abs(np.fft.ifft2(np.exp(spec_residual + 1j * phase))) ** 2

    sal = cv2.GaussianBlur(sal, (9, 9), 2)
    sal = cv2.normalize(sal, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    sal = cv2.resize(sal, (W, H), interpolation=cv2.INTER_LINEAR)

    p2 = 60   # softer threshold
    th_sal = np.percentile(sal, p2)
    sal_mask = (sal >= th_sal).astype(np.uint8) * 255

    # ---------------------------------------------------------------------
    # 4. COMBINE ALL MASKS
    # ---------------------------------------------------------------------
    combined = cv2.bitwise_or(fg_hsv, grad_mask)
    combined = cv2.bitwise_or(combined, sal_mask)

    # ---------------------------------------------------------------------
    # 5. MORPHOLOGICAL SMOOTHING
    # ---------------------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

    # ---------------------------------------------------------------------
    # 6. KEEP LARGEST CONNECTED COMPONENT
    # ---------------------------------------------------------------------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)

    if num_labels > 1:
        sizes = stats[1:, cv2.CC_STAT_AREA]
        max_idx = 1 + int(np.argmax(sizes))
        final_mask = np.where(labels == max_idx, 255, 0).astype(np.uint8)
    else:
        final_mask = combined

    # ---------------------------------------------------------------------
    # 7. SAFETY CLEANUP & OUTPUT
    # ---------------------------------------------------------------------
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    final_mask = final_mask.astype(np.uint8)

    return final_mask


def apply_mask(img, mask):
    # Ensure grayscale mask
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Resize mask if needed
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Convert to uint8
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Ensure mask contains 0 or 255 only
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    # Apply mask
    return cv2.bitwise_and(img, img, mask=mask)

def apply_mask_old(img, mask):
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype("uint8") * 255
    return cv2.bitwise_and(img, img, mask=mask)


# ---------- 1) color-based background removal (HSV) ----------
def remove_background_by_hsv(cell_bgr: np.ndarray,
                             lower_hsv=(25, 40, 40),
                             upper_hsv=(90, 255, 255),
                             invert=False) -> np.ndarray:
    """
    Remove background by thresholding in HSV. Default range approximates green (grass).
    - Returns binary mask (uint8 0/255) where foreground = 255.
    - invert=True will invert mask (useful if object is also green).
    """
    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)  # mask for "background" if default = grass
    # If mask is background, invert to get foreground:
    if not invert:
        mask_fg = cv2.bitwise_not(mask)
    else:
        mask_fg = mask
    # clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask_fg


# ---------- 2) morphology + adaptive threshold (grayscale) ----------
def remove_background_by_morph(cell_bgr: np.ndarray,
                               blur_ksize=5,
                               adapt_blocksize=31,
                               adapt_C=10) -> np.ndarray:
    """
    Use adaptive threshold + morphological ops to separate foreground.
    Good when object contrasts with background in brightness.
    Returns binary mask (uint8 0/255).
    """
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, adapt_blocksize, adapt_C)
    # keep largest connected components (remove small specks)
    nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if nb_components <= 1:
        return mask
    # keep components larger than threshold (e.g., 300 px), tune per cell size
    sizes = stats[1:, cv2.CC_STAT_AREA]
    min_size = max(50, int(0.001 * cell_bgr.shape[0] * cell_bgr.shape[1]))
    mask2 = np.zeros_like(mask)
    for i, s in enumerate(sizes):
        if s >= min_size:
            mask2[labels == (i+1)] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask2


# ---------- 3) GrabCut (refinement) ----------
def remove_background_by_grabcut(cell_bgr: np.ndarray,
                                 iter_count=5,
                                 rect_ratio=0.9) -> np.ndarray:
    """
    GrabCut with an initial rect covering most of the image.
    rect_ratio: portion of image to mark as probable foreground.
    Returns mask (uint8 0/255).
    """
    h, w = cell_bgr.shape[:2]
    # initial rect: shrink border to avoid edges
    pad_h = int((1 - rect_ratio) * h / 2)
    pad_w = int((1 - rect_ratio) * w / 2)
    rect = (pad_w, pad_h, w - 2*pad_w - 1, h - 2*pad_h - 1)
    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    try:
        cv2.grabCut(cell_bgr, mask, rect, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_RECT)
    except Exception:
        # fallback: full mask
        return np.ones((h, w), dtype=np.uint8) * 255
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    mask2 = mask2 * 255
    # refine
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask2


# ---------- 4) MOG2 background subtraction (for video) ----------
def remove_background_by_mog2(frame, mog=None):
    """
    For video: supply a cv2.createBackgroundSubtractorMOG2() object once,
    call this per-frame to get mask. Returns 0/255 mask.
    """
    if mog is None:
        mog = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    mask = mog.apply(frame)
    # clean
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask, mog


# ---------- 5) combine heuristics (recommended pipeline) ----------
def compute_foreground_mask(cell_bgr: np.ndarray,
                            use_hsv=True,
                            use_morph=True,
                            use_grabcut=True) -> np.ndarray:
    """
    Combine methods: coarse HSV to remove grass -> adaptive morph -> refine with grabcut.
    Returns binary mask.
    """
    masks = []
    if use_hsv:
        hsv_mask = remove_background_by_hsv(cell_bgr)
        masks.append(hsv_mask)
    if use_morph:
        morph_mask = remove_background_by_morph(cell_bgr)
        masks.append(morph_mask)
    if use_grabcut:
        try:
            gc_mask = remove_background_by_grabcut(cell_bgr)
            masks.append(gc_mask)
        except Exception:
            pass

    if not masks:
        h, w = cell_bgr.shape[:2]
        return np.ones((h, w), dtype=np.uint8) * 255

    # combine: OR of all masks (conservative: keep any foreground)
    combined = np.zeros_like(masks[0])
    for m in masks:
        combined = cv2.bitwise_or(combined, m)

    # final cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    return combined
