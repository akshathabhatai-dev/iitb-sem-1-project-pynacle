import cv2
import numpy as np
import random
import pandas as pd
from typing import Tuple, Optional
from collections import Counter

import cv2
import numpy as np

import uuid
import time
import string

# ---------------------------------------------
# SAME AUGMENT FUNCTIONS (operate on one cell)
# ---------------------------------------------

def rotate_cell(cell, angle_range=(-15, 15)):
    angle = random.uniform(*angle_range)
    h, w = cell.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(cell, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

def adjust_brightness(cell, factor_range=(0.7, 1.3)):
    factor = random.uniform(*factor_range)
    hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] *= factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def adjust_contrast(cell, factor_range=(0.7, 1.3)):
    factor = random.uniform(*factor_range)
    return cv2.convertScaleAbs(cell, alpha=factor, beta=0)

def gaussian_blur(cell, sigma_range=(0.0, 1.0)):
    sigma = random.uniform(*sigma_range)
    if sigma <= 0:
        return cell
    ksize = int(2 * round(sigma) + 1)
    return cv2.GaussianBlur(cell, (ksize, ksize), sigma)

def add_noise(cell, noise_level=10):
    noise = np.random.randint(-noise_level, noise_level, cell.shape, dtype=np.int16)
    noisy = cell.astype(np.int16) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def translate_cell(cell, shift_range=(-2, 2)):
    tx = random.randint(*shift_range)
    ty = random.randint(*shift_range)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    h, w = cell.shape[:2]
    return cv2.warpAffine(cell, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

def scale_cell(cell, scale_range=(0.9, 1.1)):
    scale = random.uniform(*scale_range)
    h, w = cell.shape[:2]
    scaled = cv2.resize(cell, None, fx=scale, fy=scale)
    sh, sw = scaled.shape[:2]

    # crop center if bigger
    if sh >= h and sw >= w:
        cy = (sh - h) // 2
        cx = (sw - w) // 2
        return scaled[cy:cy+h, cx:cx+w]

    # pad if smaller
    result = np.zeros_like(cell)
    y_off = (h - sh) // 2
    x_off = (w - sw) // 2
    result[y_off:y_off+sh, x_off:x_off+sw] = scaled
    return result

def flip_cell(cell):
    return cv2.flip(cell, 1)


# ---------------------------------------------------------
# FUNCTION: AUGMENT ONE CELL (random handcrafted pipeline)
# ---------------------------------------------------------

def augment_cell(cell, num_ops=3, class_no=1):
    
    rotate_cell_options = [(-15, 15), (-30, 30)] if class_no == 1 else [(-10, 10), (-20, 20)]
    adjust_brightness_options = [(0.7, 1.3), (0.5, 1.5)] if class_no == 1 else [(0.8, 1.2), (0.6, 1.4)]
    adjust_contrast_options = [(0.7, 1.3), (0.5, 1.5)] if class_no == 1 else [(0.8, 1.2), (0.6, 1.4)]
    gaussian_blur_options = [(0.0, 1.0), (0.5, 2.0)] if class_no == 1 else [(0.0, 0.5), (0.3, 1.5)]
    translate_cell_options = [(-2, 2), (-4, 4)] if  class_no == 1 else [(-1, 1), (-3, 3)]
    scale_cell_options = [(0.9, 1.1), (0.8, 1.2)] if class_no == 1 else [(0.95, 1.05), (0.85, 1.15)]
    ops = [
        lambda c: rotate_cell(c, random.choice(rotate_cell_options)),
        lambda c: adjust_brightness(c, random.choice(adjust_brightness_options)),
        lambda c: adjust_contrast(c, random.choice(adjust_contrast_options)),
        lambda c: gaussian_blur(c, random.choice(gaussian_blur_options)),
        lambda c: add_noise(c, noise_level=10 if class_no == 1 else 5),
        lambda c: translate_cell(c, random.choice(translate_cell_options)),
        lambda c: scale_cell(c, random.choice(scale_cell_options)),
        flip_cell
    ]
    # ops = [
    #     rotate_cell,
    #     adjust_brightness,
    #     adjust_contrast,
    #     gaussian_blur,
    #     add_noise,
    #     translate_cell,
    #     scale_cell,
    #     flip_cell
    # ]

    augmented = cell.copy()
    chosen = random.sample(ops, num_ops)

    for op in chosen:
        augmented = op(augmented)

    return augmented


# --------------------------------------------------------------
# FUNCTION: AUGMENT N CELLS (return all augmentations in-memory)
# --------------------------------------------------------------

def augment_cells(cells, num_augmented=3, num_ops=3):
    """
    cells: list or array of N image cells
    num_augmented: number of augmented versions per cell
    num_ops: number of augmentation operations per augmented sample

    RETURNS:
      A list of lists:
      [
         [aug1_cell0, aug2_cell0, ...],
         [aug1_cell1, aug2_cell1, ...],
         ...
      ]
    """
    all_augmented = []

    for cell in cells:
        aug_list = []
        for _ in range(num_augmented):
            aug_list.append(augment_cell(cell, num_ops=num_ops))
        all_augmented.append(aug_list)

    return all_augmented

def filter_cells_for_augmentation(cells, labels, target_class=1):
    """
    Filters grid cells that belong to the target class (e.g., class 1 = Ball).

    Parameters
    ----------
    cells : list or array
        List of grid cell images for ONE image. Example: 64 cells from 8×8 split.
    
    labels : list or array
        Corresponding labels for each grid cell. Example: [0,1,0,0,1,...]
    
    target_class : int
        Class ID to filter for augmentation (default = 1 for Ball).

    Returns
    -------
    selected_cells : list
        List of cells that belong ONLY to the target class.
    
    selected_indices : list
        The grid indices of the selected cells (useful for mapping later).
    
    selected_labels : list
        The labels (all equal to target_class).
    """

    selected_cells = []
    selected_indices = []
    selected_labels = []

    for idx, (cell, label) in enumerate(zip(cells, labels)):
        if label == target_class:
            selected_cells.append(cell)
            selected_indices.append(idx)
            selected_labels.append(label)

    return selected_cells, selected_indices, selected_labels

def generate_augmented_cells(cell, num_augmented=5, class_no=1):
    """
    Returns a list: [aug1, aug2, ..., augN]
    Does NOT save anything to disk.
    """

    return [augment_cell(cell, class_no=class_no) for _ in range(num_augmented)]

# def apply_augmentated_oversampling(
#     cells: list,
#     labels: list,
#     target_count_ratio: float = .85,
#     random_seed: int = 42,
#     verbose: bool = True
# ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
#     """
#     Apply oversampling to balance class distribution by oversampling Class 1 (Ball).

#     Args:
#         X: Feature matrix (n_samples, n_features)
#         y: Label array (n_samples,)
#         df: DataFrame with columns [image_name, cell_index, label, features...]
#         target_count_ratio: Ratio to determine target count for Class 1
#         random_seed: Random seed for reproducibility
#         verbose: Print statistics
#     Returns:
#         Tuple of (X_oversampled, y_oversampled, df_oversampled)
#     """

#     np.random.seed(random_seed)

#     if verbose:
#         print("="*70)
#         print("APPLYING OVERSAMPLING TO CLASS 1")
#         print("="*70)
#         print(f"Target count ratio: {target_count_ratio}")
#         print(f"Random seed: {random_seed}")
#         print(f"\nOriginal class distribution:")
#         # print_class_distribution(y)

#     # Apply oversampling to Class 1
#     # X_over, y_over, df_over = oversample_class_1(
#     #     X, y, df,
#     #     target_count_ratio=target_count_ratio,
#     #     random_seed=random_seed,
#     #     verbose=verbose
#     # )

#     class1_idx = np.where(labels == 1)[0]
#     current_count = len(class1_idx)

#     # Calculate target count
#     class_counts = Counter(labels)
#     minority_classes = [0, 2, 3]  # no objects, bat, stump
#     # target_count = 3*max(class_counts[cls] for cls in minority_classes)
#     target_count = int(target_count_ratio*max(class_counts[cls] for cls in minority_classes))


#     if current_count >= target_count:
#         if verbose:
#             print(f"\nClass 1 already has {current_count} samples ≥ target {target_count}. No oversampling done.")
#         return cells, labels
    
#     augmented_ball_cells = []

#     ball_cells, ball_idx, _ = filter_cells_for_augmentation(cells, labels, target_class=1)

#     for cell in ball_cells:
#         aug_versions = generate_augmented_cells(cell, num_augmented=5)
#         augmented_ball_cells.extend(aug_versions)

#     more_data_count = target_count - current_count
#     # augmented_cells = generate_augmented_cells(cell, num_augmented=more_data_count)
#     augmented = augment_cells(cells, num_augmented=more_data_count)
#     # sm = SMOTE(k_neighbors=2, sampling_strategy={1:target_count, 2:target_count}, random_state=random_seed)
#     # sm = SMOTE(k_neighbors=2, sampling_strategy={1:target_count}, random_state=random_seed)
#     # X_over, y_over = sm.fit_resample(X, y)



#     if verbose:
#         print(f"\nFinal dataset size after oversampling: {len(y_over):,} samples")
#         print("="*70)

#     return X_over, y_over, df



def generate_random_image_name(ext=".png"):
    """
    Generates a unique random image filename.
    
    Parameters
    ----------
    ext : str
        File extension (".png", ".jpg", ".jpeg", etc.)
    
    Returns
    -------
    str
        Random unique filename with extension
    """
    
    # random string part
    rand_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    
    # timestamp for uniqueness
    timestamp = str(int(time.time() * 1000))
    
    # UUID part to ensure no collisions
    uid = uuid.uuid4().hex[:8]
    
    return f"img_{timestamp}_{rand_str}_{uid}{ext}"


