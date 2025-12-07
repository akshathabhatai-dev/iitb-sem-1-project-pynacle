import numpy as np
from skimage.feature import hog
from typing import List


# -------------------------------------------------------------
# 1. Compute HOG for each grayscale cell in the 8×8 grid
# -------------------------------------------------------------
def compute_hog_per_cell(
    gray_cells,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2)
):
    """
    Compute **HOG (Histogram of Oriented Gradients)** for each cell
    in an 8x8 grid of image patches.

    INPUT:
        gray_cells : List of 64 grayscale cell patches (each 75×100)
        orientations : number of gradient bins
        pixels_per_cell : spatial size of each HOG cell
        cells_per_block : how many HOG cells form a normalization block

    OUTPUT:
        hog_list : List of 64 fixed-length HOG feature vectors

    NOTE:
        HOG feature length is determined by:
            - number of HOG cells along H,W
            - number of blocks produced
            - orientations × block_size
        and is constant for all cells → ensures stable ML features.
    """

    hog_list = []

    for gray in gray_cells:
        h = hog(
            gray,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm='L2-Hys',
            visualize=False,
            feature_vector=True
        )
        hog_list.append(h.astype(np.float32))

    return hog_list



# -------------------------------------------------------------
# 2. Neighborhood-Aware HOG (NA-HOG) – FIXED SIZE VERSION
# -------------------------------------------------------------
def extract_na_aware_hog_for_grid_features(hog_list, grid_size=8):
    """
    Compute **fixed-size Neighborhood-Aware HOG (NA-HOG)** for all 64 cells.

    HOW NA-HOG WORKS:
        For each cell:
            - Take its own HOG vector (center)
            - Collect HOG vectors from **8 neighboring cells**
            - If a neighbor does NOT exist (border/corner), use a ZERO vector

    WHY ZERO-PADDING?
        → Ensures every cell always produces the same NA-HOG size:
              (1 center + 8 neighbors) × hog_dim_per_cell

    INPUT:
        hog_list : list of 64 HOG feature vectors
        grid_size : normally 8 (8x8 grid)

    OUTPUT:
        na_hog_list : list of 64 NA-HOG feature vectors
                      each of shape: 9 × hog_dim
    """

    hog_dim = len(hog_list[0])
    zero_vec = np.zeros(hog_dim, dtype=np.float32)

    # 8-connected neighbors (top-left to bottom-right)
    directions = [
        (-1, -1), (-1,  0), (-1,  1),
        ( 0, -1),          ( 0,  1),
        ( 1, -1), ( 1,  0), ( 1,  1)
    ]

    na_hog_list = []

    for idx in range(len(hog_list)):
        r = idx // grid_size
        c = idx % grid_size

        center = hog_list[idx]
        neighbors = []

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                neighbors.append(hog_list[nr * grid_size + nc])
            else:
                neighbors.append(zero_vec)  # border padding

        combined = np.hstack([center] + neighbors).astype(np.float32)
        na_hog_list.append(combined)

    return na_hog_list

# -------------------------------------------------------------
# 4. Compute HOG Dimension Per Cell (Used for NA-HOG Size)
# -------------------------------------------------------------
def compute_hog_dim_per_cell(
    cell_height=75,
    cell_width=100,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2)
):
    """
    Compute the **exact HOG feature length per cell** using the
    skimage HOG dimension formula.

    Formula:
        H_cells = cell_height // pixels_per_cell[0]
        W_cells = cell_width  // pixels_per_cell[1]

        H_blocks = H_cells - block_h + 1
        W_blocks = W_cells - block_w + 1

        HOG_dim = (#blocks) × (orientations × block_h × block_w)

    Default parameters produce:
        HOG per cell = 3168 features
    """
    pc_h, pc_w = pixels_per_cell
    bc_h, bc_w = cells_per_block

    H_cells = cell_height // pc_h
    W_cells = cell_width // pc_w

    H_blocks = H_cells - bc_h + 1
    W_blocks = W_cells - bc_w + 1
    total_blocks = H_blocks * W_blocks

    block_feature_len = orientations * (bc_h * bc_w)

    return total_blocks * block_feature_len



# -------------------------------------------------------------
# 5. Generate Feature Names for NA-HOG
# -------------------------------------------------------------
def get_na_aware_hog_feature_names(neighbor_count=8) -> List[str]:
    """
    Generate feature names for Neighborhood-Aware HOG.

    Final NA-HOG length:
        hog_dim_per_cell × (1 + neighbor_count)
        Example:
            3168 × 9 = 28512 features per cell

    INPUT:
        neighbor_count : normally 8 (8 neighbors)
    OUTPUT:
        List of feature names:
            na_hog_bin_0 ... na_hog_bin_N
    """

    na_hog_dim = compute_hog_dim_per_cell() * (1 + neighbor_count)
    return [f'na_hog_bin_{i}' for i in range(na_hog_dim)]
