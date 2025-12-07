import numpy as np
from skimage.feature import hog
from typing import List

def compute_hog_per_cell(gray_cells, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """
    Compute HOG for each cell in the 8x8 grid.
    Returns list of 64 HOG vectors.
    """
    hog_list = []
    # print(gray_cells[0].shape)
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
        # h = hog(
        #     gray,
        #     orientations=orientations,
        #     pixels_per_cell=pixels_per_cell,
        #     cells_per_block=(1, 2),
        #     block_norm='L2-Hys',
        #     visualize=False,
        #     feature_vector=True
        # )
        hog_list.append(h.astype(np.float32))
    return hog_list

def extract_na_aware_hog_for_grid_features(hog_list, grid_size=8):
    """
    Fixed-size NA-HOG: always uses 8 neighbors.
    Missing neighbors are padded with zeros.
    """
    hog_dim = len(hog_list[0])
    zero_vec = np.zeros(hog_dim, dtype=np.float32)

    directions = [(-1,-1),(-1,0),(-1,1),
                  ( 0,-1),        ( 0,1),
                  ( 1,-1),( 1,0),( 1,1)]

    na_hog_list = []

    for idx in range(len(hog_list)):
        r = idx // grid_size
        c = idx % grid_size

        center = hog_list[idx]
        neighbors = []

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            # if neighbor inside grid → use real HOG
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                neighbors.append(hog_list[nr * grid_size + nc])
            else:
                # if missing → pad with ZERO vector
                neighbors.append(zero_vec)

        # now ALWAYS 1 + 8 = 9 blocks
        combined = np.hstack([center] + neighbors)
        na_hog_list.append(combined.astype(np.float32))

    return na_hog_list

def extract_na_aware_hog_for_grid_features_old1(hog_list, grid_size=8):
    """
    Compute neighborhood-aware HOG for all 64 cells.
    hog_list: list of 64 HOG vectors for each cell.
    """
    na_hog = []
    
    directions = [(-1,-1),(-1,0),(-1,1),
                  ( 0,-1),       ( 0,1),
                  ( 1,-1),(1,0),(1,1)]
    
    for idx in range(len(hog_list)):
        r = idx // grid_size
        c = idx % grid_size
        
        center = hog_list[idx]
        neighbors = []

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                neighbors.append(hog_list[nr * grid_size + nc])
        
        # Fixed-length: center + neighbors *always the same dims*
        combined = np.hstack([center] + neighbors)
        na_hog.append(combined)
    
    return na_hog    # list of 64 NA-HOG vectors


def extract_na_aware_hog_for_grid_features_old(hog_list, grid_size=8):
    """
    Compute neighborhood-aware HOG for all 64 cells.
    hog_list: list of 64 HOG vectors for each cell.
    """
    hog_list = np.array(hog_list, dtype=object)  # 64 objects
    hog_na = []

    directions = [(-1,-1), (-1,0), (-1,1),
                  ( 0,-1),         ( 0,1),
                  ( 1,-1), ( 1,0), ( 1,1)]

    idx = 0
    for r in range(grid_size):
        for c in range(grid_size):

            center = hog_list[idx]
            neighbors = []

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    n_idx = nr * grid_size + nc
                    neighbors.append(hog_list[n_idx])

            combined = np.hstack([center] + neighbors)
            hog_na.append(combined.astype(np.float32))

            idx += 1

    return hog_na    # list of 64 NA-HOG vectors

def compute_hog_dim_per_cell(
    cell_height=75,
    cell_width=100,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2)
):
    """
    Compute HOG feature vector length per cell using standard skimage HOG formula.

    Returns:
        int: final HOG feature length for 1 cell
    """

    pc_h, pc_w = pixels_per_cell
    bc_h, bc_w = cells_per_block

    # 1️⃣ Number of "HOG cells" inside the image cell
    H_cells = cell_height // pc_h
    W_cells = cell_width // pc_w

    # 2️⃣ Number of HOG blocks
    H_blocks = H_cells - bc_h + 1
    W_blocks = W_cells - bc_w + 1
    total_blocks = H_blocks * W_blocks

    # 3️⃣ Features per block
    block_feature_len = orientations * (bc_h * bc_w)

    # 4️⃣ Total HOG size
    hog_dim = total_blocks * block_feature_len

    return hog_dim

def get_na_aware_hog_feature_names(neighbor_count=8) -> List[str]:
    """
    Get the names of ORB features.

    Returns:
        List of 16384 feature names
    """
    
    return [f'orb_bin_{i}' for i in range(compute_hog_dim_per_cell() * (1 + neighbor_count))]  # 1 center + neighbors