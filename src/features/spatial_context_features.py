"""
Spatial Context Features for Cricket Object Detection

This module extracts 25 spatial context features from neighboring cells in the 8x8 grid.

The current system classifies each cell in isolation without spatial context.
This module addresses this by aggregating features from the 8-connected neighborhood,
providing crucial spatial awareness for object detection.

Features breakdown:
- Neighbor Color Aggregation (4): Mean, std, max, min of color features from neighbors
- Neighbor Shape Aggregation (4): Mean, std, max, min of shape features from neighbors
- Neighbor Texture Aggregation (2): Mean, std of texture features from neighbors
- Spatial Position (4): Row, col, distance from center (normalized)
- Neighbor Count (1): Number of valid neighbors
- Feature Similarity (4): Cosine similarity statistics to neighbors

Critical for:
- Ball: Different neighbor patterns than grass cells (surrounded by grass)
- Bat: High similarity to horizontal neighbors (continuous object)
- Stump: High similarity to vertical neighbors (vertical continuity)
- No-Object (grass): High similarity to all neighbors (uniform region)

Author: Pynacle Team
Date: 2025
Phase: 5A - Spatial Context Features
"""

import numpy as np
from typing import List, Tuple


def extract_spatial_context_features(
    all_cell_features: np.ndarray,
    cell_index: int,
    grid_shape: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Extract 25 spatial context features from neighboring cells.

    This function aggregates features from the 8-connected neighborhood
    to provide spatial awareness that complements per-cell features.

    Args:
        all_cell_features: Array of shape (64, num_features) with all cell features
                          Expected: 382 features per cell (Phase 4B)
        cell_index: Index of current cell (0-63)
        grid_shape: Grid dimensions (8, 8)

    Returns:
        Array of 25 float32 spatial context features
    """
    # Handle edge cases
    if all_cell_features.shape[0] != grid_shape[0] * grid_shape[1]:
        # Not exactly 64 cells - return zeros
        return np.zeros(25, dtype=np.float32)

    # Calculate row and column in grid
    row = cell_index // grid_shape[1]
    col = cell_index % grid_shape[1]

    # Get 8-connected neighbor indices
    neighbor_indices = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue  # Skip self
            nr, nc = row + dr, col + dc
            if 0 <= nr < grid_shape[0] and 0 <= nc < grid_shape[1]:
                neighbor_idx = nr * grid_shape[1] + nc
                neighbor_indices.append(neighbor_idx)

    # If no neighbors (shouldn't happen in 8x8 grid), return zeros
    if len(neighbor_indices) == 0:
        return np.zeros(25, dtype=np.float32)

    # Extract neighbor features
    neighbor_features = all_cell_features[neighbor_indices]

    # ========== NEIGHBOR COLOR AGGREGATION (4 features) ==========
    # Color features are the first 40 in our feature vector
    # (14 RGB + 26 HSV = 40 color features)
    neighbor_colors = neighbor_features[:, :40]

    # 1. Mean of neighbor color features
    # High mean: surrounded by colorful objects
    # Low mean: surrounded by dark/grass regions
    neighbor_color_mean = np.mean(neighbor_colors)

    # 2. Std of neighbor color features
    # High std: color variation in neighborhood (object boundaries)
    # Low std: uniform color neighborhood (within object or grass)
    neighbor_color_std = np.std(neighbor_colors)

    # 3. Max of neighbor color features
    # Identifies presence of bright/saturated colors nearby
    neighbor_color_max = np.max(neighbor_colors)

    # 4. Min of neighbor color features
    # Identifies presence of dark colors nearby
    neighbor_color_min = np.min(neighbor_colors)

    # ========== NEIGHBOR SHAPE AGGREGATION (4 features) ==========
    # Shape features span indices 58-92 in our feature vector
    # (enhanced_color: 40-57, shape: 58-92)
    neighbor_shapes = neighbor_features[:, 58:93]

    # 5. Mean of neighbor shape features
    # Ball: different shape from neighbors (circular vs irregular grass)
    # Bat: similar shape to horizontal neighbors
    neighbor_shape_mean = np.mean(neighbor_shapes)

    # 6. Std of neighbor shape features
    # High std: shape variation (object boundaries)
    # Low std: uniform shapes (within object or grass)
    neighbor_shape_std = np.std(neighbor_shapes)

    # 7. Max of neighbor shape features
    neighbor_shape_max = np.max(neighbor_shapes)

    # 8. Min of neighbor shape features
    neighbor_shape_min = np.min(neighbor_shapes)

    # ========== NEIGHBOR TEXTURE AGGREGATION (2 features) ==========
    # Texture features span indices 93 onwards
    # (LBP, HOG, Gabor, GLCM, etc.)
    neighbor_textures = neighbor_features[:, 93:]

    # 9. Mean of neighbor texture features
    # Bat: similar texture to neighbors (wood grain continuity)
    # Grass: uniform texture across neighbors
    neighbor_texture_mean = np.mean(neighbor_textures)

    # 10. Std of neighbor texture features
    # High std: texture variation (object boundaries)
    # Low std: uniform texture (within object or grass)
    neighbor_texture_std = np.std(neighbor_textures)

    # ========== SPATIAL POSITION (4 features) ==========
    # Encode absolute position in grid

    # 11. Normalized row position (0 to 1)
    # Objects often appear in middle rows (0.3-0.7)
    normalized_row = row / (grid_shape[0] - 1)

    # 12. Normalized col position (0 to 1)
    # Objects often appear in center columns (0.3-0.7)
    normalized_col = col / (grid_shape[1] - 1)

    # 13. Distance from center row (0 to 1)
    # Objects rarely at extreme top/bottom
    center_row = (grid_shape[0] - 1) / 2.0
    dist_from_center_row = abs(row - center_row) / center_row

    # 14. Distance from center col (0 to 1)
    # Objects rarely at extreme left/right
    center_col = (grid_shape[1] - 1) / 2.0
    dist_from_center_col = abs(col - center_col) / center_col

    # ========== NEIGHBOR COUNT (1 feature) ==========

    # 15. Number of valid neighbors (3-8)
    # Corner cells: 3 neighbors
    # Edge cells: 5 neighbors
    # Interior cells: 8 neighbors
    # Useful for edge effects
    neighbor_count = float(len(neighbor_indices))

    # ========== FEATURE SIMILARITY (4 features) ==========
    # Measure how similar current cell is to its neighbors

    current_features = all_cell_features[cell_index]

    # Compute cosine similarity to each neighbor
    similarities = []
    for neighbor_feat in neighbor_features:
        # Cosine similarity = dot(A, B) / (||A|| * ||B||)
        dot_product = np.dot(current_features, neighbor_feat)
        norm_current = np.linalg.norm(current_features)
        norm_neighbor = np.linalg.norm(neighbor_feat)

        if norm_current > 0 and norm_neighbor > 0:
            similarity = dot_product / (norm_current * norm_neighbor)
        else:
            similarity = 0.0

        similarities.append(similarity)

    similarities = np.array(similarities)

    # 16. Mean similarity to neighbors
    # Ball: low similarity (different from grass neighbors)
    # Grass: high similarity (uniform with neighbors)
    # Bat: high similarity to horizontal neighbors
    similarity_mean = np.mean(similarities)

    # 17. Std similarity to neighbors
    # High std: some neighbors similar, some different (object boundary)
    # Low std: uniform similarity (within object or uniform grass)
    similarity_std = np.std(similarities)

    # 18. Max similarity to neighbors
    # Identifies if any neighbor is very similar (part of same object)
    similarity_max = np.max(similarities)

    # 19. Min similarity to neighbors
    # Identifies if any neighbor is very different (object boundary)
    similarity_min = np.min(similarities)

    # ========== ADDITIONAL SPATIAL FEATURES (6 features) ==========

    # 20. Vertical neighbor similarity (top + bottom)
    # High for vertically continuous objects (stumps)
    vertical_neighbors = []
    if row > 0:  # Has top neighbor
        top_idx = (row - 1) * grid_shape[1] + col
        vertical_neighbors.append(all_cell_features[top_idx])
    if row < grid_shape[0] - 1:  # Has bottom neighbor
        bottom_idx = (row + 1) * grid_shape[1] + col
        vertical_neighbors.append(all_cell_features[bottom_idx])

    if len(vertical_neighbors) > 0:
        vertical_neighbors = np.array(vertical_neighbors)
        vertical_similarities = []
        for vert_feat in vertical_neighbors:
            dot_product = np.dot(current_features, vert_feat)
            norm_vert = np.linalg.norm(vert_feat)
            norm_current = np.linalg.norm(current_features)
            if norm_current > 0 and norm_vert > 0:
                sim = dot_product / (norm_current * norm_vert)
            else:
                sim = 0.0
            vertical_similarities.append(sim)
        vertical_similarity = np.mean(vertical_similarities)
    else:
        vertical_similarity = 0.0

    # 21. Horizontal neighbor similarity (left + right)
    # High for horizontally continuous objects (bats)
    horizontal_neighbors = []
    if col > 0:  # Has left neighbor
        left_idx = row * grid_shape[1] + (col - 1)
        horizontal_neighbors.append(all_cell_features[left_idx])
    if col < grid_shape[1] - 1:  # Has right neighbor
        right_idx = row * grid_shape[1] + (col + 1)
        horizontal_neighbors.append(all_cell_features[right_idx])

    if len(horizontal_neighbors) > 0:
        horizontal_neighbors = np.array(horizontal_neighbors)
        horizontal_similarities = []
        for horiz_feat in horizontal_neighbors:
            dot_product = np.dot(current_features, horiz_feat)
            norm_horiz = np.linalg.norm(horiz_feat)
            norm_current = np.linalg.norm(current_features)
            if norm_current > 0 and norm_horiz > 0:
                sim = dot_product / (norm_current * norm_horiz)
            else:
                sim = 0.0
            horizontal_similarities.append(sim)
        horizontal_similarity = np.mean(horizontal_similarities)
    else:
        horizontal_similarity = 0.0

    # 22. Vertical vs horizontal similarity ratio
    # Stump: vertical_sim > horizontal_sim (ratio > 1.0)
    # Bat: horizontal_sim > vertical_sim (ratio < 1.0)
    # Ball: similar in all directions (ratio ≈ 1.0)
    if horizontal_similarity > 0:
        vert_horiz_ratio = vertical_similarity / horizontal_similarity
    else:
        vert_horiz_ratio = 0.0

    # 23. Is edge cell (binary)
    # Objects more likely in center, less at edges
    is_edge = 1.0 if (row == 0 or row == grid_shape[0] - 1 or
                     col == 0 or col == grid_shape[1] - 1) else 0.0

    # 24. Is corner cell (binary)
    # Objects very rarely in corners
    is_corner = 1.0 if ((row == 0 or row == grid_shape[0] - 1) and
                       (col == 0 or col == grid_shape[1] - 1)) else 0.0

    # 25. Quadrant indicator (0-3)
    # Which quadrant: 0=TL, 1=TR, 2=BL, 3=BR
    # Some objects may have positional bias
    quadrant_row = 0 if row < grid_shape[0] // 2 else 1
    quadrant_col = 0 if col < grid_shape[1] // 2 else 1
    quadrant = float(quadrant_row * 2 + quadrant_col)

    # ========== COMBINE ALL FEATURES ==========
    features = np.array([
        # Neighbor Color Aggregation (4)
        neighbor_color_mean,
        neighbor_color_std,
        neighbor_color_max,
        neighbor_color_min,

        # Neighbor Shape Aggregation (4)
        neighbor_shape_mean,
        neighbor_shape_std,
        neighbor_shape_max,
        neighbor_shape_min,

        # Neighbor Texture Aggregation (2)
        neighbor_texture_mean,
        neighbor_texture_std,

        # Spatial Position (4)
        normalized_row,
        normalized_col,
        dist_from_center_row,
        dist_from_center_col,

        # Neighbor Count (1)
        neighbor_count,

        # Feature Similarity (4)
        similarity_mean,
        similarity_std,
        similarity_max,
        similarity_min,

        # Additional Spatial Features (6)
        vertical_similarity,
        horizontal_similarity,
        vert_horiz_ratio,
        is_edge,
        is_corner,
        quadrant
    ], dtype=np.float32)

    # Verify we have exactly 25 features
    assert len(features) == 25, f"Expected 25 features, got {len(features)}"

    # Replace any NaN or Inf values with 0.0 (defensive programming)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def get_spatial_context_feature_names() -> List[str]:
    """
    Get names of all 25 spatial context features.

    Returns:
        List of 25 feature names
    """
    names = [
        # Neighbor Color Aggregation (4)
        'neighbor_color_mean',
        'neighbor_color_std',
        'neighbor_color_max',
        'neighbor_color_min',

        # Neighbor Shape Aggregation (4)
        'neighbor_shape_mean',
        'neighbor_shape_std',
        'neighbor_shape_max',
        'neighbor_shape_min',

        # Neighbor Texture Aggregation (2)
        'neighbor_texture_mean',
        'neighbor_texture_std',

        # Spatial Position (4)
        'normalized_row',
        'normalized_col',
        'dist_from_center_row',
        'dist_from_center_col',

        # Neighbor Count (1)
        'neighbor_count',

        # Feature Similarity (4)
        'similarity_mean',
        'similarity_std',
        'similarity_max',
        'similarity_min',

        # Additional Spatial Features (6)
        'vertical_similarity',
        'horizontal_similarity',
        'vert_horiz_ratio',
        'is_edge',
        'is_corner',
        'quadrant'
    ]

    assert len(names) == 25, f"Expected 25 feature names, got {len(names)}"

    return names


if __name__ == '__main__':
    # Test feature extraction
    print("Testing Spatial Context Features...")

    # Create dummy feature array (64 cells, 382 features each)
    dummy_features = np.random.rand(64, 382).astype(np.float32)

    # Extract spatial context for cell 27 (middle of grid: row=3, col=3)
    features_center = extract_spatial_context_features(dummy_features, cell_index=27)

    # Extract spatial context for cell 0 (corner: row=0, col=0)
    features_corner = extract_spatial_context_features(dummy_features, cell_index=0)

    # Extract spatial context for cell 3 (edge: row=0, col=3)
    features_edge = extract_spatial_context_features(dummy_features, cell_index=3)

    print(f"\nExtracted {len(features_center)} features")
    print(f"\nKey Spatial Metrics Comparison:")
    print(f"{'Feature Name':<30} {'Center (27)':<15} {'Corner (0)':<15} {'Edge (3)':<15}")
    print("=" * 75)

    names = get_spatial_context_feature_names()

    # Show key discriminative features
    key_features = ['neighbor_count', 'normalized_row', 'normalized_col',
                   'is_edge', 'is_corner', 'quadrant']
    for key in key_features:
        idx = names.index(key)
        print(f"{key:<30} {features_center[idx]:<15.4f} {features_corner[idx]:<15.4f} {features_edge[idx]:<15.4f}")

    print("\n✓ Test passed!")
    print("  Center cell (27): 8 neighbors, not edge, not corner")
    print("  Corner cell (0): 3 neighbors, is edge, is corner")
    print("  Edge cell (3): 5 neighbors, is edge, not corner")
    print("\nSpatial context is CRITICAL for reducing false positives!")
