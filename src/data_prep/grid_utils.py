import numpy as np
import cv2
from typing import List, Tuple


def divide_into_grid(image: np.ndarray, grid_size: Tuple[int, int] = (8, 8)) -> List[np.ndarray]:
    """
    Divide an 800x600 image into an 8x8 grid.
    Each cell will be 100x75 pixels.

    Args:
        image: Input image of shape (600, 800, 3) or (600, 800)
        grid_size: Tuple of (rows, cols), default (8, 8)

    Returns:
        List of 64 cells, each of shape (75, 100, 3) or (75, 100)
    """
    height, width = image.shape[:2]  # 600, 800
    rows, cols = grid_size  # 8, 8

    cell_height = height // rows  # 600 // 8 = 75
    cell_width = width // cols    # 800 // 8 = 100

    cells = []
    for i in range(rows):  # 0 to 7
        for j in range(cols):  # 0 to 7
            # Extract cell
            y_start = i * cell_height  # e.g., row 2: 2*75 = 150
            y_end = (i + 1) * cell_height  # 3*75 = 225
            x_start = j * cell_width  # e.g., col 3: 3*100 = 300
            x_end = (j + 1) * cell_width  # 4*100 = 400

            cell = image[y_start:y_end, x_start:x_end]
            cells.append(cell)

    return cells


def draw_grid_on_image(image: np.ndarray, grid_size: Tuple[int, int] = (8, 8),
                       color: Tuple[int, int, int] = (0, 255, 0),
                       thickness: int = 2) -> np.ndarray:
    """
    Draw grid lines on an image for visualization.

    Args:
        image: Input image
        grid_size: Tuple of (rows, cols)
        color: BGR color for grid lines
        thickness: Line thickness

    Returns:
        Image with grid overlay
    """
    img_copy = image.copy()
    height, width = img_copy.shape[:2]
    rows, cols = grid_size

    cell_height = height // rows
    cell_width = width // cols

    # Draw horizontal lines
    for i in range(1, rows):
        y = i * cell_height
        cv2.line(img_copy, (0, y), (width, y), color, thickness)

    # Draw vertical lines
    for j in range(1, cols):
        x = j * cell_width
        cv2.line(img_copy, (x, 0), (x, height), color, thickness)

    return img_copy


def get_cell_index(row: int, col: int, grid_size: Tuple[int, int] = (8, 8)) -> int:
    """
    Convert (row, col) position to cell index (0-63).

    Args:
        row: Row index (0-7)
        col: Column index (0-7)
        grid_size: Tuple of (rows, cols)

    Returns:
        Cell index (0-63)
    """
    cols_count = grid_size[1]
    return row * cols_count + col


def get_cell_position(cell_idx: int, grid_size: Tuple[int, int] = (8, 8)) -> Tuple[int, int]:
    """
    Convert cell index (0-63) to (row, col) position.

    Args:
        cell_idx: Cell index (0-63)
        grid_size: Tuple of (rows, cols)

    Returns:
        Tuple of (row, col)
    """
    cols_count = grid_size[1]
    row = cell_idx // cols_count
    col = cell_idx % cols_count
    return row, col


def validate_image_size(image: np.ndarray, expected_size: Tuple[int, int] = (800, 600)) -> bool:
    """
    Validate that image has the expected dimensions.

    Args:
        image: Input image
        expected_size: Expected (width, height)

    Returns:
        True if image size matches, False otherwise
    """
    height, width = image.shape[:2]
    expected_width, expected_height = expected_size
    return width == expected_width and height == expected_height


def resize_image(image: np.ndarray, target_size: Tuple[int, int] = (800, 600)) -> np.ndarray:
    """
    Resize image to target size using INTER_AREA interpolation (downscaling).

    Args:
        image: Input image
        target_size: Target (width, height)

    Returns:
        Resized image
    """
    target_width, target_height = target_size
    resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized
