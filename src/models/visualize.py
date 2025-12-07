"""
Visualization module for cricket object detection.
Creates color-coded grid overlays on test images showing model predictions.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Dict, List


def get_prediction_color(prediction: int) -> Tuple[int, int, int]:
    """
    Map prediction class to BGR color.

    Args:
        prediction: Class prediction (0, 1, 2, or 3)

    Returns:
        BGR color tuple
    """
    # Color mapping (BGR format for OpenCV)
    # BGR means: (Blue, Green, Red) - NOT (Red, Green, Blue)!
    color_map = {
        0: (255, 255, 255),  # No Object - White (not drawn, just for completeness)
        1: (0, 255, 0),      # Ball - Green
        2: (0, 0, 255),      # Bat - RED (BGR: Blue=0, Green=0, Red=255)
        3: (255, 0, 0),      # Stump - BLUE (BGR: Blue=255, Green=0, Red=0)
    }

    return color_map.get(prediction, (255, 255, 255))  # Default to white


def draw_prediction_grid(image: np.ndarray,
                         predictions: np.ndarray,
                         grid_size: Tuple[int, int] = (8, 8),
                         alpha: float = 0.3,
                         grid_color: Tuple[int, int, int] = (50, 50, 50),
                         grid_thickness: int = 2) -> np.ndarray:
    """
    Draw color-coded grid overlay on image based on predictions.

    Args:
        image: Original image of shape (600, 800, 3) in BGR
        predictions: Array of 64 predictions (0, 1, 2, or 3)
        grid_size: Tuple of (rows, cols), default (8, 8)
        alpha: Transparency of colored fills (0.0 = transparent, 1.0 = opaque)
        grid_color: BGR color for grid lines
        grid_thickness: Thickness of grid lines in pixels

    Returns:
        Image with colored grid overlay
    """
    # Create a copy to draw on
    output = image.copy()
    overlay = image.copy()

    height, width = image.shape[:2]  # 600, 800
    rows, cols = grid_size  # 8, 8

    cell_height = height // rows  # 75
    cell_width = width // cols    # 100

    # Draw colored fills for each cell based on prediction
    pred_idx = 0
    for i in range(rows):
        for j in range(cols):
            if pred_idx >= len(predictions):
                break

            prediction = int(predictions[pred_idx])
            pred_idx += 1

            # Get color for this prediction
            color = get_prediction_color(prediction)

            # Calculate cell boundaries
            y_start = i * cell_height
            y_end = (i + 1) * cell_height
            x_start = j * cell_width
            x_end = (j + 1) * cell_width

            # Fill cell with color (only if not class 0 - no object)
            if prediction != 0:
                cv2.rectangle(overlay,
                            (x_start, y_start),
                            (x_end, y_end),
                            color,
                            -1)  # -1 = filled
            # For class 0 (no object), skip drawing any overlay

    # Blend overlay with original image
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # Draw grid lines on top
    # Horizontal lines
    for i in range(1, rows):
        y = i * cell_height
        cv2.line(output, (0, y), (width, y), grid_color, grid_thickness)

    # Vertical lines
    for j in range(1, cols):
        x = j * cell_width
        cv2.line(output, (x, 0), (x, height), grid_color, grid_thickness)

    # Draw border around entire image
    cv2.rectangle(output, (0, 0), (width - 1, height - 1), grid_color, grid_thickness)

    # Draw cell IDs (c01, c02, etc.) on each cell
    pred_idx = 0
    for i in range(rows):
        for j in range(cols):
            if pred_idx >= 64:
                break

            # Calculate cell ID
            cell_id = f"c{pred_idx + 1:02d}"

            # Position text at top-left corner of cell
            text_x = j * cell_width + 5
            text_y = i * cell_height + 20

            # Draw cell ID with black background for better visibility
            text_size = cv2.getTextSize(cell_id, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.rectangle(output,
                         (text_x - 2, text_y - text_size[1] - 2),
                         (text_x + text_size[0] + 2, text_y + 2),
                         (0, 0, 0), -1)

            cv2.putText(output, cell_id, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            pred_idx += 1

    return output


def visualize_predictions(image_path: Path,
                         predictions: np.ndarray,
                         output_path: Path,
                         grid_size: Tuple[int, int] = (8, 8),
                         alpha: float = 0.3) -> bool:
    """
    Load image, create visualization, and save to output path.

    Args:
        image_path: Path to input image
        predictions: Array of 64 predictions
        output_path: Path to save visualized image
        grid_size: Grid dimensions
        alpha: Transparency of colored fills

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            return False

        # Ensure correct size
        if image.shape != (600, 800, 3):
            image = cv2.resize(image, (800, 600), interpolation=cv2.INTER_AREA)

        # Create visualization
        viz_image = draw_prediction_grid(image, predictions, grid_size, alpha)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save visualization
        cv2.imwrite(str(output_path), viz_image)

        return True

    except Exception as e:
        print(f"Error visualizing {image_path}: {e}")
        return False


def visualize_dataset_predictions(predictions_dict: Dict,
                                   image_dir: Path,
                                   output_subdir: Path,
                                   dataset_type: str,
                                   verbose: bool = True) -> int:
    """
    Batch visualize predictions for a specific dataset (train or test).

    Args:
        predictions_dict: Dictionary mapping image names to prediction data
        image_dir: Directory containing images
        output_subdir: Subdirectory to save visualizations (e.g., output_dir/train/visualizations)
        dataset_type: 'train' or 'test'
        verbose: Print progress messages

    Returns:
        Number of successfully visualized images
    """
    # Create output directory
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Filter predictions for this dataset type
    dataset_predictions = {
        name: data for name, data in predictions_dict.items()
        if data['train_or_test'] == dataset_type
    }

    if len(dataset_predictions) == 0:
        if verbose:
            print(f"No {dataset_type} images found to visualize")
        return 0

    if verbose:
        print(f"\nVisualizing {len(dataset_predictions)} {dataset_type} images...")

    success_count = 0

    for idx, (img_name, pred_data) in enumerate(dataset_predictions.items()):
        if verbose and (idx + 1) % 10 == 0:
            print(f"  Visualized {idx + 1}/{len(dataset_predictions)} {dataset_type} images...")

        # Find image in directory
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_path = image_dir / img_name.replace(Path(img_name).suffix, ext)
            if potential_path.exists():
                image_path = potential_path
                break

        if image_path is None:
            image_path = image_dir / img_name

        if not image_path.exists():
            if verbose:
                print(f"Warning: Image not found: {img_name}")
            continue

        # Output path maintains original filename
        output_path = output_subdir / image_path.name

        # Create visualization
        success = visualize_predictions(
            image_path=image_path,
            predictions=pred_data['predictions'],
            output_path=output_path,
            alpha=0.3
        )

        if success:
            success_count += 1

    if verbose:
        print(f"Successfully visualized: {success_count}/{len(dataset_predictions)} {dataset_type} images")
        print(f"Saved to: {output_subdir}")

    return success_count


def visualize_all_predictions(predictions_dict: Dict,
                              train_image_dir: Path,
                              test_image_dir: Path,
                              output_dir: Path,
                              verbose: bool = True) -> tuple:
    """
    Batch visualize all predictions for both train and test datasets.

    Args:
        predictions_dict: Dictionary mapping image names to prediction data
                         Format: {img_name: {'filename': str,
                                            'train_or_test': str,
                                            'predictions': np.ndarray}}
        train_image_dir: Directory containing train images
        test_image_dir: Directory containing test images
        output_dir: Base output directory (creates train/visualizations and test/visualizations subdirectories)
        verbose: Print progress messages

    Returns:
        Tuple of (train_success_count, test_success_count)
    """
    output_dir = Path(output_dir)

    # Visualize train images
    train_viz_dir = output_dir / 'train' / 'visualizations'
    train_count = visualize_dataset_predictions(
        predictions_dict=predictions_dict,
        image_dir=Path(train_image_dir),
        output_subdir=train_viz_dir,
        dataset_type='train',
        verbose=verbose
    )

    # Visualize test images
    test_viz_dir = output_dir / 'test' / 'visualizations'
    test_count = visualize_dataset_predictions(
        predictions_dict=predictions_dict,
        image_dir=Path(test_image_dir),
        output_subdir=test_viz_dir,
        dataset_type='test',
        verbose=verbose
    )

    return train_count, test_count


def main():
    """
    Standalone visualization script for testing.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Visualize cricket object detection predictions')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Comma-separated predictions (64 values)')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save visualization')
    parser.add_argument('--alpha', type=float, default=0.3,
                       help='Transparency (0.0-1.0)')

    args = parser.parse_args()

    # Parse predictions
    predictions = np.array([int(x) for x in args.predictions.split(',')])

    if len(predictions) != 64:
        print(f"Error: Expected 64 predictions, got {len(predictions)}")
        return

    # Create visualization
    success = visualize_predictions(
        image_path=Path(args.image),
        predictions=predictions,
        output_path=Path(args.output),
        alpha=args.alpha
    )

    if success:
        print(f"Visualization saved to: {args.output}")
    else:
        print("Visualization failed!")


if __name__ == '__main__':
    main()
