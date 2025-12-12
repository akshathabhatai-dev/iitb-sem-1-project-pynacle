import numpy as np
import pandas as pd
import cv2
import pickle
from pathlib import Path
from typing import List, Dict
import sys
import os

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'features'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_prep'))


def predict_on_image(model, image: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Predict object classes for all grid cells in a single image.

    Args:
        model: Trained classifier
        image: RGB image of shape (600, 800, 3)
        verbose: Print progress

    Returns:
        Array of 64 predictions (0, 1, 2, or 3)
    """
    from grid_utils import divide_into_grid
    from feature_extractor import extract_features_from_cell

    # Divide into 8x8 grid
    cells = divide_into_grid(image, grid_size=(8, 8))

    # Extract features from each cell
    cell_features = []
    for idx, cell in enumerate(cells):
        features = extract_features_from_cell(cell)
        cell_features.append(features)

        if verbose and (idx + 1) % 16 == 0:
            print(f"  Extracted features for {idx + 1}/64 cells")

    X_cells = np.array(cell_features, dtype=np.float32)  # Shape: (64, 88)

    # Predict for all cells
    y_pred = model.predict(X_cells)  # Shape: (64,)
    y_pred = np.where(y_pred == 4, 0, y_pred)

    return y_pred


def predict_on_images(model, image_paths: List[Path], train_or_test: List[str],
                     verbose: bool = True) -> Dict:
    """
    Predict object classes for all grid cells in multiple images.

    Args:
        model: Trained classifier
        image_paths: List of paths to images
        train_or_test: List indicating 'train' or 'test' for each image
        verbose: Print progress

    Returns:
        Dictionary mapping image names to predictions
    """
    predictions = {}

    for idx, (img_path, dataset_type) in enumerate(zip(image_paths, train_or_test)):
        if verbose:
            print(f"Processing {idx + 1}/{len(image_paths)}: {img_path.name}")

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not load image {img_path}")
            continue

        # Validate size
        if image.shape != (600, 800, 3):
            print(f"Warning: Image {img_path.name} has shape {image.shape}, expected (600, 800, 3)")
            # Resize if needed
            image = cv2.resize(image, (800, 600), interpolation=cv2.INTER_AREA)

        # Predict
        y_pred = predict_on_image(model, image, verbose=False)

        predictions[img_path.name] = {
            'filename': img_path.name,
            'train_or_test': dataset_type,
            'predictions': y_pred
        }

    return predictions


def save_predictions_to_csv(predictions: Dict, output_path: str):
    """
    Save predictions in required CSV format.

    Format: ImageFileName, TrainOrTest, c01, c02, ..., c64

    Args:
        predictions: Dictionary mapping image names to prediction data
        output_path: Path to save CSV
    """
    rows = []

    for img_name, pred_data in predictions.items():
        row = {
            'ImageFileName': pred_data['filename'],
            'TrainOrTest': pred_data['train_or_test']
        }

        # Add 64 cell predictions (c01 to c64)
        for i, cell_pred in enumerate(pred_data['predictions']):
            col_name = f"c{i+1:02d}"  # c01, c02, ..., c64
            row[col_name] = int(cell_pred)

        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Column order: ImageFileName, TrainOrTest, c01, ..., c64
    columns = ['ImageFileName', 'TrainOrTest'] + [f"c{i:02d}" for i in range(1, 65)]
    df = df[columns]

    # Sort by filename for consistency
    df = df.sort_values('ImageFileName').reset_index(drop=True)

    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nPredictions saved to {output_path}")
    print(f"Total images: {len(df)}")


def visualize_predictions(predictions: Dict,
                         train_image_dir: str,
                         test_image_dir: str,
                         validate_image_dir: str,
                         output_dir: str,
                         verbose: bool = True):
    """
    Create color-coded grid visualizations for train and test images.

    Args:
        predictions: Dictionary mapping image names to prediction data
        train_image_dir: Directory containing train images
        test_image_dir: Directory containing test images
        output_dir: Base directory to save visualizations (creates train/visualizations and test/visualizations subdirectories)
        verbose: Print progress messages
    """
    try:
        from visualize import visualize_all_predictions
    except ImportError:
        # Try relative import
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from visualize import visualize_all_predictions

    train_image_dir = Path(train_image_dir)
    test_image_dir = Path(test_image_dir)
    validate_image_dir = Path(validate_image_dir)
    output_dir = Path(output_dir)

    # Create visualizations for both train and test
    train_count, test_count, validate_count = visualize_all_predictions(
        predictions_dict=predictions,
        train_image_dir=train_image_dir,
        test_image_dir=test_image_dir,
        validate_image_dir=validate_image_dir,
        output_dir=output_dir,
        verbose=verbose
    )

    return train_count, test_count, validate_count


def load_model(model_path: str):
    """
    Load trained model from disk.

    Args:
        model_path: Path to saved model

    Returns:
        Loaded model
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print(f"Model loaded from {model_path}")
    return model


def main():
    """Main prediction pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate predictions for cricket images')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pkl)')
    parser.add_argument('--train_dir', type=str, required=True,
                       help='Directory containing training images')
    parser.add_argument('--test_dir', type=str, required=True,
                       help='Directory containing test images')
    parser.add_argument('--output', type=str, default='outputs/predictions.csv',
                       help='Path to save predictions CSV')

    args = parser.parse_args()

    # Load model
    print("="*60)
    print("CRICKET OBJECT DETECTION - PREDICTION PIPELINE")
    print("="*60)
    model = load_model(args.model)

    # Get image paths
    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)

    train_images = sorted(list(train_dir.glob('*.png')) +
                         list(train_dir.glob('*.jpg')) +
                         list(train_dir.glob('*.jpeg')))

    test_images = sorted(list(test_dir.glob('*.png')) +
                        list(test_dir.glob('*.jpg')) +
                        list(test_dir.glob('*.jpeg')))

    print(f"\nFound {len(train_images)} training images")
    print(f"Found {len(test_images)} test images")

    # Combine image paths and labels
    image_paths = train_images + test_images
    train_or_test = ['train'] * len(train_images) + ['test'] * len(test_images)

    # Generate predictions
    print("\nGenerating predictions...")
    predictions = predict_on_images(model, image_paths, train_or_test, verbose=True)

    # Save to CSV
    save_predictions_to_csv(predictions, args.output)

    print("\n" + "="*60)
    print("Prediction pipeline complete!")
    print("="*60)


if __name__ == '__main__':
    main()
