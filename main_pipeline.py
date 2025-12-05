"""
Cricket Object Detection - Main Pipeline
This script orchestrates the complete workflow from feature extraction to prediction.
"""

import numpy as np
import pandas as pd
import cv2
import pickle
from pathlib import Path
from tqdm import tqdm
import sys
import os

# Add src directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'features'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'data_prep'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'models'))

from grid_utils import divide_into_grid, validate_image_size, resize_image
from feature_extractor import extract_features_from_cell
from train import train_random_forest, save_model
from train_compare_models import compare_models, select_best_model, save_comparison_results
from evaluate import evaluate_model, plot_confusion_matrix, save_metrics_to_file
from predict import predict_on_images, save_predictions_to_csv, visualize_predictions


def load_images_from_directory(image_dir: str) -> list:
    """Load all images from a directory."""
    image_dir = Path(image_dir)
    image_paths = sorted(list(image_dir.glob('*.png')) +
                        list(image_dir.glob('*.jpg')) +
                        list(image_dir.glob('*.jpeg')))

    images = []
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is not None:
            # Validate or resize to 800x600
            if img.shape != (600, 800, 3):
                img = resize_image(img, (800, 600))
            images.append((img_path.name, img))

    return images


def load_labels_from_csv(csv_path: str) -> pd.DataFrame:
    """Load labels from CSV file."""
    return pd.read_csv(csv_path)


def extract_features_and_labels(images: list, labels_df: pd.DataFrame, verbose: bool = True):
    """
    Extract features and labels from images.

    Args:
        images: List of (image_name, image_array) tuples
        labels_df: DataFrame with labels
        verbose: Print progress

    Returns:
        X: Features array of shape (n_samples, 88)
        y: Labels array of shape (n_samples,)
        image_names: List of image names for tracking
    """
    X_list = []
    y_list = []
    image_names_list = []
    cell_index_list = []

    # Create a mapping of image names to labels
    labels_dict = {}
    for _, row in labels_df.iterrows():
        img_name = row['ImageFileName']
        cell_labels = [int(row[f'c{i:02d}']) for i in range(1, 65)]
        labels_dict[img_name] = cell_labels

    iterator = tqdm(images, desc="Extracting features") if verbose else images

    for img_name, img in iterator:
        # Check if labels exist for this image
        if img_name not in labels_dict:
            if verbose:
                print(f"Warning: No labels found for {img_name}, skipping...")
            continue

        # Divide into grid
        cells = divide_into_grid(img, grid_size=(8, 8))

        # Extract features from each cell
        for cell_idx, cell in enumerate(cells):
            features = extract_features_from_cell(cell)
            label = labels_dict[img_name][cell_idx]

            X_list.append(features)
            y_list.append(label)
            image_names_list.append(img_name)
            cell_index_list.append(cell_idx)
    # Create DataFrame for easier analysis

    feature_columns = (
        [f"rgb_{i}" for i in range(14)] +
        [f"hsv_{i}" for i in range(14)] +
        [f"lbp_{i}" for i in range(10)] +
        [f"hog_{i}" for i in range(20)] +
        [f"gabor_{i}" for i in range(20)] +
        [f"glcm_{i}" for i in range(10)]
    )

    columns = ["image_name", "cell_index", 'label'] + feature_columns
    df = pd.DataFrame(X_list, columns=feature_columns)
    df['label'] = y_list
    df['image_name'] = image_names_list
    df['cell_index'] = cell_index_list
    df = df[columns]
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    return X, y, image_names_list, df


def main():
    """Main pipeline execution."""
    import argparse

    parser = argparse.ArgumentParser(description='Cricket Object Detection Pipeline')
    parser.add_argument('--mode', type=str, choices=['extract', 'train', 'evaluate', 'predict', 'full'],
                       required=True, help='Pipeline mode')
    parser.add_argument('--train_images', type=str, default='dataset/train/images',
                       help='Directory with training images')
    parser.add_argument('--test_images', type=str, default='dataset/test/images',
                       help='Directory with test images')
    parser.add_argument('--train_labels', type=str, default='dataset/train/labels/labels.csv',
                       help='Path to training labels CSV')
    parser.add_argument('--test_labels', type=str, default='dataset/test/labels/labels.csv',
                       help='Path to test labels CSV')
    parser.add_argument('--model_path', type=str, default='models/cricket_model.pkl',
                       help='Path to save/load model')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Directory for outputs')
    parser.add_argument('--tune', action='store_true',
                       help='Perform hyperparameter tuning (slower)')
    parser.add_argument('--compare_models', action='store_true',
                       help='Compare all 5 models and select best (slower)')
    parser.add_argument('--model_type', type=str,
                       choices=['logistic', 'rf', 'svm', 'knn', 'xgboost', 'auto'],
                       default='auto',
                       help='Model type to use (auto=best from comparison, rf=Random Forest only)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate color-coded grid visualizations for train and test images')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("CRICKET OBJECT DETECTION PIPELINE")
    print("="*70)

    # FULL PIPELINE or EXTRACT MODE
    if args.mode in ['full', 'extract']:
        print("\n[1/4] FEATURE EXTRACTION")
        print("-"*70)

        # Load training images and labels
        print("Loading training data...")
        train_images = load_images_from_directory(args.train_images)
        train_labels_df = load_labels_from_csv(args.train_labels)

        print(f"Found {len(train_images)} training images")

        # Extract features
        X_train, y_train, _, train_df = extract_features_and_labels(train_images, train_labels_df)

        print(f"Extracted features: X_train shape = {X_train.shape}")
        print(f"Labels: y_train shape = {y_train.shape}")

        # Save features
        train_df.to_csv(output_dir / 'train_features.csv', index=False)
        np.save(output_dir / 'X_train.npy', X_train)
        np.save(output_dir / 'y_train.npy', y_train)
        print(f"Saved training features to {output_dir}")

        # Load test data if available
        if Path(args.test_images).exists() and Path(args.test_labels).exists():
            print("\nLoading test data...")
            test_images = load_images_from_directory(args.test_images)
            test_labels_df = load_labels_from_csv(args.test_labels)

            print(f"Found {len(test_images)} test images")

            X_test, y_test, _, test_df = extract_features_and_labels(test_images, test_labels_df)

            print(f"Extracted features: X_test shape = {X_test.shape}")

            np.save(output_dir / 'X_test.npy', X_test)
            np.save(output_dir / 'y_test.npy', y_test)
            test_df.to_csv(output_dir / 'test_features.csv', index=False)
            print(f"Saved test features to {output_dir}")

    # Load features if not in extract mode
    if args.mode in ['train', 'evaluate', 'full']:
        if args.mode != 'full':
            print("\nLoading features...")
            X_train = np.load(output_dir / 'X_train.npy')
            y_train = np.load(output_dir / 'y_train.npy')
            print(f"Loaded X_train: {X_train.shape}, y_train: {y_train.shape}")

    # TRAIN MODE
    if args.mode in ['full', 'train']:
        print("\n[2/4] MODEL TRAINING")
        print("-"*70)

        if args.compare_models:
            # Compare all 5 models and select best
            print("Model comparison mode enabled - training all 5 models...")
            comparison_results = compare_models(
                X_train, y_train,
                tune_hyperparameters=args.tune,
                cv_folds=5,
                verbose=True
            )

            # Select best model
            best_model_key, model, justification = select_best_model(comparison_results)

            print(justification)

            # Save comparison results
            save_comparison_results(
                comparison_results,
                output_dir / 'model_comparison.csv',
                justification
            )

            # Save best model
            save_model(model, args.model_path)

        elif args.model_type == 'auto' or args.model_type == 'rf':
            # Train only Random Forest (default)
            model = train_random_forest(
                X_train, y_train,
                tune_hyperparameters=args.tune,
                verbose=True
            )
            save_model(model, args.model_path)

        else:
            # Train specific model type
            from train_compare_models import (
                train_logistic_regression, train_svm,
                train_knn, train_xgboost
            )

            model_trainers = {
                'logistic': train_logistic_regression,
                'svm': train_svm,
                'knn': train_knn,
                'xgboost': train_xgboost
            }

            train_func = model_trainers[args.model_type]
            model, metrics = train_func(X_train, y_train, args.tune, 5, True)
            save_model(model, args.model_path)

    # Load model if needed
    if args.mode in ['evaluate', 'predict']:
        print("\nLoading model...")
        with open(args.model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {args.model_path}")

    # EVALUATE MODE
    if args.mode in ['full', 'evaluate']:
        print("\n[3/4] MODEL EVALUATION")
        print("-"*70)

        # Load test data
        if args.mode != 'full':
            X_test = np.load(output_dir / 'X_test.npy')
            y_test = np.load(output_dir / 'y_test.npy')

        results = evaluate_model(model, X_test, y_test, verbose=True)

        # Save results
        save_metrics_to_file(results, output_dir / 'metrics.txt')
        plot_confusion_matrix(results['confusion_matrix'],
                            save_path=output_dir / 'confusion_matrix.png')

    # PREDICT MODE
    if args.mode in ['full', 'predict']:
        print("\n[4/4] GENERATING PREDICTIONS")
        print("-"*70)

        # Get all image paths
        train_image_paths = sorted(list(Path(args.train_images).glob('*.png')) +
                                   list(Path(args.train_images).glob('*.jpg')))
        test_image_paths = sorted(list(Path(args.test_images).glob('*.png')) +
                                  list(Path(args.test_images).glob('*.jpg')))

        image_paths = train_image_paths + test_image_paths
        train_or_test = ['train'] * len(train_image_paths) + ['test'] * len(test_image_paths)

        predictions = predict_on_images(model, image_paths, train_or_test, verbose=True)

        save_predictions_to_csv(predictions, output_dir / 'predictions.csv')

        # Generate visualizations if requested
        if args.visualize:
            print("\n[VISUALIZATION] GENERATING COLOR-CODED GRIDS")
            print("-"*70)

            train_count, test_count = visualize_predictions(
                predictions=predictions,
                train_image_dir=args.train_images,
                test_image_dir=args.test_images,
                output_dir=args.output_dir,
                verbose=True
            )

            if train_count > 0 or test_count > 0:
                print(f"\nVisualization complete!")
                if train_count > 0:
                    print(f"  Train: {train_count} images -> {output_dir}/train/visualizations/")
                if test_count > 0:
                    print(f"  Test: {test_count} images -> {output_dir}/test/visualizations/")

    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
