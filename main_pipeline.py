"""
Cricket Object Detection - Main Pipeline
This script orchestrates the complete workflow from feature extraction to prediction.
"""

from collections import Counter
import re
from turtle import st
import numpy as np
import pandas as pd
import cv2
import pickle
from pathlib import Path
from tqdm import tqdm
import sys
import os

from src.models.train import train_xgboost

# Add src directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'features'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'data_prep'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'models'))

from grid_utils import divide_into_grid, validate_image_size, resize_image
from feature_extractor import extract_features_from_cell, get_all_feature_names, compute_hog_per_cell, extract_na_aware_hog_for_grid_features
# from neighborhood_hog_features import compute_hog_per_cell, extract_na_aware_hog_for_grid_features
from train import train_random_forest, save_model
from train_compare_models import compare_models, select_best_model, save_comparison_results
from evaluate import evaluate_model, plot_confusion_matrix, save_metrics_to_file
from predict import predict_on_images, save_predictions_to_csv, visualize_predictions
from undersample import apply_undersampling, get_undersampling_report, oversample_class_1, apply_oversampling
from augmentation import filter_cells_for_augmentation, generate_augmented_cells, generate_random_image_name

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

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


def extract_features_and_labels(images: list, labels_df: pd.DataFrame, verbose: bool = True, test_data: bool = False, train_data: bool = False):
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
    if labels_df is None or labels_df.empty:
        # raise ValueError("Labels DataFrame is empty or None.")
        if verbose:
            print("Warning: Labels DataFrame is empty or None. Proceeding without labels.")
    else:
        for _, row in labels_df.iterrows():
            img_name = row['ImageFileName']
            cell_labels = [int(row[f'c{i:02d}']) for i in range(1, 65)]
            labels_dict[img_name] = cell_labels
    iterator = tqdm(images, desc="Extracting features") if verbose else images
    all_bat_image_cells = []
    all_ball_image_cells = []
    all_stump_image_cells = []
    all_ball_image_cell_ds = []
    for img_name, img in iterator:
        # Check if labels exist for this image
        if img_name not in labels_dict:
            if verbose and not test_data:
                print(f"Warning: No labels found for {img_name}, skipping...")
            continue

        # Divide into grid
        # cell_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # gray_cells = divide_into_grid(cell_gray, grid_size=(8, 8))
        cells = divide_into_grid(img, grid_size=(8, 8))
        if train_data:
            ball_cells, ball_idx, _ = filter_cells_for_augmentation(cells, labels_dict[img_name], target_class=1)
            bat_cells, bat_idx, _ = filter_cells_for_augmentation(cells, labels_dict[img_name], target_class=2)
            stump_cells, stump_idx, _ = filter_cells_for_augmentation(cells, labels_dict[img_name], target_class=3)
            all_ball_image_cells.extend(ball_cells)
            all_ball_image_cell_ds.extend(ball_idx)
            all_bat_image_cells.extend(bat_cells)
            all_stump_image_cells.extend(stump_cells)
            # all_bat_image_cell_ds.extend(bat_idx)

        # gray_cells = [cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) for cell in cells]

        # hog_list = compute_hog_per_cell(gray_cells, orientations=6,
        #                                         pixels_per_cell=(16, 16),
        #                                         cells_per_block=(1, 1))
        # hog_na_list = extract_na_aware_hog_for_grid_features(hog_list)


        # Extract features from each cell
        for cell_idx, cell in enumerate(cells):
            # features = extract_features_from_cell(cell, na_hog_vector=hog_na_list[cell_idx]) # to enable NA-HOG
            features = extract_features_from_cell(cell)  # without NA-HOG
            label = labels_dict[img_name][cell_idx]

            X_list.append(features)
            y_list.append(label)
            image_names_list.append(img_name)
            cell_index_list.append(cell_idx)
    # Create DataFrame for easier analysis

    # feature_columns = (
    #     [f"rgb_{i}" for i in range(14)] +
    #     [f"hsv_{i}" for i in range(14)] +
    #     [f"lbp_{i}" for i in range(10)] +
    #     [f"hog_{i}" for i in range(576)] +
    #     [f"gabor_{i}" for i in range(20)] +
    #     [f"glcm_{i}" for i in range(10)]
    # )
    
    # num_features = len(X_list[0])   # auto-detect feature count

    # feature_columns = [f"f_{i}" for i in range(num_features)]   # auto column names
    if train_data:
        def get_groupwise_max(labels, classes=[1,2,3]):
            counter = Counter(labels)
            vals = [counter.get(cls, 0) for cls in classes]
            return max(vals)
        max_count = get_groupwise_max(y_list, classes=[0,1,2,3])
        current_ball_count = get_groupwise_max(y_list, classes=[1])
        current_bat_count = get_groupwise_max(y_list, classes=[2])
        current_stump_count = get_groupwise_max(y_list, classes=[3])
        no_of_augumentations_ball = 8
        no_of_augumentations_bat = 4
        no_of_augumentations_stump = 3
        estimated_ball_count = current_ball_count * no_of_augumentations_ball
        estimated_bat_count = current_bat_count * no_of_augumentations_bat
        estimated_stump_count = current_stump_count * no_of_augumentations_stump
        min_count = min(estimated_ball_count, estimated_bat_count, estimated_stump_count)
        remaining_balls = int(min_count - current_ball_count)
        remaining_bats = int(min_count - current_bat_count)
        remaining_stumps = int(min_count - current_stump_count)
        # no_of_augumentations_ball = int(np.ceil(remaining_balls/len(all_ball_image_cells)))
        # no_of_augumentations_bat = int(np.ceil(remaining_bats/len(all_bat_image_cells)))
        # no_of_augumentations_stump = int(np.ceil(remaining_stumps/len(all_stump_image_cells)))
        print(f"Current counts - Ball: {current_ball_count}, Bat: {current_bat_count}, Stump: {current_stump_count}, Max: {0.5*max_count}")
        print(f"Remaining counts - Ball: {remaining_balls}, Bat: {remaining_bats}, Stump: {remaining_stumps}, Max: {0.5*max_count}")
        print(f"Number Of Augmentations - Ball: {no_of_augumentations_ball}, Bat: {no_of_augumentations_bat}, Stump: {no_of_augumentations_stump}")
        print("Applying augmented oversampling for Class 1 (ball)...")
        augmented_ball_cells = []

        # ball_cells, ball_idx, _ = filter_cells_for_augmentation(cells, labels, target_class=1)

        iterator = tqdm(all_ball_image_cells, desc="Suggesting augmentations") if verbose else all_ball_image_cells
        for cell in iterator:
            aug_versions = generate_augmented_cells(cell, num_augmented=no_of_augumentations_ball-1, class_no=1)
            augmented_ball_cells.extend(aug_versions)
        
        print(f"Total ball cells available for augmentation: {len(augmented_ball_cells)}, need to augment: {remaining_balls}")
        iterator = tqdm(augmented_ball_cells[0:int(remaining_balls)], desc="Extracting Features for augmentations") if verbose else augmented_ball_cells[0:int(remaining_balls)]
        # for cell_idx, cell in enumerate(augmented_ball_cells):
        for cell_idx, cell in enumerate(iterator):
            features = extract_features_from_cell(cell)  # without NA-HOG
            X_list.append(features)
            y_list.append(1)  # label is 1 for all augmented ball cells
            image_names_list.append(generate_random_image_name())
            cell_index_list.append(-1)  # -1 indicates augmented cell

        print("Applying augmented oversampling for Class 2 (bat)...")
        augmented_bat_cells = []
        iterator = tqdm(all_bat_image_cells, desc="Suggesting augmentations") if verbose else all_bat_image_cells
        for cell in iterator:
            aug_versions = generate_augmented_cells(cell, num_augmented=no_of_augumentations_bat-1, class_no=2)
            augmented_bat_cells.extend(aug_versions)
        
        print(f"Total bat cells available for augmentation: {len(augmented_bat_cells)}, need to augment: {remaining_bats}")
        iterator = tqdm(augmented_bat_cells[0: int(remaining_bats)], desc="Extracting Features for augmentations") if verbose else augmented_bat_cells[0:int(remaining_bats)]
        # for cell_idx, cell in enumerate(augmented_ball_cells):
        for cell_idx, cell in enumerate(iterator):
            features = extract_features_from_cell(cell)  # without NA-HOG
            X_list.append(features)
            y_list.append(2)  # label is 2 for all augmented bat cells
            image_names_list.append(generate_random_image_name())
            cell_index_list.append(-1)  # -1 indicates augmented cell
        
        print("Applying augmented oversampling for Class 3 (stump)...")
        augmented_stump_cells = []
        # print(f"Total stump cells available for augmentation: {len(all_stump_image_cells)}, need to augment: {remaining_stumps}")
        iterator = tqdm(all_stump_image_cells, desc="Suggesting augmentations") if verbose else all_stump_image_cells[0:remaining_stumps]
        for cell in iterator:
            aug_versions = generate_augmented_cells(cell, num_augmented=no_of_augumentations_stump-1, class_no=3)
            augmented_stump_cells.extend(aug_versions)
        
        print(f"Total stump cells available for augmentation: {len(augmented_stump_cells)}, need to augment: {remaining_stumps}")
        iterator = tqdm(augmented_stump_cells[0:int(remaining_stumps)], desc="Extracting Features for augmentations") if verbose else augmented_stump_cells[0:int(remaining_stumps)]
        # for cell_idx, cell in enumerate(augmented_ball_cells):
        for cell_idx, cell in enumerate(iterator):
            features = extract_features_from_cell(cell)  # without NA-HOG
            X_list.append(features)
            y_list.append(3)  # label is 3 for all augmented stump cells
            image_names_list.append(generate_random_image_name())
            cell_index_list.append(-1)  # -1 indicates augmented cell
    # label = labels_dict[img_name][cell_idx]

    # X_list.append(features)
    # y_list.append(label)
    # image_names_list.append(img_name)
    # cell_index_list.append(cell_idx)

    feature_columns = get_all_feature_names()

    columns = ["image_name", "cell_index", 'label'] + feature_columns
    df = pd.DataFrame(X_list, columns=feature_columns)
    df['label'] = y_list
    df['image_name'] = image_names_list
    df['cell_index'] = cell_index_list
    df = df[columns]
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    return X, y, image_names_list, df

def select_balanced_data(X_train, y_train, under_sample_size=1500, over_sample_size=1500, seed=42):
    """
    Returns:
        X_final: X with all class 1,2,3 + sample_size random class 0
        y_final: corresponding y values
    """

    # Set the seed for reproducibility
    rng = np.random.default_rng(seed)

    # Get indices for classes
    idx_0 = np.where(y_train == 0)[0]
    idx_1 = np.where(y_train == 1)[0]
    idx_23 = np.where(np.isin(y_train, [2, 3]))[0]

    # Randomly sample class-0 indices
    idx_0_sample = rng.choice(idx_0, size=under_sample_size, replace=False)
    idx_1_sample = rng.choice(idx_1, size=over_sample_size, replace=True)

    # Combine indices
    final_idx = np.concatenate([idx_0_sample, idx_1_sample, idx_23])

    # Shuffle final indices with the same seed
    rng.shuffle(final_idx)

    # Select final data
    X_final = X_train[final_idx]
    y_final = y_train[final_idx]

    return X_final, y_final


def main():
    """Main pipeline execution."""
    import argparse

    parser = argparse.ArgumentParser(description='Cricket Object Detection Pipeline')
    parser.add_argument('--mode', type=str, choices=['extract', 'train', 'evaluate', 'predict', 'full'],
                       required=True, help='Pipeline mode')
    parser.add_argument('--model', type=str, choices=['xgboost', 'random_forest'], 
                       help='Model type to use')
    parser.add_argument('--train_images', type=str, default='dataset/train/images',
                       help='Directory with training images')
    parser.add_argument('--test_images', type=str, default='dataset/test/images',
                       help='Directory with test images')
    parser.add_argument('--validate_images', type=str, default='dataset/validate/images',
                       help='Directory with validation images')
    parser.add_argument('--train_labels', type=str, default='dataset/train/labels/labels.csv',
                       help='Path to training labels CSV')
    parser.add_argument('--test_labels', type=str, default='dataset/test/labels/labels.csv',
                       help='Path to test labels CSV')
    parser.add_argument('--validate_labels', type=str, default='dataset/validate/labels/labels.csv',
                       help='Path to validation labels CSV')
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
    parser.add_argument('--undersample', action='store_true', default=True,
                       help='Apply undersampling to balance class distribution (default: True, use --no-undersample to disable)')
    parser.add_argument('--no-undersample', dest='undersample', action='store_false',
                       help='Disable undersampling (keep original imbalanced distribution)')
    parser.add_argument('--undersample_strategy', type=str,
                       choices=['image_stratified', 'random'],
                       default='image_stratified',
                       help='Undersampling strategy: image_stratified (recommended) or random (default: image_stratified)')

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
        X_train, y_train, _, train_df = extract_features_and_labels(train_images, train_labels_df, train_data=True)

        print(f"Extracted features: X_train shape = {X_train.shape}")
        print(f"Labels: y_train shape = {y_train.shape}")

        # Save features
        train_df.to_csv(output_dir / 'train_features.csv', index=False)

        # Apply undersampling if requested
        if args.undersample:
            print("\n" + "-"*70)
            print("UNDERSAMPLING CLASS 0 (NO OBJECT)")
            print("-"*70)

            # Save original y_train for report generation
            y_train_original = y_train.copy()

            X_train, y_train, train_df = apply_undersampling(
                X_train, y_train, train_df,
                strategy=args.undersample_strategy,
                random_seed=42,
                verbose=True
            )

            print(f"\nAfter undersampling: X_train shape = {X_train.shape}")
            print(f"After undersampling: y_train shape = {y_train.shape}")

            # Generate and save undersampling report
            report = get_undersampling_report(y_train_original, y_train)
            report.to_csv(output_dir / 'undersampling_report.csv', index=False)
            print(f"\nUndersampling report saved to {output_dir / 'undersampling_report.csv'}")
            print("-"*70)

            # X_train, y_train, train_df = oversample_class_1(
            #     X_train, y_train, train_df,
            #     target_count_ratio=0.95,
            #     random_seed=42,
            #     verbose=True
            # )

            # X_train, y_train, _ = apply_oversampling(
            #     X_train, y_train, train_df,
            #     target_count_ratio=0.95,
            #     random_seed=42,
            #     verbose=True
            # )

            # print(f"\nAfter oversampling: X_train shape = {X_train.shape}")
            # print(f"After oversampling: y_train shape = {y_train.shape}")

            # # Generate and save oversampling report
            # report = get_undersampling_report(y_train_original, y_train)
            # report.to_csv(output_dir / 'oversampling_report.csv', index=False)
            # print(f"\nOversampling report saved to {output_dir / 'oversampling_report.csv'}")
            # print("-"*70)

        np.save(output_dir / 'X_train.npy', X_train)
        np.save(output_dir / 'y_train.npy', y_train)
        print(f"Saved training features to {output_dir}")

        # Load test data if available
        if Path(args.test_images).exists():
            print("\nLoading test data...")
            test_images = load_images_from_directory(args.test_images)
            test_labels_df = load_labels_from_csv(args.test_labels) if Path(args.test_labels).exists() else None

            print(f"Found {len(test_images)} test images")

            X_test, y_test, _, test_df = extract_features_and_labels(test_images, test_labels_df, test_data=True)

            print(f"Extracted features: X_test shape = {X_test.shape}")

            np.save(output_dir / 'X_test.npy', X_test)
            if y_test is not None: 
                np.save(output_dir / 'y_test.npy', y_test)
            test_df.to_csv(output_dir / 'test_features.csv', index=False)
            print(f"Saved test features to {output_dir}")

        # Load test data if available
        if Path(args.validate_images).exists() and Path(args.validate_labels).exists():
            print("\nLoading validation data...")
            validate_images = load_images_from_directory(args.validate_images)
            validate_labels_df = load_labels_from_csv(args.validate_labels)

            print(f"Found {len(validate_images)} validation images")

            X_validate, y_validate, _, validate_df = extract_features_and_labels(validate_images, validate_labels_df)
            print(f"Extracted features: X_validate shape = {X_validate.shape}")

            np.save(output_dir / 'X_validate.npy', X_validate)
            np.save(output_dir / 'y_validate.npy', y_validate)
            validate_df.to_csv(output_dir / 'validate_features.csv', index=False)
            print(f"Saved validation features to {output_dir}")

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

        # Load validation data
        if args.mode != 'full':
            X_validate = np.load(output_dir / 'X_validate.npy')
            y_validate = np.load(output_dir / 'y_validate.npy')
            print(f"Loaded X_validate: {X_validate.shape}, y_validate: {y_validate.shape}")

        results = evaluate_model(model, X_validate, y_validate, verbose=True)

        # Save results
        save_metrics_to_file(results, output_dir / 'metrics.txt')
        plot_confusion_matrix(results['confusion_matrix'],
                            save_path=output_dir / 'confusion_matrix.png')

    # PREDICT MODE
    if False and args.mode in ['full', 'predict']:
        print("\n[4/4] GENERATING PREDICTIONS")
        print("-"*70)

        # Get all image paths
        train_image_paths = sorted(list(Path(args.train_images).glob('*.png')) +
                                   list(Path(args.train_images).glob('*.jpg')))
        test_image_paths = sorted(list(Path(args.test_images).glob('*.png')) +
                                  list(Path(args.test_images).glob('*.jpg')))
        validate_image_paths = sorted(list(Path(args.validate_images).glob('*.png')) +
                                  list(Path(args.validate_images).glob('*.jpg')))

        image_paths = train_image_paths + test_image_paths + validate_image_paths
        train_or_test = ['train'] * len(train_image_paths) + ['test'] * len(test_image_paths) + ['validate'] * len(validate_image_paths)
        predictions = predict_on_images(model, image_paths, train_or_test, verbose=True)

        save_predictions_to_csv(predictions, output_dir / 'predictions.csv')

        # Generate visualizations if requested
        if args.visualize:
            print("\n[VISUALIZATION] GENERATING COLOR-CODED GRIDS")
            print("-"*70)

            train_count, test_count, validate_count = visualize_predictions(
                predictions=predictions,
                train_image_dir=args.train_images,
                test_image_dir=args.test_images,
                validate_image_dir=args.validate_images,
                output_dir=args.output_dir,
                verbose=True
            )

            if train_count > 0 or test_count > 0:
                print(f"\nVisualization complete!")
                if train_count > 0:
                    print(f"  Train: {train_count} images -> {output_dir}/train/visualizations/")
                if test_count > 0:
                    print(f"  Test: {test_count} images -> {output_dir}/test/visualizations/")
                if validate_count > 0:
                    print(f"  Validate: {validate_count} images -> {output_dir}/validate/visualizations/")
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
