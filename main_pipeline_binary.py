"""
Binary Model Pipeline - Cricket Object Detection

Implements three binary classifiers approach:
- Ball vs Rest (Not-Ball)
- Bat vs Rest (Not-Bat)
- Stump vs Rest (Not-Stump)

Pipeline Modes:
- extract: Feature extraction (train, validation, test sets)
- train: Train 3 binary models on training set
- evaluate: Evaluate binary models on validation set
- predict: Generate predictions on test set with conflict resolution
- full: Complete pipeline (extract → train → evaluate → predict)

Usage:
    # Full pipeline with Random Forest
    python main_pipeline_binary.py --mode full --algorithm random_forest --tune

    # Full pipeline with both RF and XGBoost
    python main_pipeline_binary.py --mode full --algorithm both --tune

    # Train only
    python main_pipeline_binary.py --mode train --algorithm xgboost --tune
"""

import numpy as np
import pandas as pd
import cv2
import pickle
from pathlib import Path
from tqdm import tqdm
import sys
import os
import argparse

# Add src directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'features'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'data_prep'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'models'))

from grid_utils import divide_into_grid, validate_image_size, resize_image
from feature_extractor import extract_features_from_cell, get_all_feature_names
from binary_train import train_three_binary_models, save_binary_models, load_binary_models
from binary_predict import predict_with_binary_models
from binary_evaluate import (
    evaluate_binary_models_independently,
    evaluate_final_predictions,
    analyze_conflict_rate
)
from threshold_tuning import optimize_thresholds_all_models, evaluate_threshold_impact, plot_threshold_comparison
from visualize import visualize_predictions as viz_single_image
from undersample import apply_undersampling, get_undersampling_report

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


def convert_multiclass_to_binary(y_multiclass):
    """
    Convert multi-class labels (0,1,2,3) to three binary label arrays.

    Args:
        y_multiclass: Array of labels where 0=No Object, 1=Ball, 2=Bat, 3=Stump

    Returns:
        y_ball: Binary array (1 if ball present, 0 otherwise)
        y_bat: Binary array (1 if bat present, 0 otherwise)
        y_stump: Binary array (1 if stump present, 0 otherwise)
    """
    y_ball = (y_multiclass == 1).astype(int)
    y_bat = (y_multiclass == 2).astype(int)
    y_stump = (y_multiclass == 3).astype(int)
    return y_ball, y_bat, y_stump


def extract_features_and_labels_binary(images: list, labels_df: pd.DataFrame,
                                       verbose: bool = True, is_test: bool = False):
    """
    Extract features and convert labels to binary format.

    Supports TWO label formats:
    1. Single-label (64 columns): c01, c02, ..., c64 (values: 0,1,2,3)
    2. Multi-label (196 columns): c01_ball, c01_bat, c01_stump, ..., c64_stump (values: 0/1)

    Args:
        images: List of (image_name, image_array) tuples
        labels_df: DataFrame with labels (auto-detects format)
        verbose: Print progress
        is_test: If True, skip warnings about missing labels

    Returns:
        X: Features array (N, 2343)
        y_ball: Binary ball labels (N,)
        y_bat: Binary bat labels (N,)
        y_stump: Binary stump labels (N,)
        y_multiclass: Multi-class labels (N,) - for evaluation
        image_names_list: List of image names
        df: DataFrame with columns [image_name, cell_index, label] - for undersampling
    """
    X_list = []
    y_ball_list = []
    y_bat_list = []
    y_stump_list = []
    image_names_list = []
    cell_index_list = []

    # Auto-detect label format
    if labels_df is None or labels_df.empty:
        if verbose and not is_test:
            print("Warning: Labels DataFrame is empty or None. Proceeding without labels.")
        label_format = None
    else:
        # Check if multi-label format (196 columns: c01_ball, c01_bat, c01_stump, ...)
        if 'c01_ball' in labels_df.columns:
            label_format = 'multilabel'
            if verbose:
                print("Detected multi-label format (196 columns)")
        # Check if single-label format (64 columns: c01, c02, ...)
        elif 'c01' in labels_df.columns:
            label_format = 'singlelabel'
            if verbose:
                print("Detected single-label format (64 columns) - will convert to binary")
        else:
            raise ValueError("Unknown label format! Expected either 'c01' or 'c01_ball' columns")

    # Create a mapping of image names to labels
    labels_dict = {}
    if label_format == 'multilabel':
        # Multi-label format: c01_ball, c01_bat, c01_stump, ..., c64_ball, c64_bat, c64_stump
        for _, row in labels_df.iterrows():
            img_name = row['ImageFileName']
            cell_labels = []
            for cell_idx in range(1, 65):
                ball = int(row[f'c{cell_idx:02d}_ball'])
                bat = int(row[f'c{cell_idx:02d}_bat'])
                stump = int(row[f'c{cell_idx:02d}_stump'])
                cell_labels.append({'ball': ball, 'bat': bat, 'stump': stump})
            labels_dict[img_name] = cell_labels

    elif label_format == 'singlelabel':
        # Single-label format: c01, c02, ..., c64 (values: 0,1,2,3)
        for _, row in labels_df.iterrows():
            img_name = row['ImageFileName']
            cell_labels = []
            for cell_idx in range(1, 65):
                label = int(row[f'c{cell_idx:02d}'])
                # Convert to binary: 0→(0,0,0), 1→(1,0,0), 2→(0,1,0), 3→(0,0,1)
                ball = 1 if label == 1 else 0
                bat = 1 if label == 2 else 0
                stump = 1 if label == 3 else 0
                cell_labels.append({'ball': ball, 'bat': bat, 'stump': stump})
            labels_dict[img_name] = cell_labels

    iterator = tqdm(images, desc="Extracting features") if verbose else images

    for img_name, img in iterator:
        # Check if labels exist for this image
        if img_name not in labels_dict:
            if verbose and not is_test:
                print(f"Warning: No labels found for {img_name}, skipping...")
            continue

        # Divide into grid
        cells = divide_into_grid(img, grid_size=(8, 8))

        # Extract features from each cell
        for cell_idx, cell in enumerate(cells):
            features = extract_features_from_cell(cell)
            labels = labels_dict[img_name][cell_idx]

            X_list.append(features)
            y_ball_list.append(labels['ball'])
            y_bat_list.append(labels['bat'])
            y_stump_list.append(labels['stump'])
            image_names_list.append(img_name)
            cell_index_list.append(cell_idx)

    X = np.array(X_list, dtype=np.float32)
    y_ball = np.array(y_ball_list, dtype=np.int32)
    y_bat = np.array(y_bat_list, dtype=np.int32)
    y_stump = np.array(y_stump_list, dtype=np.int32)

    # Create multi-label array for evaluation (N, 3) - supports multiple objects per cell
    # Each row: [ball_present, bat_present, stump_present] (0/1 binary flags)
    # This format allows cells to have multiple objects (e.g., [1,1,0] = ball+bat)
    y_multiclass = np.column_stack([y_ball, y_bat, y_stump]).astype(np.int32)

    # Create single-label version for DataFrame metadata (for undersampling backward compatibility)
    # Priority: ball > bat > stump > no object
    y_single_label = np.zeros(len(y_ball), dtype=np.int32)
    for i in range(len(y_ball)):
        if y_ball[i] == 1:
            y_single_label[i] = 1
        elif y_bat[i] == 1:
            y_single_label[i] = 2
        elif y_stump[i] == 1:
            y_single_label[i] = 3

    # Create DataFrame with metadata (required for undersampling)
    df = pd.DataFrame({
        'image_name': image_names_list,
        'cell_index': cell_index_list,
        'label': y_single_label.tolist()
    })

    if verbose:
        print(f"\n{'='*70}")
        print(f"Feature Extraction Complete")
        print(f"{'='*70}")
        print(f"Total samples: {len(X)}")
        print(f"Features per sample: {X.shape[1]}")
        print(f"\nBinary Label Distribution:")
        print(f"  Ball:  {np.sum(y_ball)} positive ({100*np.mean(y_ball):.2f}%)")
        print(f"  Bat:   {np.sum(y_bat)} positive ({100*np.mean(y_bat):.2f}%)")
        print(f"  Stump: {np.sum(y_stump)} positive ({100*np.mean(y_stump):.2f}%)")
        print(f"{'='*70}\n")

    # Create combined DataFrames for CSV export (metadata + features + binary label)
    # Each DataFrame will have: image_name, cell_index, label, + 2343 feature columns
    if verbose:
        print("Creating combined DataFrames for CSV export...")

    feature_names = get_all_feature_names()  # 2343 descriptive feature names

    # Ball DataFrame (label is binary: 0=not ball, 1=ball)
    df_ball_combined = pd.DataFrame({
        'image_name': image_names_list,
        'cell_index': cell_index_list,
        'label': y_ball.tolist()
    })
    for i, name in enumerate(feature_names):
        df_ball_combined[name] = X[:, i]

    # Bat DataFrame (label is binary: 0=not bat, 1=bat)
    df_bat_combined = pd.DataFrame({
        'image_name': image_names_list,
        'cell_index': cell_index_list,
        'label': y_bat.tolist()
    })
    for i, name in enumerate(feature_names):
        df_bat_combined[name] = X[:, i]

    # Stump DataFrame (label is binary: 0=not stump, 1=stump)
    df_stump_combined = pd.DataFrame({
        'image_name': image_names_list,
        'cell_index': cell_index_list,
        'label': y_stump.tolist()
    })
    for i, name in enumerate(feature_names):
        df_stump_combined[name] = X[:, i]

    if verbose:
        print(f"  ✓ Created ball DataFrame: {df_ball_combined.shape} (3 metadata + 2343 features)")
        print(f"  ✓ Created bat DataFrame: {df_bat_combined.shape}")
        print(f"  ✓ Created stump DataFrame: {df_stump_combined.shape}\n")

    return X, y_ball, y_bat, y_stump, y_multiclass, image_names_list, df, df_ball_combined, df_bat_combined, df_stump_combined


def mode_extract(args):
    """Extract features and save to disk."""
    print(f"\n{'='*70}")
    print(f"MODE: EXTRACT - Extracting features from images")
    print(f"{'='*70}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    print("Loading training images...")
    train_images = load_images_from_directory(args.train_images)
    train_labels_df = load_labels_from_csv(args.train_labels)

    print(f"Extracting training features from {len(train_images)} images...")
    X_train, y_ball_train, y_bat_train, y_stump_train, y_train_mc, _, train_df, \
        df_ball_train, df_bat_train, df_stump_train = \
        extract_features_and_labels_binary(train_images, train_labels_df, verbose=True)

    # Apply undersampling if requested (TRAINING DATA ONLY)
    if args.undersample:
        print(f"\n{'='*70}")
        print(f"UNDERSAMPLING TRAINING DATA (SEPARATE FOR EACH BINARY MODEL)")
        print(f"{'='*70}\n")

        # Calculate sample weights BEFORE undersampling for each binary classifier
        from sklearn.utils.class_weight import compute_class_weight

        print("Calculating sample weights before undersampling...")

        # BALL sample weights (before undersampling)
        classes_ball = np.unique(y_ball_train)
        class_weights_ball = compute_class_weight('balanced', classes=classes_ball, y=y_ball_train)
        weight_dict_ball = dict(zip(classes_ball, class_weights_ball))
        sample_weights_ball = np.array([weight_dict_ball[c] for c in y_ball_train])

        # BAT sample weights (before undersampling)
        classes_bat = np.unique(y_bat_train)
        class_weights_bat = compute_class_weight('balanced', classes=classes_bat, y=y_bat_train)
        weight_dict_bat = dict(zip(classes_bat, class_weights_bat))
        sample_weights_bat = np.array([weight_dict_bat[c] for c in y_bat_train])

        # STUMP sample weights (before undersampling)
        classes_stump = np.unique(y_stump_train)
        class_weights_stump = compute_class_weight('balanced', classes=classes_stump, y=y_stump_train)
        weight_dict_stump = dict(zip(classes_stump, class_weights_stump))
        sample_weights_stump = np.array([weight_dict_stump[c] for c in y_stump_train])

        print(f"  Ball weights calculated: shape {sample_weights_ball.shape}")
        print(f"  Bat weights calculated: shape {sample_weights_bat.shape}")
        print(f"  Stump weights calculated: shape {sample_weights_stump.shape}\n")
        print(f"  Ball Class weights : {weight_dict_ball}\n")
        print(f"  Bat Class weights : {weight_dict_bat}\n")
        print(f"  Stump Class weights : {weight_dict_stump}\n")

        # BALL MODEL - undersample negative class (not-ball: 0, bat, stump)
        print("--- BALL MODEL ---")
        X_train_ball, y_ball_train, df_ball , ball_indices = apply_undersampling(
            X_train, y_ball_train, train_df,
            strategy=args.undersample_strategy,
            target_ratio='max_minority',
            random_seed=42,
            verbose=True,
            target_count_ratio=args.undersample_ratio
        )

        # Get undersampled indices to map weights
        # ball_indices = df_ball.index.values
        sample_weights_ball_undersampled = sample_weights_ball[ball_indices]

        print(f"Ball model after undersampling: X_train_ball shape = {X_train_ball.shape}")
        print(f"  Sample weights shape: {sample_weights_ball_undersampled.shape}\n")
        

        # BAT MODEL - undersample negative class (not-bat: 0, ball, stump)
        print("--- BAT MODEL ---")
        X_train_bat, y_bat_train, df_bat , bat_indices = apply_undersampling(
            X_train, y_bat_train, train_df,
            strategy=args.undersample_strategy,
            target_ratio='max_minority',
            random_seed=42,
            verbose=True,
            target_count_ratio=args.undersample_ratio
        )

        # Get undersampled indices to map weights
        # bat_indices = df_bat.index.values
        sample_weights_bat_undersampled = sample_weights_bat[bat_indices]

        print(f"Bat model after undersampling: X_train_bat shape = {X_train_bat.shape}")
        print(f"  Sample weights shape: {sample_weights_bat_undersampled.shape}\n")

        # STUMP MODEL - undersample negative class (not-stump: 0, ball, bat)
        print("--- STUMP MODEL ---")
        X_train_stump, y_stump_train, df_stump , stump_indices = apply_undersampling(
            X_train, y_stump_train, train_df,
            strategy=args.undersample_strategy,
            target_ratio='max_minority',
            random_seed=42,
            verbose=True,
            target_count_ratio=args.undersample_ratio
        )

        # Get undersampled indices to map weights
        # stump_indices = df_stump.index.values
        sample_weights_stump_undersampled = sample_weights_stump[stump_indices]

        print(f"Stump model after undersampling: X_train_stump shape = {X_train_stump.shape}")
        print(f"  Sample weights shape: {sample_weights_stump_undersampled.shape}\n")

        # Save separate training matrices for each model
        np.save(output_dir / 'X_train_ball.npy', X_train_ball)
        np.save(output_dir / 'y_ball_train.npy', y_ball_train)
        np.save(output_dir / 'X_train_bat.npy', X_train_bat)
        np.save(output_dir / 'y_bat_train.npy', y_bat_train)
        np.save(output_dir / 'X_train_stump.npy', X_train_stump)
        np.save(output_dir / 'y_stump_train.npy', y_stump_train)

        # Save sample weights (undersampled)
        np.save(output_dir / 'weights_ball.npy', sample_weights_ball_undersampled)
        np.save(output_dir / 'weights_bat.npy', sample_weights_bat_undersampled)
        np.save(output_dir / 'weights_stump.npy', sample_weights_stump_undersampled)
        print(f"\nSaved undersampled sample weights:")
        print(f"  weights_ball.npy: {sample_weights_ball_undersampled.shape}")
        print(f"  weights_bat.npy: {sample_weights_bat_undersampled.shape}")
        print(f"  weights_stump.npy: {sample_weights_stump_undersampled.shape}")

        # Save metadata
        df_ball.to_csv(output_dir / 'train_metadata_ball.csv', index=False)
        df_bat.to_csv(output_dir / 'train_metadata_bat.csv', index=False)
        df_stump.to_csv(output_dir / 'train_metadata_stump.csv', index=False)

        # Save original X_train for reference
        np.save(output_dir / 'X_train_original.npy', X_train)
        np.save(output_dir / 'y_train_multiclass_original.npy', y_train_mc)

        # Save original (non-undersampled) training CSVs
        print("\nSaving original (non-undersampled) training CSVs...")
        df_ball_train.to_csv(output_dir / 'train_ball_features.csv', index=False)
        df_bat_train.to_csv(output_dir / 'train_bat_features.csv', index=False)
        df_stump_train.to_csv(output_dir / 'train_stump_features.csv', index=False)
        print(f"  ✓ Saved train_ball_features.csv ({len(df_ball_train):,} rows, {len(df_ball_train.columns)} columns)")
        print(f"  ✓ Saved train_bat_features.csv ({len(df_bat_train):,} rows, {len(df_bat_train.columns)} columns)")
        print(f"  ✓ Saved train_stump_features.csv ({len(df_stump_train):,} rows, {len(df_stump_train.columns)} columns)")

        # Reconstruct undersampled DataFrames from undersampled X, y, and metadata
        print("\nSaving undersampled training CSVs...")
        feature_names = get_all_feature_names()

        # Ball undersampled
        df_ball_train_undersampled = pd.DataFrame({
            'image_name': df_ball['image_name'].values,
            'cell_index': df_ball['cell_index'].values,
            'label': y_ball_train  # Binary: 0=not ball, 1=ball (already undersampled)
        })
        for i, name in enumerate(feature_names):
            df_ball_train_undersampled[name] = X_train_ball[:, i]

        # Bat undersampled
        df_bat_train_undersampled = pd.DataFrame({
            'image_name': df_bat['image_name'].values,
            'cell_index': df_bat['cell_index'].values,
            'label': y_bat_train  # Binary: 0=not bat, 1=bat (already undersampled)
        })
        for i, name in enumerate(feature_names):
            df_bat_train_undersampled[name] = X_train_bat[:, i]

        # Stump undersampled
        df_stump_train_undersampled = pd.DataFrame({
            'image_name': df_stump['image_name'].values,
            'cell_index': df_stump['cell_index'].values,
            'label': y_stump_train  # Binary: 0=not stump, 1=stump (already undersampled)
        })
        for i, name in enumerate(feature_names):
            df_stump_train_undersampled[name] = X_train_stump[:, i]

        df_ball_train_undersampled.to_csv(output_dir / 'train_ball_undersampled_features.csv', index=False)
        df_bat_train_undersampled.to_csv(output_dir / 'train_bat_undersampled_features.csv', index=False)
        df_stump_train_undersampled.to_csv(output_dir / 'train_stump_undersampled_features.csv', index=False)
        print(f"  ✓ Saved train_ball_undersampled_features.csv ({len(df_ball_train_undersampled):,} rows, {len(df_ball_train_undersampled.columns)} columns)")
        print(f"  ✓ Saved train_bat_undersampled_features.csv ({len(df_bat_train_undersampled):,} rows, {len(df_bat_train_undersampled.columns)} columns)")
        print(f"  ✓ Saved train_stump_undersampled_features.csv ({len(df_stump_train_undersampled):,} rows, {len(df_stump_train_undersampled.columns)} columns)")

        print(f"\n{'='*70}")
        print(f"UNDERSAMPLING COMPLETE - Separate matrices saved for each model")
        print(f"{'='*70}\n")

    else:
        # No undersampling - save standard format
        np.save(output_dir / 'X_train.npy', X_train)
        np.save(output_dir / 'y_ball_train.npy', y_ball_train)
        np.save(output_dir / 'y_bat_train.npy', y_bat_train)
        np.save(output_dir / 'y_stump_train.npy', y_stump_train)
        np.save(output_dir / 'y_train_multiclass.npy', y_train_mc)
        train_df.to_csv(output_dir / 'train_metadata.csv', index=False)

        # Save training CSVs (no undersampling)
        print("\nSaving training CSVs...")
        df_ball_train.to_csv(output_dir / 'train_ball_features.csv', index=False)
        df_bat_train.to_csv(output_dir / 'train_bat_features.csv', index=False)
        df_stump_train.to_csv(output_dir / 'train_stump_features.csv', index=False)
        print(f"  ✓ Saved train_ball_features.csv ({len(df_ball_train):,} rows, {len(df_ball_train.columns)} columns)")
        print(f"  ✓ Saved train_bat_features.csv ({len(df_bat_train):,} rows, {len(df_bat_train.columns)} columns)")
        print(f"  ✓ Saved train_stump_features.csv ({len(df_stump_train):,} rows, {len(df_stump_train.columns)} columns)")

    # Load validation data
    print("\nLoading validation images...")
    validate_images = load_images_from_directory(args.validate_images)
    validate_labels_df = load_labels_from_csv(args.validate_labels)

    print(f"Extracting validation features from {len(validate_images)} images...")
    X_val, y_ball_val, y_bat_val, y_stump_val, y_val_mc, _, val_df, \
        df_ball_val, df_bat_val, df_stump_val = \
        extract_features_and_labels_binary(validate_images, validate_labels_df, verbose=True)

    # Save validation features (NO undersampling for validation - keep real distribution)
    np.save(output_dir / 'X_val.npy', X_val)
    np.save(output_dir / 'y_ball_val.npy', y_ball_val)
    np.save(output_dir / 'y_bat_val.npy', y_bat_val)
    np.save(output_dir / 'y_stump_val.npy', y_stump_val)
    np.save(output_dir / 'y_val_multiclass.npy', y_val_mc)
    val_df.to_csv(output_dir / 'val_metadata.csv', index=False)

    # Save validation CSVs for each object (NO undersampling)
    print("\nSaving validation CSVs...")
    df_ball_val.to_csv(output_dir / 'validate_ball_features.csv', index=False)
    df_bat_val.to_csv(output_dir / 'validate_bat_features.csv', index=False)
    df_stump_val.to_csv(output_dir / 'validate_stump_features.csv', index=False)
    print(f"  ✓ Saved validate_ball_features.csv ({len(df_ball_val):,} rows, {len(df_ball_val.columns)} columns)")
    print(f"  ✓ Saved validate_bat_features.csv ({len(df_bat_val):,} rows, {len(df_bat_val.columns)} columns)")
    print(f"  ✓ Saved validate_stump_features.csv ({len(df_stump_val):,} rows, {len(df_stump_val.columns)} columns)")

    # Load test data (labels optional)
    print("\nLoading test images...")
    test_images = load_images_from_directory(args.test_images)

    if args.test_labels is not None:
        test_labels_df = load_labels_from_csv(args.test_labels)
        print(f"Extracting test features from {len(test_images)} images...")
        X_test, y_ball_test, y_bat_test, y_stump_test, y_test_mc, _, test_df = \
            extract_features_and_labels_binary(test_images, test_labels_df, verbose=True, is_test=True)

        # Save test features with labels (NO undersampling for test)
        np.save(output_dir / 'X_test.npy', X_test)
        np.save(output_dir / 'y_ball_test.npy', y_ball_test)
        np.save(output_dir / 'y_bat_test.npy', y_bat_test)
        np.save(output_dir / 'y_stump_test.npy', y_stump_test)
        np.save(output_dir / 'y_test_multiclass.npy', y_test_mc)
        test_df.to_csv(output_dir / 'test_metadata.csv', index=False)
    else:
        print("No test labels provided. Extracting features only (no labels)...")
        # For test without labels, extract features only
        X_list = []
        image_names_list = []

        for img_name, img in tqdm(test_images, desc="Extracting test features"):
            cells = divide_into_grid(img, grid_size=(8, 8))
            for cell in cells:
                features = extract_features_from_cell(cell)
                X_list.append(features)
                image_names_list.append(img_name)

        X_test = np.array(X_list, dtype=np.float32)
        np.save(output_dir / 'X_test.npy', X_test)
        print(f"Extracted {len(X_test)} test samples (features only)")

    print(f"\n{'='*70}")
    print(f"Features saved to: {output_dir}")
    print(f"{'='*70}\n")


def mode_train(args):
    """Train binary models."""
    print(f"\n{'='*70}")
    print(f"MODE: TRAIN - Training Binary Models")
    print(f"{'='*70}\n")

    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load features - check if undersampling was used
    print("Loading extracted features...")

    # Check if separate undersampled matrices exist
    has_undersampled = (output_dir / 'X_train_ball.npy').exists()

    # Initialize sample weights as None
    sample_weights_ball = None
    sample_weights_bat = None
    sample_weights_stump = None

    if has_undersampled:
        print("Loading separate undersampled matrices for each model...")
        X_train_ball = np.load(output_dir / 'X_train_ball.npy')
        y_ball_train = np.load(output_dir / 'y_ball_train.npy')
        X_train_bat = np.load(output_dir / 'X_train_bat.npy')
        y_bat_train = np.load(output_dir / 'y_bat_train.npy')
        X_train_stump = np.load(output_dir / 'X_train_stump.npy')
        y_stump_train = np.load(output_dir / 'y_stump_train.npy')

        print(f"  Ball model:  X_train_ball shape = {X_train_ball.shape}")
        print(f"  Bat model:   X_train_bat shape = {X_train_bat.shape}")
        print(f"  Stump model: X_train_stump shape = {X_train_stump.shape}")

        # Load sample weights if they exist
        weights_ball_path = output_dir / 'weights_ball.npy'
        weights_bat_path = output_dir / 'weights_bat.npy'
        weights_stump_path = output_dir / 'weights_stump.npy'

        if weights_ball_path.exists() and weights_bat_path.exists() and weights_stump_path.exists():
            sample_weights_ball = np.load(weights_ball_path)
            sample_weights_bat = np.load(weights_bat_path)
            sample_weights_stump = np.load(weights_stump_path)
            print(f"\nLoaded pre-computed sample weights:")
            print(f"  Ball weights:  shape = {sample_weights_ball.shape}")
            print(f"  Bat weights:   shape = {sample_weights_bat.shape}")
            print(f"  Stump weights: shape = {sample_weights_stump.shape}")
        else:
            print(f"\nNo pre-computed sample weights found. Will compute during training.")
    else:
        print("Loading standard X_train (no undersampling)...")
        X_train = np.load(output_dir / 'X_train.npy')
        y_ball_train = np.load(output_dir / 'y_ball_train.npy')
        y_bat_train = np.load(output_dir / 'y_bat_train.npy')
        y_stump_train = np.load(output_dir / 'y_stump_train.npy')

        # Use same matrix for all models
        X_train_ball = X_train_bat = X_train_stump = X_train
        print(f"  X_train shape = {X_train.shape}")

    algorithms = []
    if args.algorithm == 'both':
        algorithms = ['random_forest', 'xgboost']
    else:
        algorithms = [args.algorithm]

    for algorithm in algorithms:
        print(f"\n{'='*70}")
        print(f"Training {algorithm.upper()} Binary Models")
        print(f"{'='*70}")

        # Train 3 binary models with separate training sets (if undersampled) and optional feature selection/PCA
        model_data = train_three_binary_models(
            X_train_ball, y_ball_train,
            X_train_bat, y_bat_train,
            X_train_stump, y_stump_train,
            algorithm=algorithm,
            tune=args.tune,
            verbose=True,
            use_feature_selection=args.feature_selection,
            use_pca=args.use_pca,
            pca_variance=args.pca_variance,
            sample_weights_ball=sample_weights_ball,
            sample_weights_bat=sample_weights_bat,
            sample_weights_stump=sample_weights_stump
        )

        # Save models
        save_binary_models(model_data, model_dir, algorithm)

    print(f"\n{'='*70}")
    print(f"Training Complete")
    print(f"{'='*70}\n")


def mode_evaluate(args):
    """Evaluate binary models on validation set."""
    print(f"\n{'='*70}")
    print(f"MODE: EVALUATE - Evaluating Binary Models on Validation Set")
    print(f"{'='*70}\n")

    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)

    # Load validation features
    print("Loading validation features...")
    X_val = np.load(output_dir / 'X_val.npy')
    y_ball_val = np.load(output_dir / 'y_ball_val.npy')
    y_bat_val = np.load(output_dir / 'y_bat_val.npy')
    y_stump_val = np.load(output_dir / 'y_stump_val.npy')
    y_val_multiclass = np.load(output_dir / 'y_val_multiclass.npy')

    # Note: y_val_multiclass is now a 2D array (N, 3) where each row is [ball, bat, stump]
    # For backward compatibility, convert to 1D binary arrays if needed
    if y_val_multiclass.ndim == 2:
        # New multi-label format
        print(f"Loaded multi-label ground truth: shape {y_val_multiclass.shape}")
    else:
        # Old single-label format (backward compatibility)
        print(f"Warning: Old single-label format detected. Consider re-running feature extraction.")

    algorithms = []
    if args.algorithm == 'both':
        algorithms = ['random_forest', 'xgboost']
    else:
        algorithms = [args.algorithm]

    for algorithm in algorithms:
        print(f"\n{'='*70}")
        print(f"Evaluating {algorithm.upper()} Binary Models")
        print(f"{'='*70}")

        # Create algorithm-specific output directory
        algo_output_dir = output_dir / algorithm
        algo_output_dir.mkdir(parents=True, exist_ok=True)

        # Load models
        models = load_binary_models(model_dir, algorithm)

        # Evaluate independently
        print("\n1. Independent Binary Evaluation")
        evaluate_binary_models_independently(
            models, X_val, y_ball_val, y_bat_val, y_stump_val,
            output_dir=algo_output_dir, verbose=True
        )

        # Threshold optimization (if requested)
        if args.optimize_thresholds:
            print("\n2. Threshold Optimization")
            print(f"{'='*70}")

            # Extract models dict from model_data (handle both old dict and new model_data format)
            if isinstance(models, dict) and 'models' in models:
                models_dict = models['models']
                feature_indices = models.get('feature_indices')
                scalers = models.get('scalers')
                pca_models = models.get('pca_models')
            else:
                models_dict = models
                feature_indices = None

            # Apply feature selection/PCA to validation data (if used during training)
            X_val_ball = X_val_bat = X_val_stump = X_val

            if feature_indices:
                from feature_selector import get_ball_feature_indices, get_bat_feature_indices, get_stump_feature_indices
                X_val_ball = X_val[:, feature_indices['ball']]
                X_val_bat = X_val[:, feature_indices['bat']]
                X_val_stump = X_val[:, feature_indices['stump']]

                if pca_models:
                    X_val_ball = scalers['ball'].transform(X_val_ball)
                    X_val_ball = pca_models['ball'].transform(X_val_ball)

                    X_val_bat = scalers['bat'].transform(X_val_bat)
                    X_val_bat = pca_models['bat'].transform(X_val_bat)

                    X_val_stump = scalers['stump'].transform(X_val_stump)
                    X_val_stump = pca_models['stump'].transform(X_val_stump)

            # Create temporary dict with transformed features for each model
            X_val_dict = {
                'ball': X_val_ball,
                'bat': X_val_bat,
                'stump': X_val_stump
            }

            # Optimize thresholds
            optimal_thresholds = optimize_thresholds_all_models(
                models_dict, X_val_dict, y_ball_val, y_bat_val, y_stump_val,
                metric=args.threshold_metric, verbose=True
            )

            # Evaluate threshold impact (generates plots)
            evaluate_threshold_impact(
                models_dict, X_val_dict, y_ball_val, y_bat_val, y_stump_val,
                output_dir=algo_output_dir, verbose=True
            )

            # Compare default vs optimal thresholds
            default_thresholds = {'ball': 0.5, 'bat': 0.5, 'stump': 0.5}
            plot_threshold_comparison(
                models_dict, X_val_dict, y_ball_val, y_bat_val, y_stump_val,
                default_thresholds, optimal_thresholds,
                output_dir=algo_output_dir
            )

            # Save optimized thresholds
            import json
            threshold_file = algo_output_dir / 'optimal_thresholds.json'
            with open(threshold_file, 'w') as f:
                json.dump(optimal_thresholds, f, indent=4)
            print(f"\nSaved optimal thresholds to: {threshold_file}")
            print(f"{'='*70}\n")

            step_num = 3
        else:
            step_num = 2

        # Load optimal thresholds if available
        thresholds = None
        threshold_file = algo_output_dir / 'optimal_thresholds.json'
        if threshold_file.exists():
            import json
            with open(threshold_file, 'r') as f:
                thresholds = json.load(f)
            print(f"Loaded optimized thresholds from: {threshold_file}")

        evaluate_binary_models_independently(
            models, X_val, y_ball_val, y_bat_val, y_stump_val,
            output_dir=algo_output_dir / 'after-threshold', verbose=True, thresholds=thresholds
        )
        # Get predictions
        print(f"\n{step_num}. Generating Final Predictions")
        y_pred = predict_with_binary_models(models, X_val, thresholds=thresholds, verbose=True)

        # Evaluate final predictions - create single-label ground truth based on predictions
        print(f"\n{step_num + 1}. Final Multi-Label Evaluation")

        if y_val_multiclass.ndim == 2:
            # New multi-label format: Create true_y_val_multiclass based on predictions
            # A prediction is correct if the predicted object is present in the ground truth
            y_ball_true = y_val_multiclass[:, 0]
            y_bat_true = y_val_multiclass[:, 1]
            y_stump_true = y_val_multiclass[:, 2]

            # Create single-label ground truth from multi-label based on predictions
            true_y_val_multiclass = np.zeros(len(y_pred), dtype=np.int32)
            for i in range(len(y_pred)):
                if y_pred[i] == 0:
                    # Predicted "no object" - mark as correct (0) only if no objects present
                    if y_ball_true[i] == 0 and y_bat_true[i] == 0 and y_stump_true[i] == 0:
                        true_y_val_multiclass[i] = 0
                    else:
                        # Predicted "no object" but objects are present - assign actual object with priority
                        if y_ball_true[i] == 1:
                            true_y_val_multiclass[i] = 1
                        elif y_bat_true[i] == 1:
                            true_y_val_multiclass[i] = 2
                        elif y_stump_true[i] == 1:
                            true_y_val_multiclass[i] = 3
                elif y_pred[i] == 1:
                    # Predicted "ball" - mark as correct (1) if ball is present
                    true_y_val_multiclass[i] = 1 if y_ball_true[i] == 1 else 0
                elif y_pred[i] == 2:
                    # Predicted "bat" - mark as correct (2) if bat is present
                    true_y_val_multiclass[i] = 2 if y_bat_true[i] == 1 else 0
                elif y_pred[i] == 3:
                    # Predicted "stump" - mark as correct (3) if stump is present
                    true_y_val_multiclass[i] = 3 if y_stump_true[i] == 1 else 0

            print(f"Created prediction-aligned ground truth from multi-label array")
        else:
            # Backward compatibility: use loaded single-label ground truth
            true_y_val_multiclass = y_val_multiclass
            print(f"Using single-label ground truth")

        evaluate_final_predictions(
            y_pred, true_y_val_multiclass,
            output_dir=algo_output_dir, verbose=True
        )

        # Analyze conflicts
        print(f"\n{step_num + 2}. Conflict Analysis")
        # analyze_conflict_rate(models, X_val, y_val_multiclass, verbose=True)

    print(f"\n{'='*70}")
    print(f"Evaluation Complete")
    print(f"{'='*70}\n")


def predict_and_save_dataset(
    models: dict,
    X: np.ndarray,
    images: list,
    image_dir: Path,
    output_dir: Path,
    dataset_name: str,
    thresholds: dict = None,
    visualize: bool = False,
    verbose: bool = True
) -> None:
    """
    Generate predictions for a single dataset and save to segregated folder.

    Args:
        models: Dictionary with keys 'ball', 'bat', 'stump' (trained models)
        X: Feature array (N, 2343) where N = num_images * 64
        images: List of (image_name, image_array) tuples
        image_dir: Path to source images (for visualization)
        output_dir: Base output directory (will create dataset_name subdirectory)
        dataset_name: One of 'train', 'validate', 'test'
        thresholds: Optional dict {'ball': 0.6, 'bat': 0.4, 'stump': 0.4}
                   If None, uses default 0.5 for all models
        visualize: Whether to generate visualization overlays
        verbose: Print progress messages

    Side Effects:
        Creates directory structure:
        - {output_dir}/{dataset_name}/predictions.csv
        - {output_dir}/{dataset_name}/visualizations/*.png (if visualize=True)
    """
    # Create dataset-specific output directory
    dataset_output_dir = output_dir / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nProcessing {dataset_name} dataset...")
        print(f"  Samples: {len(X)}")
        print(f"  Images: {len(images)}")

    # Generate predictions
    y_pred = predict_with_binary_models(models, X, thresholds=thresholds, verbose=verbose)

    # Convert predictions to DataFrame format
    # Group predictions by image (64 cells per image)
    predictions_list = []
    n_images = len(y_pred) // 64

    for img_idx in range(n_images):
        img_name = images[img_idx][0]
        cell_preds = y_pred[img_idx * 64:(img_idx + 1) * 64]

        # Use dataset_name for TrainOrTest column
        row = {'ImageFileName': img_name, 'TrainOrTest': dataset_name}
        for cell_idx, pred in enumerate(cell_preds):
            row[f'c{cell_idx+1:02d}'] = pred

        predictions_list.append(row)

    pred_df = pd.DataFrame(predictions_list)

    # Save to CSV
    pred_file = dataset_output_dir / 'predictions.csv'
    pred_df.to_csv(pred_file, index=False)

    if verbose:
        print(f"  Saved predictions: {pred_file}")

    # Visualize if requested
    if visualize:
        if verbose:
            print(f"  Generating visualizations...")

        vis_dir = dataset_output_dir / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)

        for img_idx, (img_name, img) in enumerate(images):
            cell_preds = y_pred[img_idx * 64:(img_idx + 1) * 64]
            img_path = image_dir / img_name
            viz_single_image(img_path, cell_preds, vis_dir / img_name)

        if verbose:
            print(f"  Visualizations saved: {vis_dir}")


def mode_predict(args):
    """Generate predictions on train, validate, and test sets."""
    print(f"\n{'='*70}")
    print(f"MODE: PREDICT - Generating Predictions for All Datasets")
    print(f"{'='*70}\n")

    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)

    # Define dataset configurations
    datasets = [
        {
            'name': 'train',
            'features_file': 'X_train_original.npy',
            'image_dir': args.train_images,
            'required': True
        },
        {
            'name': 'validate',
            'features_file': 'X_val.npy',
            'image_dir': args.validate_images,
            'required': True
        },
        {
            'name': 'test',
            'features_file': 'X_test.npy',
            'image_dir': args.test_images,
            'required': False
        }
    ]

    # Load all available datasets
    print("Loading datasets...")
    loaded_datasets = []

    for ds_config in datasets:
        features_path = output_dir / ds_config['features_file']

        if not features_path.exists():
            if ds_config['required']:
                print(f"\nERROR: Required features not found: {features_path}")
                print(f"Please run: python main_pipeline_binary.py --mode extract")
                return
            else:
                print(f"  Skipping {ds_config['name']}: features not found at {features_path}")
                continue

        try:
            X = np.load(features_path)
            images = load_images_from_directory(ds_config['image_dir'])

            # Validate data consistency
            expected_samples = len(images) * 64
            if len(X) != expected_samples:
                print(f"\nWARNING: {ds_config['name']} data mismatch!")
                print(f"  Features: {len(X)} samples")
                print(f"  Images: {len(images)} images ({expected_samples} expected samples)")
                print(f"  Proceeding anyway, but results may be incorrect")

            loaded_datasets.append({
                'name': ds_config['name'],
                'X': X,
                'images': images,
                'image_dir': Path(ds_config['image_dir'])
            })

            print(f"  Loaded {ds_config['name']}: {len(X)} samples from {len(images)} images")

        except Exception as e:
            if ds_config['required']:
                print(f"\nERROR loading {ds_config['name']}: {e}")
                return
            else:
                print(f"  Warning: Could not load {ds_config['name']}: {e}")
                continue

    if len(loaded_datasets) == 0:
        print("\nERROR: No datasets available for prediction")
        print("Please run: python main_pipeline_binary.py --mode extract")
        return

    print(f"\nSuccessfully loaded {len(loaded_datasets)} dataset(s)")

    # Determine which algorithms to use
    algorithms = []
    if args.algorithm == 'both':
        algorithms = ['random_forest', 'xgboost']
    else:
        algorithms = [args.algorithm]

    # Process each algorithm
    for algorithm in algorithms:
        print(f"\n{'='*70}")
        print(f"Algorithm: {algorithm.upper()}")
        print(f"{'='*70}")

        # Load models
        try:
            models = load_binary_models(model_dir, algorithm)
        except Exception as e:
            print(f"\nERROR: Could not load {algorithm} models: {e}")
            print(f"Please run: python main_pipeline_binary.py --mode train --algorithm {algorithm}")
            continue

        # Create algorithm-specific output directory
        algo_output_dir = output_dir / algorithm
        algo_output_dir.mkdir(parents=True, exist_ok=True)

        # Load optimal thresholds if available
        thresholds = None
        threshold_file = algo_output_dir / 'optimal_thresholds.json'
        if threshold_file.exists():
            import json
            with open(threshold_file, 'r') as f:
                thresholds = json.load(f)
            print(f"\nUsing optimized thresholds from: {threshold_file}")
            print(f"Thresholds: Ball={thresholds['ball']:.2f}, Bat={thresholds['bat']:.2f}, Stump={thresholds['stump']:.2f}")

        # Process each dataset
        for dataset in loaded_datasets:
            predict_and_save_dataset(
                models=models,
                X=dataset['X'],
                images=dataset['images'],
                image_dir=dataset['image_dir'],
                output_dir=algo_output_dir,
                dataset_name=dataset['name'],
                thresholds=thresholds,
                visualize=args.visualize,
                verbose=True
            )

        # Backward compatibility: copy test predictions to root level
        test_pred_file = algo_output_dir / 'test' / 'predictions.csv'
        if test_pred_file.exists():
            try:
                import shutil
                legacy_pred_file = algo_output_dir / 'predictions.csv'
                shutil.copy(test_pred_file, legacy_pred_file)
                print(f"\n  Created backward-compatible predictions.csv at {legacy_pred_file}")
            except Exception as e:
                print(f"\n  Warning: Could not create legacy predictions.csv: {e}")

        print(f"\n{algorithm.upper()} predictions complete")
        print(f"Output directory: {algo_output_dir}")

    print(f"\n{'='*70}")
    print(f"PREDICTION COMPLETE - Processed {len(loaded_datasets)} dataset(s)")
    print(f"{'='*70}\n")


def mode_full(args):
    """Run full pipeline: extract → train → evaluate → predict."""
    print(f"\n{'='*70}")
    print(f"MODE: FULL - Complete Binary Model Pipeline")
    print(f"{'='*70}\n")

    # Step 1: Extract
    mode_extract(args)

    # Step 2: Train
    mode_train(args)

    # Step 3: Evaluate
    mode_evaluate(args)

    # Step 4: Predict
    mode_predict(args)

    print(f"\n{'='*70}")
    print(f"FULL PIPELINE COMPLETE")
    print(f"{'='*70}\n")


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(
        description='Binary Model Pipeline for Cricket Object Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline with Random Forest
    python main_pipeline_binary.py --mode full --algorithm random_forest --tune

    # Full pipeline with both RF and XGBoost
    python main_pipeline_binary.py --mode full --algorithm both --tune

    # Train only
    python main_pipeline_binary.py --mode train --algorithm xgboost --tune

    # Evaluate with visualization
    python main_pipeline_binary.py --mode evaluate --algorithm random_forest --visualize
        """
    )

    parser.add_argument('--mode', type=str,
                       choices=['extract', 'train', 'evaluate', 'predict', 'full'],
                       required=True,
                       help='Pipeline mode')

    parser.add_argument('--algorithm', type=str,
                       choices=['random_forest', 'xgboost', 'both'],
                       default='random_forest',
                       help='Algorithm to use (default: random_forest)')

    parser.add_argument('--train_images', type=str,
                       default='dataset/train/images',
                       help='Directory with training images')

    parser.add_argument('--train_labels', type=str,
                       default='dataset/train/labels/labels_multilabel.csv',
                       help='Path to training labels CSV (single-label or multi-label format auto-detected)')

    parser.add_argument('--validate_images', type=str,
                       default='dataset/validate/images',
                       help='Directory with validation images')

    parser.add_argument('--validate_labels', type=str,
                       default='dataset/validate/labels/labels_multilabel.csv',
                       help='Path to validation labels CSV (single-label or multi-label format auto-detected)')

    parser.add_argument('--test_images', type=str,
                       default='dataset/test/images',
                       help='Directory with test images')

    parser.add_argument('--test_labels', type=str,
                       default=None,
                       help='Path to test labels CSV (optional, single-label or multi-label format auto-detected)')

    parser.add_argument('--model_dir', type=str,
                       default='models/binary_models',
                       help='Directory to save/load binary models')

    parser.add_argument('--output_dir', type=str,
                       default='outputs/binary_models',
                       help='Directory for outputs')

    parser.add_argument('--tune', action='store_true',
                       help='Perform hyperparameter tuning (slower but better)')

    parser.add_argument('--visualize', action='store_true',
                       help='Generate color-coded prediction grids')

    # Feature selection and PCA arguments
    parser.add_argument('--feature_selection', action='store_true',
                       help='Use selective features per model (reduces 2343->~1000 features)')

    parser.add_argument('--use_pca', action='store_true',
                       help='Apply PCA after feature selection for dimensionality reduction')

    parser.add_argument('--pca_variance', type=float, default=0.95,
                       help='PCA variance threshold (default: 0.95, i.e., 95%% variance retained)')

    # Undersampling arguments
    parser.add_argument('--undersample', action='store_true',
                       help='Apply undersampling to balance class distribution (separate for each binary model)')

    parser.add_argument('--undersample_strategy', type=str,
                       choices=['image_stratified', 'random'],
                       default='image_stratified',
                       help='Undersampling strategy (default: image_stratified - preserves spatial diversity)')

    parser.add_argument('--undersample_ratio', type=float, default=40,
                       help='Undersampling target ratio (default: 1.25x max minority class count)')

    # Threshold optimization arguments
    parser.add_argument('--optimize_thresholds', action='store_true',
                       help='Optimize decision thresholds using validation data (improves F1 score)')
    parser.add_argument('--threshold_metric', type=str, default='f1',
                       choices=['f1', 'precision', 'recall', 'accuracy'],
                       help='Metric to optimize when tuning thresholds (default: f1)')

    args = parser.parse_args()

    # Execute based on mode
    if args.mode == 'extract':
        mode_extract(args)
    elif args.mode == 'train':
        mode_train(args)
    elif args.mode == 'evaluate':
        mode_evaluate(args)
    elif args.mode == 'predict':
        mode_predict(args)
    elif args.mode == 'full':
        mode_full(args)


if __name__ == '__main__':
    main()
