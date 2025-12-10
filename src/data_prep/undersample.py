"""
Undersampling utilities for handling class imbalance.

This module implements stratified random undersampling to balance the heavily
imbalanced cricket object detection dataset. Class 0 (no object) dominates with
87% of samples, while minority classes (ball, bat, stump) have only 3-5% each.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from collections import Counter

from imblearn.over_sampling import SMOTE

def print_class_distribution(y):
    class_names = {0: 'No Object', 1: 'Ball', 2: 'Bat', 3: 'Stump', 4: 'Other No Objects '}
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    for cls, count in zip(unique, counts):
        pct = (count / total) * 100
        print(f"  Class {cls} ({class_names.get(cls,'Unknown')}): {count} samples ({pct:.2f}%)")


def oversample_class_1(
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    target_count_ratio: float = .85,
    random_seed: int = 42,
    verbose: bool = True
):
    """
    Oversample ONLY Class 1 (Ball) to reach target_count.

    Args:
        X: Feature matrix
        y: Label vector
        df: DataFrame aligned with X,y
        target_count: Final desired number of samples for class 1
        random_seed: Reproducibility
        verbose: Print distribution before/after

    Returns:
        X_over, y_over, df_over
    """

    np.random.seed(random_seed)

    if verbose:
        print("="*70)
        print("OVERSAMPLING ONLY CLASS 1")
        print("="*70)
        print("Original class distribution:")
        print_class_distribution(y)

    # --------------------------
    # Class 1 indices
    # --------------------------
    class1_idx = np.where(y == 1)[0]
    current_count = len(class1_idx)

    # Calculate target count
    class_counts = Counter(y)
    minority_classes = [0, 2, 3, 4]  # no objects, bat, stump, other no object
    # target_count = 3*max(class_counts[cls] for cls in minority_classes)
    target_count = int(target_count_ratio*max(class_counts[cls] for cls in minority_classes))


    if current_count >= target_count:
        if verbose:
            print(f"\nClass 1 already has {current_count} samples ≥ target {target_count}. No oversampling done.")
        return X, y, df

    # --------------------------
    # Oversampling
    # --------------------------
    extra_needed = target_count - current_count
    extra_samples = np.random.choice(class1_idx, size=extra_needed, replace=True)

    # Combine indices
    final_indices = np.concatenate([np.arange(len(y)), extra_samples])
    np.random.shuffle(final_indices)

    X_over = X[final_indices]
    y_over = y[final_indices]
    df_over = df.iloc[final_indices].reset_index(drop=True)

    if verbose:
        print(f"\nTarget Class 1 size: {target_count}")
        print("\nNew class distribution:")
        print_class_distribution(y_over)
        print("="*70)

    return X_over, y_over, df_over

def apply_oversampling (X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    target_count_ratio: float = .85,
    random_seed: int = 42,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Apply oversampling to balance class distribution by oversampling Class 1 (Ball).

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Label array (n_samples,)
        df: DataFrame with columns [image_name, cell_index, label, features...]
        target_count_ratio: Ratio to determine target count for Class 1
        random_seed: Random seed for reproducibility
        verbose: Print statistics
    Returns:
        Tuple of (X_oversampled, y_oversampled, df_oversampled)
    """

    np.random.seed(random_seed)

    if verbose:
        print("="*70)
        print("APPLYING OVERSAMPLING TO CLASS 1")
        print("="*70)
        print(f"Target count ratio: {target_count_ratio}")
        print(f"Random seed: {random_seed}")
        print(f"\nOriginal class distribution:")
        print_class_distribution(y)

    # Apply oversampling to Class 1
    # X_over, y_over, df_over = oversample_class_1(
    #     X, y, df,
    #     target_count_ratio=target_count_ratio,
    #     random_seed=random_seed,
    #     verbose=verbose
    # )

    class1_idx = np.where(y == 1)[0]
    current_count = len(class1_idx)

    # Calculate target count
    class_counts = Counter(y)
    minority_classes = [0, 2, 3, 4]  # no objects, bat, stump, other no object
    # target_count = 3*max(class_counts[cls] for cls in minority_classes)
    target_count = int(target_count_ratio*max(class_counts[cls] for cls in minority_classes))


    if current_count >= target_count:
        if verbose:
            print(f"\nClass 1 already has {current_count} samples ≥ target {target_count}. No oversampling done.")
        return X, y, df

    # sm = SMOTE(k_neighbors=2, sampling_strategy={1:target_count, 2:target_count}, random_state=random_seed)
    sm = SMOTE(k_neighbors=2, sampling_strategy={1:target_count}, random_state=random_seed)
    X_over, y_over = sm.fit_resample(X, y)



    if verbose:
        print(f"\nFinal dataset size after oversampling: {len(y_over):,} samples")
        print("="*70)

    return X_over, y_over, df

def apply_undersampling(
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    strategy: str = 'image_stratified',
    target_ratio: str = 'max_minority',
    random_seed: int = 42,
    verbose: bool = True,
    target_count_ratio: Optional[float] = 1.25
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Apply undersampling to balance class distribution.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Label array (n_samples,)
        df: DataFrame with columns [image_name, cell_index, label, features...]
        strategy: 'image_stratified' or 'random'
        target_ratio: 'max_minority' (use max minority class count as target)
        random_seed: Random seed for reproducibility
        verbose: Print statistics

    Returns:
        Tuple of (X_undersampled, y_undersampled, df_undersampled)
    """

    np.random.seed(random_seed)

    if verbose:
        print("="*70)
        print("APPLYING UNDERSAMPLING")
        print("="*70)
        print(f"Strategy: {strategy}")
        print(f"Target ratio: {target_ratio}")
        print(f"Random seed: {random_seed}")
        print(f"\nOriginal class distribution:")
        print_class_distribution(y)

    # Calculate target count
    class_counts = Counter(y)
    minority_classes = [1, 2, 3]  # ball, bat, stump
    # target_count = 3*max(class_counts[cls] for cls in minority_classes)
    target_count = int(target_count_ratio*max(class_counts[cls] for cls in minority_classes))

    if verbose:
        print(f"\nTarget sample count: {target_count} (from largest minority class)")

    # Get indices for each class
    class_0_indices = np.where(y == 0)[0]
    class_4_indices = np.where(y == 4)[0]
    minority_indices = np.where((y != 0) & (y != 4))[0]

    if strategy == 'image_stratified':
        # Image-stratified undersampling
        undersampled_class_0_indices = image_stratified_undersample(
            df, class_0_indices, target_count, random_seed
        )
        undersampled_class_4_indices = image_stratified_undersample(
            df, class_4_indices, target_count, random_seed
        )
    elif strategy == 'random':
        # Random undersampling
        undersampled_class_0_indices = np.random.choice(
            class_0_indices,
            size=min(target_count, len(class_0_indices)),
            replace=False
        )
        undersampled_class_4_indices = np.random.choice(
            class_4_indices,
            size=min(target_count, len(class_4_indices)),
            replace=False
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Choose 'image_stratified' or 'random'.")

    # Combine indices
    final_indices = np.concatenate([undersampled_class_0_indices, undersampled_class_4_indices, minority_indices])
    np.random.shuffle(final_indices)  # Shuffle to avoid class ordering bias

    # Apply undersampling
    X_undersampled = X[final_indices]
    y_undersampled = y[final_indices]
    df_undersampled = df.iloc[final_indices].reset_index(drop=True)

    if verbose:
        print(f"\nUndersampled class distribution:")
        print_class_distribution(y_undersampled)
        print(f"\nReduction:")
        print(f"  Original: {len(y):,} samples")
        print(f"  Undersampled: {len(y_undersampled):,} samples")
        print(f"  Removed: {len(y) - len(y_undersampled):,} samples ({(1 - len(y_undersampled)/len(y))*100:.2f}%)")
        print("="*70)

    return X_undersampled, y_undersampled, df_undersampled


def image_stratified_undersample(
    df: pd.DataFrame,
    class_0_indices: np.ndarray,
    target_count: int,
    random_seed: int
) -> np.ndarray:
    """
    Undersample Class 0 evenly across all images.

    This ensures that Class 0 samples are selected proportionally from each image,
    preserving the spatial distribution and diversity of backgrounds, lighting
    conditions, and camera angles present in the dataset.

    Args:
        df: DataFrame with image_name column
        class_0_indices: Indices of Class 0 samples in the DataFrame
        target_count: Target number of Class 0 samples to keep
        random_seed: Random seed for reproducibility

    Returns:
        Array of selected indices
    """
    np.random.seed(random_seed)

    # Get Class 0 samples with image names
    class_0_df = df.iloc[class_0_indices].copy()

    # Group by image
    images = class_0_df['image_name'].unique()
    n_images = len(images)

    # Calculate samples per image (evenly distributed)
    samples_per_image = target_count // n_images
    remaining_samples = target_count % n_images

    selected_indices = []

    for i, image_name in enumerate(images):
        # Get all Class 0 cells from this image
        image_class_0 = class_0_df[class_0_df['image_name'] == image_name]
        image_indices = image_class_0.index.values

        # Determine how many to sample from this image
        n_to_sample = samples_per_image
        if i < remaining_samples:
            n_to_sample += 1  # Distribute remainder across first images

        # Sample randomly from this image's Class 0 cells
        if len(image_indices) <= n_to_sample:
            # Keep all if fewer than target
            selected = image_indices
        else:
            # Random sample
            selected = np.random.choice(
                image_indices,
                size=n_to_sample,
                replace=False
            )

        selected_indices.extend(selected)

    return np.array(selected_indices, dtype=int)


def print_class_distribution(y: np.ndarray) -> None:
    """Print class distribution statistics."""
    class_names = {0: 'No Object', 1: 'Ball', 2: 'Bat', 3: 'Stump', 4: 'Other No Objects '}
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)

    for cls, count in zip(unique, counts):
        percentage = (count / total) * 100
        print(f"  Class {cls} ({class_names.get(cls, 'Unknown')}): {count:,} samples ({percentage:.2f}%)")


def get_undersampling_report(
    y_original: np.ndarray,
    y_undersampled: np.ndarray
) -> pd.DataFrame:
    """
    Generate detailed undersampling report.

    Args:
        y_original: Original label array before undersampling
        y_undersampled: Label array after undersampling

    Returns:
        DataFrame with columns [Class, Original_Count, Undersampled_Count,
                                Reduction, Reduction_Pct, Final_Pct]
    """
    original_counts = Counter(y_original)
    undersampled_counts = Counter(y_undersampled)

    class_names = {0: 'No Object', 1: 'Ball', 2: 'Bat', 3: 'Stump', 4: 'Other No Objects '}

    report_data = []
    for cls in sorted(original_counts.keys()):
        orig_count = original_counts[cls]
        under_count = undersampled_counts.get(cls, 0)
        reduction = orig_count - under_count
        reduction_pct = (reduction / orig_count) * 100 if orig_count > 0 else 0
        final_pct = (under_count / len(y_undersampled)) * 100 if len(y_undersampled) > 0 else 0

        report_data.append({
            'Class': f"{cls} ({class_names.get(cls, 'Unknown')})",
            'Original_Count': orig_count,
            'Undersampled_Count': under_count,
            'Reduction': reduction,
            'Reduction_Pct': f"{reduction_pct:.2f}%",
            'Final_Pct': f"{final_pct:.2f}%"
        })

    return pd.DataFrame(report_data)


def main():
    """
    Standalone test script for undersampling functionality.
    """
    print("Testing undersampling module...")
    print("="*70)

    # Create synthetic test data
    print("\n1. Creating synthetic test data...")
    n_class_0 = 1000
    n_class_1 = 50
    n_class_2 = 50
    n_class_3 = 70  # Max minority

    y = np.array([0]*n_class_0 + [1]*n_class_1 + [2]*n_class_2 + [3]*n_class_3)
    X = np.random.randn(len(y), 88)

    # Create synthetic image names
    image_names = []
    for i in range(10):
        image_names.extend([f"image_{i}.jpg"] * (len(y) // 10))
    image_names.extend([f"image_0.jpg"] * (len(y) % 10))

    df = pd.DataFrame({
        'image_name': image_names[:len(y)],
        'cell_index': np.arange(len(y)) % 64,
        'label': y
    })

    print(f"Created {len(y)} samples across 10 images")

    # Test random undersampling
    print("\n2. Testing random undersampling...")
    X_under, y_under, df_under = apply_undersampling(
        X, y, df,
        strategy='random',
        random_seed=42,
        verbose=True
    )

    # Test image-stratified undersampling
    print("\n3. Testing image-stratified undersampling...")
    X_under, y_under, df_under = apply_undersampling(
        X, y, df,
        strategy='image_stratified',
        random_seed=42,
        verbose=True
    )

    # Generate report
    print("\n4. Generating undersampling report...")
    report = get_undersampling_report(y, y_under)
    print("\n" + report.to_string(index=False))

    print("\n" + "="*70)
    print("✓ All tests completed successfully!")


if __name__ == '__main__':
    main()
