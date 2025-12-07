"""
PCA Analysis and Transformation Script

This script analyzes PCA dimensionality reduction on extracted features and optionally
transforms the features to their PCA representations.

Usage:
    python analyze_pca.py --input outputs/train_features.csv
    python analyze_pca.py --input outputs/train_features.csv --transform --output outputs/train_features_pca.csv
    python analyze_pca.py --input outputs/train_features.csv --variance 0.90 --transform
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import sys
import os

# Add src directory to path for config import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from config import PCA_VARIANCE_THRESHOLD, RANDOM_STATE


def analyze_pca(input_path: str, variance_threshold: float = PCA_VARIANCE_THRESHOLD,
                transform: bool = False, output_path: str = None,
                save_pca_model: bool = False, pca_model_path: str = None):
    """
    Analyze PCA dimensionality reduction and optionally transform features.

    Args:
        input_path: Path to input CSV file with features
        variance_threshold: Amount of variance to preserve (default: 0.95)
        transform: Whether to transform and save PCA features
        output_path: Path to save transformed features (if transform=True)
        save_pca_model: Whether to save the fitted PCA model
        pca_model_path: Path to save PCA model and scaler
    """
    print("="*70)
    print("PCA DIMENSIONALITY REDUCTION ANALYSIS")
    print("="*70)
    print(f"Input file: {input_path}")
    print(f"Variance threshold: {variance_threshold*100:.0f}%")
    print()

    # Load data
    print("Loading features...")
    df = pd.read_csv(input_path)

    # Identify metadata columns
    metadata_cols = ['image_name', 'cell_index', 'label']
    feature_cols = [col for col in df.columns if col not in metadata_cols]

    print(f"Total samples: {len(df)}")
    print(f"Total features: {len(feature_cols)}")
    print()

    # Count feature types
    feature_types = {
        'RGB': [col for col in feature_cols if col.startswith('rgb_')],
        'HSV': [col for col in feature_cols if col.startswith('hsv_')],
        'LBP': [col for col in feature_cols if col.startswith('lbp_')],
        'HOG': [col for col in feature_cols if col.startswith('hog_')],
        'Gabor': [col for col in feature_cols if col.startswith('gabor_')],
        'GLCM': [col for col in feature_cols if col.startswith('glcm_')]
    }

    print("Feature Breakdown:")
    print("-" * 40)
    for feature_type, features in feature_types.items():
        print(f"  {feature_type:6s}: {len(features):3d} features")
    print("-" * 40)
    print(f"  Total : {len(feature_cols):3d} features")
    print()

    # Get feature data
    X = df[feature_cols].values

    # Scale features (important for PCA!)
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit PCA
    print(f"Fitting PCA ({variance_threshold*100:.0f}% variance)...")
    pca = PCA(n_components=variance_threshold, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    n_components = pca.n_components_
    variance_explained = pca.explained_variance_ratio_.sum()

    print()
    print("="*70)
    print("PCA REDUCTION RESULTS")
    print("="*70)
    print(f"Original features:  {len(feature_cols)}")
    print(f"Reduced features:   {n_components}")
    print(f"Reduction:          {len(feature_cols)} -> {n_components} ({n_components/len(feature_cols)*100:.1f}% of original)")
    print(f"Variance retained:  {variance_explained*100:.2f}%")
    print(f"Dimensionality cut: {(1 - n_components/len(feature_cols))*100:.1f}%")
    print()

    # Show top principal components
    print("Top 20 Principal Components (variance explained):")
    print("-" * 70)
    print("  PC   Individual   Cumulative")
    print("-" * 70)
    for i in range(min(20, len(pca.explained_variance_ratio_))):
        cumulative = pca.explained_variance_ratio_[:i+1].sum()
        print(f"  {i+1:2d}     {pca.explained_variance_ratio_[i]*100:5.2f}%       {cumulative*100:5.2f}%")

    if len(pca.explained_variance_ratio_) > 20:
        print(f"  ...  (showing 20 of {len(pca.explained_variance_ratio_)} components)")

    print()

    # Show components needed for different thresholds
    print("Components needed for different variance thresholds:")
    print("-" * 70)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    for threshold in [0.80, 0.85, 0.90, 0.95, 0.99]:
        n_needed = np.argmax(cumsum >= threshold) + 1
        print(f"  {threshold*100:3.0f}% variance: {n_needed:3d} components ({n_needed/len(feature_cols)*100:5.1f}% of original)")

    print()

    # Transform and save if requested
    if transform:
        print("="*70)
        print("TRANSFORMING FEATURES TO PCA SPACE")
        print("="*70)

        # Create PCA feature names
        pca_feature_cols = [f'PC{i+1:03d}' for i in range(n_components)]

        # Create new dataframe with PCA features
        df_pca = pd.DataFrame(X_pca, columns=pca_feature_cols)

        # Add metadata columns back
        for col in metadata_cols:
            if col in df.columns:
                df_pca[col] = df[col].values

        # Reorder columns: metadata first, then features
        meta_present = [col for col in metadata_cols if col in df.columns]
        df_pca = df_pca[meta_present + pca_feature_cols]

        # Determine output path
        if output_path is None:
            input_path_obj = Path(input_path)
            output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_pca.csv")

        # Save transformed features
        df_pca.to_csv(output_path, index=False)
        print(f"Transformed features saved to: {output_path}")
        print(f"  Shape: {df_pca.shape}")
        print(f"  Columns: {list(df_pca.columns[:3])} ... {list(df_pca.columns[-3:])}")
        print()

    # Save PCA model and scaler if requested
    if save_pca_model:
        print("="*70)
        print("SAVING PCA MODEL AND SCALER")
        print("="*70)

        if pca_model_path is None:
            input_path_obj = Path(input_path)
            pca_model_path = str(input_path_obj.parent / "pca_model.pkl")

        # Save both scaler and PCA model
        model_data = {
            'scaler': scaler,
            'pca': pca,
            'n_components': n_components,
            'variance_explained': variance_explained,
            'variance_threshold': variance_threshold,
            'feature_names_in': feature_cols,
            'feature_names_out': [f'PC{i+1:03d}' for i in range(n_components)]
        }

        with open(pca_model_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"PCA model and scaler saved to: {pca_model_path}")
        print(f"  Use this model to transform new data consistently")
        print()

    print("="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

    return {
        'n_original_features': len(feature_cols),
        'n_pca_components': n_components,
        'variance_explained': variance_explained,
        'pca_model': pca,
        'scaler': scaler
    }


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Analyze PCA dimensionality reduction on feature data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python analyze_pca.py --input outputs/train_features.csv

  # Analysis with 90% variance threshold
  python analyze_pca.py --input outputs/train_features.csv --variance 0.90

  # Transform features and save
  python analyze_pca.py --input outputs/train_features.csv --transform

  # Transform with custom output path
  python analyze_pca.py --input outputs/train_features.csv --transform --output outputs/custom_pca.csv

  # Save PCA model for later use
  python analyze_pca.py --input outputs/train_features.csv --save-model --model-path models/pca.pkl
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input CSV file with features'
    )

    parser.add_argument(
        '--variance', '-v',
        type=float,
        default=PCA_VARIANCE_THRESHOLD,
        help=f'Variance threshold to preserve (default: {PCA_VARIANCE_THRESHOLD} = {PCA_VARIANCE_THRESHOLD*100:.0f}%%)'
    )

    parser.add_argument(
        '--transform', '-t',
        action='store_true',
        help='Transform features to PCA space and save'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output path for transformed features (default: input_pca.csv)'
    )

    parser.add_argument(
        '--save-model', '-s',
        action='store_true',
        help='Save fitted PCA model and scaler'
    )

    parser.add_argument(
        '--model-path', '-m',
        type=str,
        default=None,
        help='Path to save PCA model (default: same dir as input)'
    )

    args = parser.parse_args()

    # Validate input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Validate variance threshold
    if not 0 < args.variance <= 1:
        print(f"Error: Variance threshold must be between 0 and 1, got {args.variance}")
        return 1

    # Run analysis
    try:
        analyze_pca(
            input_path=args.input,
            variance_threshold=args.variance,
            transform=args.transform,
            output_path=args.output,
            save_pca_model=args.save_model,
            pca_model_path=args.model_path
        )
        return 0
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
