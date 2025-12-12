"""
Standalone Model Comparison Script

Compares performance of multi-class vs binary models.

This script:
1. Loads test data
2. Loads all available trained models
3. Compares performance on same test set
4. Generates comprehensive comparison report
5. Saves report to outputs/model_comparison_report.txt

Usage:
    # Compare all available models
    python compare_all_models.py

    # Specify custom paths
    python compare_all_models.py \
        --test_features outputs/binary_models/X_val.npy \
        --test_labels outputs/binary_models/y_val_multiclass.npy \
        --multiclass_model models/cricket_model.pkl \
        --binary_models_dir models/binary_models \
        --output outputs/model_comparison_report.txt
"""

import numpy as np
import argparse
from pathlib import Path
import sys
import os

# Add src directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'models'))

from compare_models import (
    load_all_available_models,
    compare_all_models,
    generate_comparison_report
)


def main():
    """Main comparison script."""
    parser = argparse.ArgumentParser(
        description='Compare Multi-Class vs Binary Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    # Compare all models with default paths (on validation set)
    python compare_all_models.py

    # Specify custom paths
    python compare_all_models.py \\
        --test_features outputs/binary_models/X_val.npy \\
        --test_labels outputs/binary_models/y_val_multiclass.npy \\
        --multiclass_model models/cricket_model.pkl \\
        --binary_models_dir models/binary_models

Output:
    Generates detailed comparison report showing:
    - Overall metrics (accuracy, F1-macro, F1-weighted)
    - Per-class metrics (precision, recall, F1 for each class)
    - Best model recommendations
    - Improvement analysis
        """
    )

    parser.add_argument('--test_features', type=str,
                       default='outputs/binary_models/X_val.npy',
                       help='Path to validation features .npy file')

    parser.add_argument('--test_labels', type=str,
                       default='outputs/binary_models/y_val_multiclass.npy',
                       help='Path to validation labels .npy file (multi-class format)')

    parser.add_argument('--multiclass_model', type=str,
                       default='models/cricket_model.pkl',
                       help='Path to multi-class model')

    parser.add_argument('--binary_models_dir', type=str,
                       default='models/binary_models',
                       help='Directory containing binary models')

    parser.add_argument('--output', type=str,
                       default='outputs/model_comparison_report.txt',
                       help='Path to save comparison report')

    parser.add_argument('--thresholds', type=str,
                       default='0.5,0.5,0.5',
                       help='Thresholds for ball,bat,stump (default: 0.5,0.5,0.5)')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"MODEL COMPARISON - Cricket Object Detection")
    print(f"{'='*70}\n")

    # Parse thresholds
    threshold_values = [float(x) for x in args.thresholds.split(',')]
    thresholds = {
        'ball': threshold_values[0],
        'bat': threshold_values[1],
        'stump': threshold_values[2]
    }

    print(f"Configuration:")
    print(f"  Test Features:      {args.test_features}")
    print(f"  Test Labels:        {args.test_labels}")
    print(f"  Multi-Class Model:  {args.multiclass_model}")
    print(f"  Binary Models Dir:  {args.binary_models_dir}")
    print(f"  Thresholds:         Ball={thresholds['ball']:.2f}, Bat={thresholds['bat']:.2f}, Stump={thresholds['stump']:.2f}")
    print(f"  Output Report:      {args.output}\n")

    # Load validation data
    print(f"{'-'*70}")
    print(f"Loading Validation Data")
    print(f"{'-'*70}\n")

    X_test_path = Path(args.test_features)
    y_test_path = Path(args.test_labels)

    if not X_test_path.exists():
        print(f"ERROR: Validation features not found: {X_test_path}")
        print(f"\nPlease run feature extraction first:")
        print(f"  python main_pipeline_binary.py --mode extract")
        return 1

    if not y_test_path.exists():
        print(f"ERROR: Validation labels not found: {y_test_path}")
        print(f"\nPlease run feature extraction first:")
        print(f"  python main_pipeline_binary.py --mode extract")
        return 1

    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)

    print(f"✓ Loaded validation data:")
    print(f"  Features: {X_test.shape}")
    print(f"  Labels:   {y_test.shape}")

    # Load models
    models = load_all_available_models(
        multiclass_model_path=args.multiclass_model,
        binary_rf_dir=args.binary_models_dir,
        binary_xgb_dir=args.binary_models_dir,
        verbose=True
    )

    if len(models) == 0:
        print("\nERROR: No models found!")
        print("\nPlease train models first:")
        print("  Multi-Class: python main_pipeline.py --mode train")
        print("  Binary:      python main_pipeline_binary.py --mode train --algorithm both")
        return 1

    # Compare models
    results = compare_all_models(models, X_test, y_test, thresholds=thresholds, verbose=True)

    # Generate report
    print(f"\n{'-'*70}")
    print(f"Generating Comparison Report")
    print(f"{'-'*70}\n")

    report_path = generate_comparison_report(results, args.output, verbose=True)

    # Print summary
    print(f"\n{'='*70}")
    print(f"COMPARISON COMPLETE")
    print(f"{'='*70}\n")

    print(f"Summary:")
    model_names = {
        'multiclass_rf': 'Multi-Class RF',
        'binary_rf': 'Binary RF',
        'binary_xgb': 'Binary XGBoost'
    }

    print(f"\n{'Model':<20s} {'Accuracy':>12s} {'F1-Macro':>12s} {'F1-Weighted':>12s}")
    print(f"{'-'*60}")
    for key, metrics in results.items():
        print(f"{model_names[key]:<20s} {metrics['accuracy']:>12.4f} "
              f"{metrics['f1_macro']:>12.4f} {metrics['f1_weighted']:>12.4f}")

    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['f1_macro'])
    print(f"\n{'='*70}")
    print(f"RECOMMENDED MODEL: {model_names[best_model[0]]}")
    print(f"  F1-Macro Score: {best_model[1]['f1_macro']:.4f}")
    print(f"{'='*70}\n")

    print(f"Detailed report saved to: {report_path}\n")

    return 0


if __name__ == '__main__':
    exit(main())
