"""
Model Comparison Framework

Compares performance of multi-class vs binary models.

Supports comparison of:
1. Multi-Class Random Forest (baseline)
2. Binary Random Forest (3 models)
3. Binary XGBoost (3 models)

Generates comprehensive comparison reports showing:
- Overall metrics (accuracy, F1-macro, F1-weighted)
- Per-class metrics (precision, recall, F1 for each class)
- Best model recommendations by metric
- Improvement analysis

Usage:
    from compare_models import load_all_models, compare_all_models

    # Load models
    models = load_all_models(
        multiclass_model_path='models/cricket_model.pkl',
        binary_rf_dir='models/binary_models',
        binary_xgb_dir='models/binary_models'
    )

    # Compare
    results = compare_all_models(models, X_test, y_test)
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.binary_predict import predict_with_binary_models


def load_all_available_models(multiclass_model_path=None,
                              binary_rf_dir=None,
                              binary_xgb_dir=None,
                              verbose=True):
    """
    Load all available models for comparison.

    Args:
        multiclass_model_path: Path to multi-class model (e.g., 'models/cricket_model.pkl')
        binary_rf_dir: Directory containing binary RF models (e.g., 'models/binary_models')
        binary_xgb_dir: Directory containing binary XGBoost models
        verbose: Print loading status

    Returns:
        models: Dictionary with available models
                Keys: 'multiclass_rf', 'binary_rf', 'binary_xgb'
    """
    models = {}

    if verbose:
        print(f"\n{'='*70}")
        print(f"LOADING MODELS FOR COMPARISON")
        print(f"{'='*70}\n")

    # Load multi-class Random Forest
    if multiclass_model_path and Path(multiclass_model_path).exists():
        with open(multiclass_model_path, 'rb') as f:
            models['multiclass_rf'] = pickle.load(f)
        if verbose:
            print(f"✓ Loaded Multi-Class Random Forest: {multiclass_model_path}")
    else:
        if verbose:
            print(f"✗ Multi-Class Random Forest not found: {multiclass_model_path}")

    # Load Binary Random Forest
    if binary_rf_dir:
        binary_rf_dir = Path(binary_rf_dir)
        ball_path = binary_rf_dir / 'binary_random_forest_ball.pkl'
        bat_path = binary_rf_dir / 'binary_random_forest_bat.pkl'
        stump_path = binary_rf_dir / 'binary_random_forest_stump.pkl'

        if all(p.exists() for p in [ball_path, bat_path, stump_path]):
            models['binary_rf'] = {}
            with open(ball_path, 'rb') as f:
                models['binary_rf']['ball'] = pickle.load(f)
            with open(bat_path, 'rb') as f:
                models['binary_rf']['bat'] = pickle.load(f)
            with open(stump_path, 'rb') as f:
                models['binary_rf']['stump'] = pickle.load(f)
            if verbose:
                print(f"✓ Loaded Binary Random Forest: {binary_rf_dir}")
        else:
            if verbose:
                print(f"✗ Binary Random Forest not found: {binary_rf_dir}")

    # Load Binary XGBoost
    if binary_xgb_dir:
        binary_xgb_dir = Path(binary_xgb_dir)
        ball_path = binary_xgb_dir / 'binary_xgboost_ball.pkl'
        bat_path = binary_xgb_dir / 'binary_xgboost_bat.pkl'
        stump_path = binary_xgb_dir / 'binary_xgboost_stump.pkl'

        if all(p.exists() for p in [ball_path, bat_path, stump_path]):
            models['binary_xgb'] = {}
            with open(ball_path, 'rb') as f:
                models['binary_xgb']['ball'] = pickle.load(f)
            with open(bat_path, 'rb') as f:
                models['binary_xgb']['bat'] = pickle.load(f)
            with open(stump_path, 'rb') as f:
                models['binary_xgb']['stump'] = pickle.load(f)
            if verbose:
                print(f"✓ Loaded Binary XGBoost: {binary_xgb_dir}")
        else:
            if verbose:
                print(f"✗ Binary XGBoost not found: {binary_xgb_dir}")

    if verbose:
        print(f"\n{'-'*70}")
        print(f"Total models loaded: {len(models)}")
        print(f"{'-'*70}\n")

    return models


def compare_all_models(models, X_test, y_test, thresholds=None, verbose=True):
    """
    Compare all available models on the same test set.

    Args:
        models: Dictionary from load_all_available_models()
        X_test: Test features (N, 2343)
        y_test: Test labels (N,) - multi-class labels (0,1,2,3)
        thresholds: Optional thresholds for binary models
        verbose: Print comparison

    Returns:
        results: Dictionary with results for each model
    """
    if thresholds is None:
        thresholds = {'ball': 0.5, 'bat': 0.5, 'stump': 0.5}

    if verbose:
        print(f"\n{'='*70}")
        print(f"COMPARING ALL MODELS ON TEST SET")
        print(f"{'='*70}")
        print(f"Test Samples: {len(y_test)}")
        print(f"Features: {X_test.shape[1]}\n")

    results = {}

    # Evaluate Multi-Class Random Forest
    if 'multiclass_rf' in models:
        if verbose:
            print(f"\n{'-'*70}")
            print(f"Evaluating: Multi-Class Random Forest (Baseline)")
            print(f"{'-'*70}\n")

        y_pred = models['multiclass_rf'].predict(X_test)
        results['multiclass_rf'] = evaluate_predictions(y_test, y_pred, 'Multi-Class RF', verbose)

    # Evaluate Binary Random Forest
    if 'binary_rf' in models:
        if verbose:
            print(f"\n{'-'*70}")
            print(f"Evaluating: Binary Random Forest (3 models)")
            print(f"{'-'*70}\n")

        y_pred = predict_with_binary_models(models['binary_rf'], X_test, thresholds, verbose=False)
        results['binary_rf'] = evaluate_predictions(y_test, y_pred, 'Binary RF', verbose)

    # Evaluate Binary XGBoost
    if 'binary_xgb' in models:
        if verbose:
            print(f"\n{'-'*70}")
            print(f"Evaluating: Binary XGBoost (3 models)")
            print(f"{'-'*70}\n")

        y_pred = predict_with_binary_models(models['binary_xgb'], X_test, thresholds, verbose=False)
        results['binary_xgb'] = evaluate_predictions(y_test, y_pred, 'Binary XGBoost', verbose)

    if verbose:
        print(f"\n{'='*70}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*70}\n")

    return results


def evaluate_predictions(y_true, y_pred, model_name, verbose=True):
    """
    Evaluate predictions and return metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        model_name: Name of the model
        verbose: Print results

    Returns:
        metrics: Dictionary with all metrics
    """
    # Overall metrics
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    prec_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Per-class metrics
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if verbose:
        print(f"{model_name} Results:")
        print(f"  Overall Accuracy:  {acc:.4f}")
        print(f"  F1-Macro:          {f1_macro:.4f}")
        print(f"  F1-Weighted:       {f1_weighted:.4f}\n")

        print(f"  Per-Class Metrics:")
        class_names = ['No Object', 'Ball', 'Bat', 'Stump']
        for i, name in enumerate(class_names):
            support = np.sum(y_true == i)
            print(f"    {name:12s}: Prec={prec_per_class[i]:.4f}, Rec={rec_per_class[i]:.4f}, "
                  f"F1={f1_per_class[i]:.4f}, Support={support}")

    return {
        'accuracy': acc,
        'precision_macro': prec_macro,
        'recall_macro': rec_macro,
        'f1_macro': f1_macro,
        'precision_weighted': prec_weighted,
        'recall_weighted': rec_weighted,
        'f1_weighted': f1_weighted,
        'precision_per_class': prec_per_class,
        'recall_per_class': rec_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm
    }


def generate_comparison_report(results, output_path, verbose=True):
    """
    Generate detailed text report comparing all models.

    Args:
        results: Dictionary from compare_all_models()
        output_path: Where to save report
        verbose: Print report location

    Saves:
        Detailed comparison report as text file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE MODEL COMPARISON - Cricket Object Detection\n")
        f.write("=" * 80 + "\n\n")

        f.write("Models Compared:\n")
        model_names = {
            'multiclass_rf': '1. Multi-Class Random Forest (Baseline)',
            'binary_rf': '2. Binary Random Forest (3 models: Ball, Bat, Stump)',
            'binary_xgb': '3. Binary XGBoost (3 models: Ball, Bat, Stump)'
        }
        for key in results.keys():
            f.write(f"  {model_names[key]}\n")
        f.write("\n")

        # Overall Metrics Comparison
        f.write("-" * 80 + "\n")
        f.write("OVERALL METRICS\n")
        f.write("-" * 80 + "\n\n")

        f.write(f"{'Metric':<20s}")
        for key in results.keys():
            short_name = {'multiclass_rf': 'MC-RF', 'binary_rf': 'Bin-RF', 'binary_xgb': 'Bin-XGB'}[key]
            f.write(f"{short_name:>15s}")
        f.write("\n" + "-" * 80 + "\n")

        metrics = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
        metric_names = ['Accuracy', 'F1-Macro', 'F1-Weighted', 'Precision-Macro', 'Recall-Macro']

        for metric, name in zip(metrics, metric_names):
            f.write(f"{name:<20s}")
            for key in results.keys():
                f.write(f"{results[key][metric]:>15.4f}")
            f.write("\n")

        f.write("\n")

        # Per-Class Metrics
        class_names = ['No Object', 'Ball', 'Bat', 'Stump']
        for class_id, class_name in enumerate(class_names):
            f.write("-" * 80 + "\n")
            f.write(f"PER-CLASS METRICS - {class_name.upper()} (Class {class_id})\n")
            f.write("-" * 80 + "\n\n")

            f.write(f"{'Metric':<20s}")
            for key in results.keys():
                short_name = {'multiclass_rf': 'MC-RF', 'binary_rf': 'Bin-RF', 'binary_xgb': 'Bin-XGB'}[key]
                f.write(f"{short_name:>15s}")
            f.write("\n" + "-" * 80 + "\n")

            # Precision
            f.write(f"{'Precision':<20s}")
            for key in results.keys():
                f.write(f"{results[key]['precision_per_class'][class_id]:>15.4f}")
            f.write("\n")

            # Recall
            f.write(f"{'Recall':<20s}")
            for key in results.keys():
                f.write(f"{results[key]['recall_per_class'][class_id]:>15.4f}")
            f.write("\n")

            # F1-Score
            f.write(f"{'F1-Score':<20s}")
            for key in results.keys():
                f.write(f"{results[key]['f1_per_class'][class_id]:>15.4f}")
            f.write("\n\n")

        # Best Models by Metric
        f.write("-" * 80 + "\n")
        f.write("BEST MODELS BY METRIC\n")
        f.write("-" * 80 + "\n\n")

        # Find best for overall metrics
        best_accuracy = max(results.items(), key=lambda x: x[1]['accuracy'])
        best_f1_macro = max(results.items(), key=lambda x: x[1]['f1_macro'])
        best_f1_weighted = max(results.items(), key=lambda x: x[1]['f1_weighted'])

        f.write(f"Best Overall Accuracy:  {model_names[best_accuracy[0]]} ({best_accuracy[1]['accuracy']:.4f})\n")
        f.write(f"Best F1-Macro:          {model_names[best_f1_macro[0]]} ({best_f1_macro[1]['f1_macro']:.4f})\n")
        f.write(f"Best F1-Weighted:       {model_names[best_f1_weighted[0]]} ({best_f1_weighted[1]['f1_weighted']:.4f})\n\n")

        # Best for per-class metrics
        for class_id, class_name in enumerate(class_names):
            best_prec = max(results.items(), key=lambda x: x[1]['precision_per_class'][class_id])
            best_rec = max(results.items(), key=lambda x: x[1]['recall_per_class'][class_id])
            best_f1 = max(results.items(), key=lambda x: x[1]['f1_per_class'][class_id])

            f.write(f"{class_name} Best Precision: {model_names[best_prec[0]]} ({best_prec[1]['precision_per_class'][class_id]:.4f})\n")
            f.write(f"{class_name} Best Recall:    {model_names[best_rec[0]]} ({best_rec[1]['recall_per_class'][class_id]:.4f})\n")
            f.write(f"{class_name} Best F1:        {model_names[best_f1[0]]} ({best_f1[1]['f1_per_class'][class_id]:.4f})\n\n")

        # Recommendation
        f.write("-" * 80 + "\n")
        f.write("RECOMMENDATION\n")
        f.write("-" * 80 + "\n\n")

        # Simple heuristic: best F1-macro
        recommended_model = model_names[best_f1_macro[0]]
        f.write(f"Based on F1-Macro score, the recommended model is:\n")
        f.write(f"  {recommended_model}\n\n")

        # Check if binary models improved over baseline
        if 'multiclass_rf' in results and 'binary_rf' in results:
            mc_f1 = results['multiclass_rf']['f1_macro']
            bin_rf_f1 = results['binary_rf']['f1_macro']
            improvement = bin_rf_f1 - mc_f1

            f.write(f"Binary RF vs Multi-Class RF:\n")
            f.write(f"  F1-Macro Improvement: {improvement:+.4f} ({100*improvement/max(mc_f1, 0.0001):+.2f}%)\n\n")

        f.write("=" * 80 + "\n")

    if verbose:
        print(f"\nComparison report saved: {output_path}")

    return output_path


# Test function
if __name__ == '__main__':
    """
    Test comparison framework with synthetic models.
    """
    print("=" * 70)
    print("MODEL COMPARISON FRAMEWORK TEST")
    print("=" * 70)

    # This is a placeholder test - in practice you would load real models
    print("\nNote: This module requires trained models to test properly.")
    print("Run main_pipeline.py and main_pipeline_binary.py first to generate models.")
    print("\nExample usage:")
    print("  from compare_models import load_all_available_models, compare_all_models")
    print("  models = load_all_available_models(...)")
    print("  results = compare_all_models(models, X_test, y_test)")

    print(f"\n{'='*70}")
    print(f"MODULE LOADED SUCCESSFULLY ✓")
    print(f"{'='*70}")
