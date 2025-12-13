"""
Binary Model Evaluation Module

Provides two evaluation modes:
1. Independent Binary Evaluation - How well does each binary model perform on its task?
2. Final Multi-Class Evaluation - How well does the combined system perform?

Independent Binary Evaluation:
- Evaluates Ball model on binary task (ball vs not-ball)
- Evaluates Bat model on binary task (bat vs not-bat)
- Evaluates Stump model on binary task (stump vs not-stump)
- Metrics: Accuracy, Precision, Recall, F1, AUC-ROC per model

Final Multi-Class Evaluation:
- After conflict resolution, compare predictions to ground truth
- Same metrics as multi-class baseline
- Directly comparable to evaluate.py output

Usage:
    from binary_evaluate import evaluate_binary_models_independently, evaluate_final_predictions

    # Independent evaluation
    evaluate_binary_models_independently(models, X_val, y_ball, y_bat, y_stump)

    # Final evaluation
    evaluate_final_predictions(y_pred_final, y_test_multiclass)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


def evaluate_binary_models_independently(model_data, X_val, y_ball, y_bat, y_stump,
                                        thresholds=None, output_dir='outputs/binary_models',
                                        verbose=True):
    """
    Evaluate each binary model independently on its binary task.

    Args:
        model_data: Dictionary returned by train_three_binary_models() or load_binary_models() containing:
                   - 'models': Dictionary with keys 'ball', 'bat', 'stump'
                   - 'feature_indices': Feature indices per model (optional)
                   - 'scalers': StandardScalers per model (optional, if PCA used)
                   - 'pca_models': PCA models per model (optional, if PCA used)
                   OR just a dictionary with model keys (backward compatibility)
        X_val: Validation features (N, 2343)
        y_ball: Binary ball labels (N,)
        y_bat: Binary bat labels (N,)
        y_stump: Binary stump labels (N,)
        thresholds: Decision thresholds (optional)
        output_dir: Where to save results
        verbose: Print detailed metrics

    Saves:
        binary_metrics_independent.txt - Detailed metrics per binary model
        binary_confusion_matrices.png - Confusion matrices for all 3 models
    """
    # Handle both new format (dict with 'models' key) and old format (direct model dict)
    if isinstance(model_data, dict) and 'models' in model_data:
        models = model_data['models']
        feature_indices = model_data.get('feature_indices', {})
        scalers = model_data.get('scalers')
        pca_models = model_data.get('pca_models')
    else:
        # Backward compatibility: assume model_data is just the models dict
        models = model_data
        feature_indices = {}
        scalers = None
        pca_models = None

    if thresholds is None:
        thresholds = {'ball': 0.5, 'bat': 0.5, 'stump': 0.5}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    if verbose:
        print(f"\n{'='*70}")
        print(f"INDEPENDENT BINARY MODEL EVALUATION")
        print(f"{'='*70}")
        if feature_indices:
            print(f"Feature Selection: ENABLED")
        if scalers and pca_models:
            print(f"PCA: ENABLED")

    # Open output file
    output_file = output_dir / 'binary_metrics_independent.txt'
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("INDEPENDENT BINARY MODEL EVALUATION\n")
        f.write("="*70 + "\n\n")

        # Evaluate each binary model
        for class_name, y_true in [('ball', y_ball), ('bat', y_bat), ('stump', y_stump)]:
            model = models[class_name]
            threshold = thresholds[class_name]

            # Apply feature selection
            if feature_indices and class_name in feature_indices:
                X_class = X_val[:, feature_indices[class_name]]
            else:
                X_class = X_val

            # Apply PCA if used during training
            if pca_models and class_name in pca_models:
                X_class = scalers[class_name].transform(X_class)
                X_class = pca_models[class_name].transform(X_class)

            # Get predictions
            y_proba = model.predict_proba(X_class)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)

            # Calculate metrics
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            # AUC-ROC (only if both classes present)
            if len(np.unique(y_true)) > 1:
                auc = roc_auc_score(y_true, y_proba)
            else:
                auc = 0.0

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            # Store results
            results[class_name] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'auc_roc': auc,
                'confusion_matrix': cm,
                'threshold': threshold
            }

            # Write to file
            f.write(f"\n{'-'*70}\n")
            f.write(f"{class_name.upper()} Binary Model (threshold={threshold})\n")
            f.write(f"{'-'*70}\n\n")

            f.write(f"Metrics:\n")
            f.write(f"  Accuracy:  {acc:.4f}\n")
            f.write(f"  Precision: {prec:.4f}\n")
            f.write(f"  Recall:    {rec:.4f}\n")
            f.write(f"  F1-Score:  {f1:.4f}\n")
            f.write(f"  AUC-ROC:   {auc:.4f}\n\n")

            f.write(f"Confusion Matrix:\n")
            f.write(f"  [[TN={cm[0,0]:4d}  FP={cm[0,1]:4d}]\n")
            f.write(f"   [FN={cm[1,0]:4d}  TP={cm[1,1]:4d}]]\n\n")

            f.write(f"Class Distribution:\n")
            f.write(f"  Negative (0): {np.sum(y_true == 0)} samples\n")
            f.write(f"  Positive (1): {np.sum(y_true == 1)} samples\n\n")

            # Classification report
            f.write(f"Classification Report:\n")
            report = classification_report(y_true, y_pred,
                                          target_names=[f'Not-{class_name}', class_name],
                                          digits=4)
            f.write(report + "\n")

            if verbose:
                print(f"\n{class_name.upper()} Model:")
                print(f"  Accuracy:  {acc:.4f}")
                print(f"  Precision: {prec:.4f}")
                print(f"  Recall:    {rec:.4f}")
                print(f"  F1-Score:  {f1:.4f}")
                print(f"  AUC-ROC:   {auc:.4f}")

        f.write("\n" + "="*70 + "\n")

    if verbose:
        print(f"\nSaved independent binary metrics: {output_file}")

    # Plot confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, class_name in enumerate(['ball', 'bat', 'stump']):
        cm = results[class_name]['confusion_matrix']
        threshold = results[class_name]['threshold']

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=[f'Not-{class_name}', class_name],
                   yticklabels=[f'Not-{class_name}', class_name])

        axes[idx].set_title(f'{class_name.capitalize()} Model\n(threshold={threshold:.2f})')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')

    plt.tight_layout()
    cm_file = output_dir / 'binary_confusion_matrices.png'
    plt.savefig(cm_file, dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"Saved confusion matrices: {cm_file}")
        print(f"{'='*70}\n")

    return results


def evaluate_final_predictions(y_pred_final, y_test_multiclass,
                               output_dir='outputs/binary_models', verbose=True):
    """
    Evaluate final multi-class predictions after conflict resolution.

    This evaluation is directly comparable to the multi-class baseline.

    Args:
        y_pred_final: Final predictions (N,) - 0=No Object, 1=Ball, 2=Bat, 3=Stump
        y_test_multiclass: Ground truth (N,) - 0, 1, 2, 3
        output_dir: Where to save results
        verbose: Print detailed metrics

    Saves:
        binary_metrics_final.txt - Overall and per-class metrics
        binary_confusion_matrix_final.png - 4-class confusion matrix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*70}")
        print(f"FINAL MULTI-CLASS EVALUATION (After Conflict Resolution)")
        print(f"{'='*70}")

    # Calculate overall metrics
    acc = accuracy_score(y_test_multiclass, y_pred_final)

    # Per-class metrics (use zero_division to handle missing classes)
    prec_per_class = precision_score(y_test_multiclass, y_pred_final, average=None, zero_division=0)
    rec_per_class = recall_score(y_test_multiclass, y_pred_final, average=None, zero_division=0)
    f1_per_class = f1_score(y_test_multiclass, y_pred_final, average=None, zero_division=0)

    # Macro averages
    prec_macro = precision_score(y_test_multiclass, y_pred_final, average='macro', zero_division=0)
    rec_macro = recall_score(y_test_multiclass, y_pred_final, average='macro', zero_division=0)
    f1_macro = f1_score(y_test_multiclass, y_pred_final, average='macro', zero_division=0)

    # Weighted averages
    prec_weighted = precision_score(y_test_multiclass, y_pred_final, average='weighted', zero_division=0)
    rec_weighted = recall_score(y_test_multiclass, y_pred_final, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test_multiclass, y_pred_final, average='weighted', zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_test_multiclass, y_pred_final)

    # Write to file
    output_file = output_dir / 'binary_metrics_final.txt'
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("FINAL MULTI-CLASS EVALUATION (After Conflict Resolution)\n")
        f.write("="*70 + "\n\n")

        f.write(f"Overall Metrics:\n")
        f.write(f"  Accuracy:         {acc:.4f}\n\n")

        f.write(f"Macro Averages:\n")
        f.write(f"  Precision:        {prec_macro:.4f}\n")
        f.write(f"  Recall:           {rec_macro:.4f}\n")
        f.write(f"  F1-Score:         {f1_macro:.4f}\n\n")

        f.write(f"Weighted Averages:\n")
        f.write(f"  Precision:        {prec_weighted:.4f}\n")
        f.write(f"  Recall:           {rec_weighted:.4f}\n")
        f.write(f"  F1-Score:         {f1_weighted:.4f}\n\n")

        f.write(f"{'-'*70}\n")
        f.write(f"PER-CLASS METRICS\n")
        f.write(f"{'-'*70}\n\n")

        class_names = ['No Object', 'Ball', 'Bat', 'Stump']
        for class_id, class_name in enumerate(class_names):
            support = np.sum(y_test_multiclass == class_id)
            predicted = np.sum(y_pred_final == class_id)

            f.write(f"\nClass {class_id} ({class_name}):\n")
            f.write(f"  Precision:  {prec_per_class[class_id]:.4f}\n")
            f.write(f"  Recall:     {rec_per_class[class_id]:.4f}\n")
            f.write(f"  F1-Score:   {f1_per_class[class_id]:.4f}\n")
            f.write(f"  Support:    {support}\n")
            f.write(f"  Predicted:  {predicted}\n")

        f.write(f"\n{'-'*70}\n")
        f.write(f"CONFUSION MATRIX\n")
        f.write(f"{'-'*70}\n\n")

        # Write confusion matrix
        f.write("     Predicted:\n")
        f.write("          No Obj    Ball      Bat     Stump\n")
        f.write("Actual:\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name:10s}")
            for j in range(4):
                f.write(f" {cm[i,j]:6d}")
            f.write("\n")

        f.write("\n" + "="*70 + "\n\n")

        # Classification report
        f.write("DETAILED CLASSIFICATION REPORT\n")
        f.write("="*70 + "\n\n")
        report = classification_report(y_test_multiclass, y_pred_final,
                                      target_names=class_names, digits=4)
        f.write(report)

        f.write("\n" + "="*70 + "\n")

    if verbose:
        print(f"\nOverall Accuracy: {acc:.4f}")
        print(f"F1-Macro:         {f1_macro:.4f}")
        print(f"F1-Weighted:      {f1_weighted:.4f}\n")

        print("Per-Class Metrics:")
        print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print("-" * 60)
        for class_id, class_name in enumerate(class_names):
            support = np.sum(y_test_multiclass == class_id)
            print(f"{class_name:<15} {prec_per_class[class_id]:>10.4f} {rec_per_class[class_id]:>10.4f} "
                  f"{f1_per_class[class_id]:>10.4f} {support:>10d}")

        print(f"\nSaved final metrics: {output_file}")

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names)
    plt.title('Final Multi-Class Confusion Matrix\n(After Binary Model Conflict Resolution)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()

    cm_file = output_dir / 'binary_confusion_matrix_final.png'
    plt.savefig(cm_file, dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"Saved confusion matrix: {cm_file}")
        print(f"{'='*70}\n")

    # Return results dictionary
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


def analyze_conflict_rate(model_data, X_test, y_test_multiclass, thresholds=None, verbose=True):
    """
    Analyze how often conflicts occur and identify problematic cells.

    Args:
        model_data: Dictionary returned by train_three_binary_models() or load_binary_models() containing:
                   - 'models': Dictionary with keys 'ball', 'bat', 'stump'
                   - 'feature_indices': Feature indices per model (optional)
                   - 'scalers': StandardScalers per model (optional, if PCA used)
                   - 'pca_models': PCA models per model (optional, if PCA used)
                   OR just a dictionary with model keys (backward compatibility)
        X_test: Test features (N, 2343)
        y_test_multiclass: Ground truth (N,)
        thresholds: Decision thresholds (optional)
        verbose: Print analysis

    Returns:
        conflict_analysis: Dictionary with conflict statistics
    """
    # Handle both new format (dict with 'models' key) and old format (direct model dict)
    if isinstance(model_data, dict) and 'models' in model_data:
        models = model_data['models']
        feature_indices = model_data.get('feature_indices', {})
        scalers = model_data.get('scalers')
        pca_models = model_data.get('pca_models')
    else:
        # Backward compatibility: assume model_data is just the models dict
        models = model_data
        feature_indices = {}
        scalers = None
        pca_models = None

    if thresholds is None:
        thresholds = {'ball': 0.5, 'bat': 0.5, 'stump': 0.5}

    # Get binary predictions with feature selection and PCA
    probabilities = {}
    for class_name in ['ball', 'bat', 'stump']:
        # Apply feature selection
        if feature_indices and class_name in feature_indices:
            X_class = X_test[:, feature_indices[class_name]]
        else:
            X_class = X_test

        # Apply PCA if used during training
        if pca_models and class_name in pca_models:
            X_class = scalers[class_name].transform(X_class)
            X_class = pca_models[class_name].transform(X_class)

        # Get probability predictions
        probabilities[class_name] = models[class_name].predict_proba(X_class)[:, 1]

    ball_proba = probabilities['ball']
    bat_proba = probabilities['bat']
    stump_proba = probabilities['stump']

    ball_pred = (ball_proba >= thresholds['ball']).astype(int)
    bat_pred = (bat_proba >= thresholds['bat']).astype(int)
    stump_pred = (stump_proba >= thresholds['stump']).astype(int)

    # Count predictions per cell
    sum_pred = ball_pred + bat_pred + stump_pred

    n_samples = len(X_test)
    n_no_pred = np.sum(sum_pred == 0)
    n_single = np.sum(sum_pred == 1)
    n_double = np.sum(sum_pred == 2)
    n_triple = np.sum(sum_pred == 3)

    if verbose:
        print(f"\n{'='*70}")
        print(f"CONFLICT ANALYSIS")
        print(f"{'='*70}")

        print(f"\nPrediction Patterns:")
        print(f"  No predictions:       {n_no_pred:5d} ({100*n_no_pred/n_samples:5.2f}%)")
        print(f"  Single prediction:    {n_single:5d} ({100*n_single/n_samples:5.2f}%)")
        print(f"  Double predictions:   {n_double:5d} ({100*n_double/n_samples:5.2f}%)")
        print(f"  Triple predictions:   {n_triple:5d} ({100*n_triple/n_samples:5.2f}%)")
        print(f"  Total samples:        {n_samples:5d}")

        conflict_rate = 100 * (n_double + n_triple) / n_samples
        print(f"\n  Overall Conflict Rate: {conflict_rate:.2f}%")

        if conflict_rate > 10:
            print(f"  ⚠ Warning: High conflict rate!")
            print(f"    Consider adjusting thresholds to reduce conflicts")

        # Analyze conflicts by ground truth class
        print(f"\n{'-'*70}")
        print(f"Conflicts by Ground Truth Class:")
        print(f"{'-'*70}")

        for class_id, class_name in [(0, 'No Object'), (1, 'Ball'), (2, 'Bat'), (3, 'Stump')]:
            mask = (y_test_multiclass == class_id)
            n_class = np.sum(mask)

            if n_class == 0:
                continue

            conflicts_in_class = np.sum(sum_pred[mask] >= 2)
            print(f"\n{class_name} (Class {class_id}):")
            print(f"  Samples:   {n_class}")
            print(f"  Conflicts: {conflicts_in_class} ({100*conflicts_in_class/n_class:.2f}%)")

        print(f"\n{'='*70}\n")

    return {
        'n_no_prediction': n_no_pred,
        'n_single_prediction': n_single,
        'n_double_prediction': n_double,
        'n_triple_prediction': n_triple,
        'conflict_rate': (n_double + n_triple) / n_samples
    }


# Test function
if __name__ == '__main__':
    """
    Test binary evaluation module with synthetic data.
    """
    print("="*70)
    print("BINARY EVALUATION MODULE TEST")
    print("="*70)

    np.random.seed(42)

    # Create synthetic data
    n_samples = 1000
    n_features = 2343

    X_test = np.random.randn(n_samples, n_features).astype(np.float32)

    # Create binary ground truth
    y_ball = (np.random.rand(n_samples) < 0.05).astype(int)
    y_bat = (np.random.rand(n_samples) < 0.10).astype(int)
    y_stump = (np.random.rand(n_samples) < 0.08).astype(int)

    # Create multi-class ground truth (for final evaluation)
    y_multiclass = np.zeros(n_samples, dtype=int)
    y_multiclass[y_ball == 1] = 1
    y_multiclass[y_bat == 1] = 2
    y_multiclass[y_stump == 1] = 3

    # Create mock models
    class MockModel:
        def __init__(self, y_true):
            self.y_true = y_true

        def predict_proba(self, X):
            n = len(X)
            # Generate probabilities correlated with ground truth
            probas = np.random.rand(n) * 0.5  # Base: 0-0.5
            probas[self.y_true[:n] == 1] += 0.4  # Boost positives
            probas = np.clip(probas, 0, 1)
            return np.column_stack([1 - probas, probas])

    models = {
        'ball': MockModel(y_ball),
        'bat': MockModel(y_bat),
        'stump': MockModel(y_stump)
    }

    # Test 1: Independent evaluation
    print("\nTest 1: Independent Binary Evaluation")
    print("-" * 70)
    evaluate_binary_models_independently(models, X_test, y_ball, y_bat, y_stump,
                                        output_dir='test_outputs', verbose=True)

    # Test 2: Generate final predictions
    from binary_predict import predict_with_binary_models
    y_pred = predict_with_binary_models(models, X_test, verbose=False)

    # Test 3: Final evaluation
    print("\n\nTest 2: Final Multi-Class Evaluation")
    print("-" * 70)
    evaluate_final_predictions(y_pred, y_multiclass, output_dir='test_outputs', verbose=True)

    # Test 4: Conflict analysis
    print("\n\nTest 3: Conflict Analysis")
    print("-" * 70)
    analyze_conflict_rate(models, X_test, y_multiclass, verbose=True)

    # Clean up
    import shutil
    shutil.rmtree('test_outputs')
    print("\nCleaned up test outputs")

    print(f"\n{'='*70}")
    print(f"ALL TESTS PASSED ✓")
    print(f"{'='*70}")
