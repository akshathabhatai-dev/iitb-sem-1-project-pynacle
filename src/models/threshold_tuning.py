"""
Threshold Tuning Module

Optimizes decision thresholds for binary classifiers to improve performance.

Default threshold (0.5) may not be optimal for imbalanced datasets.
This module finds the best threshold for each binary model by:
- Grid searching over threshold values [0.1, 0.15, ..., 0.9]
- Evaluating on validation set
- Maximizing specified metric (F1, precision, recall, etc.)

Expected Optimal Thresholds:
- Ball: Higher threshold (0.6-0.8) to reduce false positives
- Bat: Lower threshold (0.3-0.5) to increase recall
- Stump: Lower threshold (0.3-0.5) to increase recall

Usage:
    from threshold_tuning import optimize_thresholds_all_models

    # Optimize thresholds for all 3 models
    optimal_thresholds = optimize_thresholds_all_models(
        models, X_val, y_ball, y_bat, y_stump, metric='f1'
    )

    # Returns: {'ball': 0.6, 'bat': 0.4, 'stump': 0.4}
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


def optimize_threshold_single_model(model, X_val, y_val_binary, class_name,
                                   metric='f1', threshold_range=None, verbose=True):
    """
    Optimize threshold for a single binary model.

    Args:
        model: Trained binary classifier
        X_val: Validation features (N, 2343)
        y_val_binary: Binary ground truth (N,)
        class_name: Name of the class ('ball', 'bat', 'stump')
        metric: Metric to optimize ('f1', 'precision', 'recall', 'accuracy', 'balanced')
        threshold_range: List of thresholds to try (default: 0.1 to 0.9 in steps of 0.05)
        verbose: Print optimization progress

    Returns:
        best_threshold: Optimal threshold value
        best_score: Score at optimal threshold
    """
    if threshold_range is None:
        threshold_range = np.arange(0.1, 0.95, 0.05)

    if verbose:
        print(f"\n{'='*70}")
        print(f"OPTIMIZING THRESHOLD - {class_name.upper()} Model")
        print(f"{'='*70}")
        print(f"Metric: {metric}")
        print(f"Threshold Range: [{threshold_range[0]:.2f}, {threshold_range[-1]:.2f}]")
        print(f"Step Size: {threshold_range[1] - threshold_range[0]:.2f}")

    # Get probability predictions
    y_proba = model.predict_proba(X_val)[:, 1]

    # Metric function selector
    metric_funcs = {
        'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, zero_division=0),
        'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0),
        'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, zero_division=0),
        'accuracy': lambda y_true, y_pred: accuracy_score(y_true, y_pred)
    }

    if metric != 'balanced' and metric not in metric_funcs:
        raise ValueError(f"Unknown metric: {metric}. Choose from {list(metric_funcs.keys())}")
    if metric != 'balanced':
        metric_func = metric_funcs[metric]

    # Try each threshold
    scores = []
    best_score = -1
    best_threshold = 0.5

    if metric == 'balanced':
        for threshold in threshold_range:
            y_pred = (y_proba >= threshold).astype(int)

            precision = precision_score(y_val_binary, y_pred, zero_division=0)
            recall = recall_score(y_val_binary, y_pred, zero_division=0)
            f1 = f1_score(y_val_binary, y_pred, zero_division=0)

            # BALANCE SCORE (lower means metrics are closer)
            balance_err = abs(precision - recall) + abs(precision - f1) + abs(recall - f1)
            scores.append(balance_err)

            if best_score == -1 or balance_err < best_score:
                best_score = balance_err
                best_threshold = threshold
    else:
        for threshold in threshold_range:
            y_pred = (y_proba >= threshold).astype(int)
            score = metric_func(y_val_binary, y_pred)
            scores.append(score)
        if score > best_score:
            best_score = score
            best_threshold = threshold

    if verbose:
        print(f"\nOptimization Results:")
        print(f"  Best Threshold: {best_threshold:.2f}")
        print(f"  Best {metric.upper()} Score: {best_score:.4f}")

        # Show performance at default threshold (0.5)
        default_pred = (y_proba >= 0.5).astype(int)
        if metric == 'balanced':
            precision = precision_score(y_val_binary, default_pred, zero_division=0)
            recall = recall_score(y_val_binary, default_pred, zero_division=0)
            f1 = f1_score(y_val_binary, default_pred, zero_division=0)
            default_score = abs(precision - recall) + abs(precision - f1) + abs(recall - f1)
        else:
            default_score = metric_func(y_val_binary, default_pred)
        improvement = best_score - default_score

        print(f"\nComparison to Default (0.5):")
        print(f"  Default {metric.upper()}: {default_score:.4f}")
        print(f"  Improvement: {improvement:+.4f} ({100*improvement/max(default_score, 0.0001):+.2f}%)")

        # Show other metrics at best threshold
        best_pred = (y_proba >= best_threshold).astype(int)
        print(f"\nMetrics at Best Threshold ({best_threshold:.2f}):")
        print(f"  Accuracy:  {accuracy_score(y_val_binary, best_pred):.4f}")
        print(f"  Precision: {precision_score(y_val_binary, best_pred, zero_division=0):.4f}")
        print(f"  Recall:    {recall_score(y_val_binary, best_pred, zero_division=0):.4f}")
        print(f"  F1-Score:  {f1_score(y_val_binary, best_pred, zero_division=0):.4f}")

    return best_threshold, best_score, threshold_range, scores


def optimize_thresholds_all_models(models, X_val_dict, y_ball, y_bat, y_stump,
                                   metric='f1', threshold_range=None, verbose=True):
    """
    Optimize thresholds for all three binary models.

    Args:
        models: Dictionary with keys 'ball', 'bat', 'stump'
        X_val_dict: Dictionary {'ball': X_val_ball, 'bat': X_val_bat, 'stump': X_val_stump}
                   Each X_val has already been transformed with feature selection/PCA
        y_ball: Binary ball labels (N,)
        y_bat: Binary bat labels (N,)
        y_stump: Binary stump labels (N,)
        metric: Metric to optimize ('f1', 'precision', 'recall', 'accuracy')
        threshold_range: List of thresholds to try
        verbose: Print optimization progress

    Returns:
        optimal_thresholds: Dictionary {'ball': 0.6, 'bat': 0.4, 'stump': 0.4}
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"OPTIMIZING THRESHOLDS FOR ALL BINARY MODELS")
        print(f"{'='*70}")
        print(f"Optimization Metric: {metric.upper()}")

    optimal_thresholds = {}

    # Optimize Ball model
    ball_threshold, ball_score, _, _ = optimize_threshold_single_model(
        models['ball'], X_val_dict['ball'], y_ball, 'ball',
        metric=metric, threshold_range=threshold_range, verbose=verbose
    )
    optimal_thresholds['ball'] = ball_threshold

    # Optimize Bat model
    bat_threshold, bat_score, _, _ = optimize_threshold_single_model(
        models['bat'], X_val_dict['bat'], y_bat, 'bat',
        metric=metric, threshold_range=threshold_range, verbose=verbose
    )
    optimal_thresholds['bat'] = bat_threshold

    # Optimize Stump model
    stump_threshold, stump_score, _, _ = optimize_threshold_single_model(
        models['stump'], X_val_dict['stump'], y_stump, 'stump',
        metric=metric, threshold_range=threshold_range, verbose=verbose
    )
    optimal_thresholds['stump'] = stump_threshold

    if verbose:
        print(f"\n{'='*70}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"Optimal Thresholds:")
        print(f"  Ball:  {optimal_thresholds['ball']:.2f} ({metric}={ball_score:.4f})")
        print(f"  Bat:   {optimal_thresholds['bat']:.2f} ({metric}={bat_score:.4f})")
        print(f"  Stump: {optimal_thresholds['stump']:.2f} ({metric}={stump_score:.4f})")
        print(f"{'='*70}\n")

    return optimal_thresholds


def evaluate_threshold_impact(models, X_val_dict, y_ball, y_bat, y_stump,
                              threshold_range=None, output_dir='outputs/binary_models',
                              verbose=True):
    """
    Evaluate and visualize the impact of threshold on all metrics.

    Generates plots showing how precision, recall, F1 vary with threshold.

    Args:
        models: Dictionary with keys 'ball', 'bat', 'stump'
        X_val_dict: Dictionary {'ball': X_val_ball, 'bat': X_val_bat, 'stump': X_val_stump}
                   Each X_val has already been transformed with feature selection/PCA
        y_ball, y_bat, y_stump: Binary ground truth
        threshold_range: List of thresholds to evaluate
        output_dir: Where to save plots
        verbose: Print analysis

    Saves:
        threshold_impact_{class}.png for each binary model
    """
    if threshold_range is None:
        threshold_range = np.arange(0.1, 0.95, 0.05)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*70}")
        print(f"THRESHOLD IMPACT ANALYSIS")
        print(f"{'='*70}")

    for class_name, y_true in [('ball', y_ball), ('bat', y_bat), ('stump', y_stump)]:
        model = models[class_name]

        # Get probabilities
        y_proba = model.predict_proba(X_val_dict[class_name])[:, 1]

        # Evaluate metrics at each threshold
        precisions = []
        recalls = []
        f1_scores = []
        accuracies = []

        for threshold in threshold_range:
            y_pred = (y_proba >= threshold).astype(int)

            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
            accuracies.append(accuracy_score(y_true, y_pred))

        # Find optimal thresholds for each metric
        best_f1_idx = np.argmax(f1_scores)
        best_prec_idx = np.argmax(precisions)
        best_rec_idx = np.argmax(recalls)

        if verbose:
            print(f"\n{class_name.upper()} Model:")
            print(f"  Best F1 Threshold:        {threshold_range[best_f1_idx]:.2f} (F1={f1_scores[best_f1_idx]:.4f})")
            print(f"  Best Precision Threshold: {threshold_range[best_prec_idx]:.2f} (Prec={precisions[best_prec_idx]:.4f})")
            print(f"  Best Recall Threshold:    {threshold_range[best_rec_idx]:.2f} (Rec={recalls[best_rec_idx]:.4f})")

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(threshold_range, precisions, 'b-', label='Precision', linewidth=2)
        ax.plot(threshold_range, recalls, 'r-', label='Recall', linewidth=2)
        ax.plot(threshold_range, f1_scores, 'g-', label='F1-Score', linewidth=2)
        ax.plot(threshold_range, accuracies, 'orange', label='Accuracy', linewidth=2, linestyle='--')

        # Mark optimal F1 threshold
        ax.axvline(x=threshold_range[best_f1_idx], color='green', linestyle=':', alpha=0.7,
                  label=f'Best F1 Threshold ({threshold_range[best_f1_idx]:.2f})')

        # Mark default threshold
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Default (0.5)')

        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Threshold Impact on {class_name.capitalize()} Model Performance', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(threshold_range[0], threshold_range[-1])
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        plot_file = output_dir / f'threshold_impact_{class_name}.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()

        if verbose:
            print(f"  Saved plot: {plot_file}")

    if verbose:
        print(f"\n{'='*70}\n")


def plot_threshold_comparison(models, X_val_dict, y_ball, y_bat, y_stump,
                              default_thresholds, optimal_thresholds,
                              output_dir='outputs/binary_models'):
    """
    Compare performance at default vs optimal thresholds.

    Args:
        models: Dictionary with keys 'ball', 'bat', 'stump'
        X_val_dict: Dictionary {'ball': X_val_ball, 'bat': X_val_bat, 'stump': X_val_stump}
                   Each X_val has already been transformed with feature selection/PCA
        y_ball, y_bat, y_stump: Binary ground truth
        default_thresholds: Default thresholds (usually {'ball': 0.5, ...})
        optimal_thresholds: Optimized thresholds
        output_dir: Where to save plot

    Saves:
        threshold_comparison.png - Bar chart comparing default vs optimal
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate at default thresholds
    default_results = {}
    for class_name, y_true in [('ball', y_ball), ('bat', y_bat), ('stump', y_stump)]:
        y_proba = models[class_name].predict_proba(X_val_dict[class_name])[:, 1]
        y_pred = (y_proba >= default_thresholds[class_name]).astype(int)

        default_results[class_name] = {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }

    # Evaluate at optimal thresholds
    optimal_results = {}
    for class_name, y_true in [('ball', y_ball), ('bat', y_bat), ('stump', y_stump)]:
        y_proba = models[class_name].predict_proba(X_val_dict[class_name])[:, 1]
        y_pred = (y_proba >= optimal_thresholds[class_name]).astype(int)

        optimal_results[class_name] = {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['precision', 'recall', 'f1']
    metric_titles = ['Precision', 'Recall', 'F1-Score']

    for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[idx]

        classes = ['Ball', 'Bat', 'Stump']
        x = np.arange(len(classes))
        width = 0.35

        default_scores = [default_results[c.lower()][metric] for c in classes]
        optimal_scores = [optimal_results[c.lower()][metric] for c in classes]

        ax.bar(x - width/2, default_scores, width, label=f'Default ({default_thresholds["ball"]:.1f})', alpha=0.8)
        ax.bar(x + width/2, optimal_scores, width, label='Optimized', alpha=0.8)

        ax.set_xlabel('Class')
        ax.set_ylabel(title)
        ax.set_title(f'{title} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.05)

        # Add value labels on bars
        for i, (d, o) in enumerate(zip(default_scores, optimal_scores)):
            ax.text(i - width/2, d + 0.02, f'{d:.3f}', ha='center', va='bottom', fontsize=8)
            ax.text(i + width/2, o + 0.02, f'{o:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plot_file = output_dir / 'threshold_comparison.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved threshold comparison plot: {plot_file}")


# Test function
if __name__ == '__main__':
    """
    Test threshold tuning module with synthetic data.
    """
    print("="*70)
    print("THRESHOLD TUNING MODULE TEST")
    print("="*70)

    np.random.seed(42)

    # Create synthetic data
    n_samples = 1000
    n_features = 2343

    X_val = np.random.randn(n_samples, n_features).astype(np.float32)

    # Binary ground truth
    y_ball = (np.random.rand(n_samples) < 0.05).astype(int)
    y_bat = (np.random.rand(n_samples) < 0.10).astype(int)
    y_stump = (np.random.rand(n_samples) < 0.08).astype(int)

    # Create mock models
    class MockModel:
        def __init__(self, y_true):
            self.y_true = y_true

        def predict_proba(self, X):
            n = len(X)
            probas = np.random.rand(n) * 0.5
            probas[self.y_true[:n] == 1] += 0.4
            probas = np.clip(probas, 0, 1)
            return np.column_stack([1 - probas, probas])

    models = {
        'ball': MockModel(y_ball),
        'bat': MockModel(y_bat),
        'stump': MockModel(y_stump)
    }

    # Test 1: Optimize single model
    print("\nTest 1: Optimize Ball Model Threshold")
    print("-" * 70)
    optimize_threshold_single_model(models['ball'], X_val, y_ball, 'ball', verbose=True)

    # Test 2: Optimize all models
    print("\n\nTest 2: Optimize All Models")
    print("-" * 70)
    optimal_thresholds = optimize_thresholds_all_models(
        models, X_val, y_ball, y_bat, y_stump, metric='f1', verbose=True
    )

    # Test 3: Threshold impact analysis
    print("\n\nTest 3: Threshold Impact Analysis")
    print("-" * 70)
    evaluate_threshold_impact(models, X_val, y_ball, y_bat, y_stump,
                             output_dir='test_outputs', verbose=True)

    # Test 4: Comparison plot
    print("\n\nTest 4: Threshold Comparison Plot")
    print("-" * 70)
    default_thresholds = {'ball': 0.5, 'bat': 0.5, 'stump': 0.5}
    plot_threshold_comparison(models, X_val, y_ball, y_bat, y_stump,
                             default_thresholds, optimal_thresholds,
                             output_dir='test_outputs')

    # Clean up
    import shutil
    shutil.rmtree('test_outputs')
    print("\nCleaned up test outputs")

    print(f"\n{'='*70}")
    print(f"ALL TESTS PASSED ✓")
    print(f"{'='*70}")
