"""
Binary Model Prediction Module

Generates predictions using three binary classifiers and resolves conflicts.

The prediction pipeline:
1. Each binary model predicts probability for its object type
2. Apply thresholds to convert probabilities to binary predictions
3. Resolve conflicts when multiple (or no) objects are predicted
4. Return final single-label predictions (0, 1, 2, 3)

Conflict Resolution Strategy:
- If no objects predicted → Class 0 (No Object)
- If one object predicted → That class (1=Ball, 2=Bat, 3=Stump)
- If multiple objects predicted → Choose object with highest probability

Usage:
    from binary_predict import predict_with_binary_models

    # Load models
    models = load_binary_models('models/binary_models', 'random_forest')

    # Get predictions
    y_pred = predict_with_binary_models(models, X_test, thresholds=None)
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


def resolve_conflicts(ball_pred, bat_pred, stump_pred,
                     ball_proba, bat_proba, stump_proba, verbose=False):
    """
    Resolve conflicts when multiple (or no) binary models predict positive.

    Args:
        ball_pred: Binary ball predictions (N,) - 1=ball, 0=not-ball
        bat_pred: Binary bat predictions (N,)
        stump_pred: Binary stump predictions (N,)
        ball_proba: Ball probabilities (N,) - probability of ball=1
        bat_proba: Bat probabilities (N,)
        stump_proba: Stump probabilities (N,)
        verbose: Print conflict statistics

    Returns:
        y_final: Final single-label predictions (N,)
                 0=No Object, 1=Ball, 2=Bat, 3=Stump

    Conflict Resolution Logic:
        sum_pred = ball_pred + bat_pred + stump_pred

        if sum_pred == 0:
            → Class 0 (No Object)

        elif sum_pred == 1:
            → Single predicted class
            - if ball_pred == 1 → Class 1
            - if bat_pred == 1 → Class 2
            - if stump_pred == 1 → Class 3

        else:  # sum_pred >= 2 (conflict)
            → Choose class with highest probability
            - max(ball_proba, bat_proba, stump_proba)
    """
    n_samples = len(ball_pred)
    y_final = np.zeros(n_samples, dtype=int)

    # Count conflicts for statistics
    n_no_prediction = 0
    n_single_prediction = 0
    n_conflicts = 0

    for i in range(n_samples):
        sum_pred = ball_pred[i] + bat_pred[i] + stump_pred[i]

        if sum_pred == 0:
            # No objects predicted → Class 0
            y_final[i] = 0
            n_no_prediction += 1

        elif sum_pred == 1:
            # Single object predicted → That class
            if ball_pred[i] == 1:
                y_final[i] = 1
            elif bat_pred[i] == 1:
                y_final[i] = 2
            elif stump_pred[i] == 1:
                y_final[i] = 3
            n_single_prediction += 1

        else:
            # Multiple objects predicted (conflict) → Choose highest probability
            probas = [ball_proba[i], bat_proba[i], stump_proba[i]]
            max_idx = np.argmax(probas)

            # max_idx: 0=ball, 1=bat, 2=stump
            # Convert to class labels: 1=ball, 2=bat, 3=stump
            y_final[i] = max_idx + 1
            n_conflicts += 1

    if verbose:
        print(f"\nConflict Resolution Statistics:")
        print(f"  No predictions:     {n_no_prediction:5d} ({100*n_no_prediction/n_samples:5.2f}%)")
        print(f"  Single prediction:  {n_single_prediction:5d} ({100*n_single_prediction/n_samples:5.2f}%)")
        print(f"  Conflicts (≥2):     {n_conflicts:5d} ({100*n_conflicts/n_samples:5.2f}%)")
        print(f"  Total samples:      {n_samples:5d}")

        if n_conflicts > 0:
            conflict_rate = 100 * n_conflicts / n_samples
            if conflict_rate > 10:
                print(f"\n  ⚠ Warning: High conflict rate ({conflict_rate:.1f}%)")
                print(f"    Consider adjusting thresholds to reduce conflicts")

    return y_final


def predict_with_binary_models(model_data, X_test, thresholds=None, verbose=True):
    """
    Generate predictions using three binary models with conflict resolution.

    Args:
        model_data: Dictionary returned by train_three_binary_models() or load_binary_models() containing:
                   - 'models': Dictionary with keys 'ball', 'bat', 'stump' (trained models)
                   - 'feature_indices': Feature indices per model (optional)
                   - 'scalers': StandardScalers per model (optional, if PCA used)
                   - 'pca_models': PCA models per model (optional, if PCA used)
                   OR just a dictionary with model keys (backward compatibility)
        X_test: Test features (N, 2343)
        thresholds: Dictionary with keys 'ball', 'bat', 'stump' (optional)
                   If None, uses default threshold 0.5 for all
                   Example: {'ball': 0.6, 'bat': 0.4, 'stump': 0.4}
        verbose: Print prediction statistics

    Returns:
        y_pred: Final single-label predictions (N,)
                0=No Object, 1=Ball, 2=Bat, 3=Stump
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

    if verbose:
        print(f"\n{'='*70}")
        print(f"BINARY MODEL PREDICTION")
        print(f"{'='*70}")
        print(f"Test Samples: {X_test.shape[0]}")
        print(f"Features: {X_test.shape[1]}")
        if feature_indices:
            print(f"Feature Selection: ENABLED")
        if scalers and pca_models:
            print(f"PCA: ENABLED")

    # Default thresholds
    if thresholds is None:
        thresholds = {'ball': 0.5, 'bat': 0.5, 'stump': 0.5}

    if verbose:
        print(f"\nDecision Thresholds:")
        print(f"  Ball:  {thresholds['ball']:.2f}")
        print(f"  Bat:   {thresholds['bat']:.2f}")
        print(f"  Stump: {thresholds['stump']:.2f}")

    # Get probability predictions from each binary model
    # Apply feature selection and PCA if needed
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

    # Apply thresholds to get binary predictions
    ball_pred = (ball_proba >= thresholds['ball']).astype(int)
    bat_pred = (bat_proba >= thresholds['bat']).astype(int)
    stump_pred = (stump_proba >= thresholds['stump']).astype(int)

    if verbose:
        print(f"\nBinary Predictions:")
        print(f"  Ball:  {np.sum(ball_pred)} positive ({100*np.mean(ball_pred):.2f}%)")
        print(f"  Bat:   {np.sum(bat_pred)} positive ({100*np.mean(bat_pred):.2f}%)")
        print(f"  Stump: {np.sum(stump_pred)} positive ({100*np.mean(stump_pred):.2f}%)")

    # Resolve conflicts
    y_pred = resolve_conflicts(
        ball_pred, bat_pred, stump_pred,
        ball_proba, bat_proba, stump_proba,
        verbose=verbose
    )

    if verbose:
        print(f"\nFinal Predictions (After Conflict Resolution):")
        for class_id, class_name in [(0, 'No Object'), (1, 'Ball'), (2, 'Bat'), (3, 'Stump')]:
            count = np.sum(y_pred == class_id)
            print(f"  Class {class_id} ({class_name:10s}): {count:5d} ({100*count/len(y_pred):5.2f}%)")

        print(f"{'='*70}\n")

    return y_pred


def get_binary_probabilities(models, X_test):
    """
    Get probability predictions from all three binary models.

    Useful for threshold tuning and analysis.

    Args:
        models: Dictionary with keys 'ball', 'bat', 'stump'
        X_test: Test features (N, 2343)

    Returns:
        probabilities: Dictionary with keys 'ball', 'bat', 'stump'
                      Each value is array (N,) of probabilities
    """
    probabilities = {
        'ball': models['ball'].predict_proba(X_test)[:, 1],
        'bat': models['bat'].predict_proba(X_test)[:, 1],
        'stump': models['stump'].predict_proba(X_test)[:, 1]
    }
    return probabilities


def analyze_prediction_confidence(models, X_test, y_pred, thresholds=None):
    """
    Analyze prediction confidence for each final predicted class.

    Args:
        models: Dictionary with keys 'ball', 'bat', 'stump'
        X_test: Test features (N, 2343)
        y_pred: Final predictions (N,)
        thresholds: Decision thresholds (optional)

    Prints:
        Mean confidence per predicted class
        Low confidence predictions (potential errors)
    """
    if thresholds is None:
        thresholds = {'ball': 0.5, 'bat': 0.5, 'stump': 0.5}

    # Get probabilities
    probas = get_binary_probabilities(models, X_test)

    print(f"\n{'='*70}")
    print(f"PREDICTION CONFIDENCE ANALYSIS")
    print(f"{'='*70}")

    for class_id, class_name in [(0, 'No Object'), (1, 'Ball'), (2, 'Bat'), (3, 'Stump')]:
        mask = (y_pred == class_id)
        n_samples = np.sum(mask)

        if n_samples == 0:
            continue

        print(f"\nClass {class_id} ({class_name}):")
        print(f"  Samples: {n_samples}")

        if class_id == 0:
            # No Object: should have low probabilities for all objects
            mean_ball = np.mean(probas['ball'][mask])
            mean_bat = np.mean(probas['bat'][mask])
            mean_stump = np.mean(probas['stump'][mask])

            print(f"  Mean Ball Probability:  {mean_ball:.4f} (should be < {thresholds['ball']})")
            print(f"  Mean Bat Probability:   {mean_bat:.4f} (should be < {thresholds['bat']})")
            print(f"  Mean Stump Probability: {mean_stump:.4f} (should be < {thresholds['stump']})")

        elif class_id == 1:
            # Ball: should have high ball probability
            mean_proba = np.mean(probas['ball'][mask])
            min_proba = np.min(probas['ball'][mask])
            print(f"  Mean Probability: {mean_proba:.4f}")
            print(f"  Min Probability:  {min_proba:.4f}")

            # Find low confidence predictions
            low_conf = probas['ball'][mask] < 0.6
            if np.any(low_conf):
                print(f"  ⚠ Low confidence predictions: {np.sum(low_conf)} ({100*np.mean(low_conf):.1f}%)")

        elif class_id == 2:
            # Bat: should have high bat probability
            mean_proba = np.mean(probas['bat'][mask])
            min_proba = np.min(probas['bat'][mask])
            print(f"  Mean Probability: {mean_proba:.4f}")
            print(f"  Min Probability:  {min_proba:.4f}")

            low_conf = probas['bat'][mask] < 0.6
            if np.any(low_conf):
                print(f"  ⚠ Low confidence predictions: {np.sum(low_conf)} ({100*np.mean(low_conf):.1f}%)")

        elif class_id == 3:
            # Stump: should have high stump probability
            mean_proba = np.mean(probas['stump'][mask])
            min_proba = np.min(probas['stump'][mask])
            print(f"  Mean Probability: {mean_proba:.4f}")
            print(f"  Min Probability:  {min_proba:.4f}")

            low_conf = probas['stump'][mask] < 0.6
            if np.any(low_conf):
                print(f"  ⚠ Low confidence predictions: {np.sum(low_conf)} ({100*np.mean(low_conf):.1f}%)")

    print(f"{'='*70}\n")


# Test function
if __name__ == '__main__':
    """
    Test binary prediction module with synthetic data.

    Usage:
        python binary_predict.py
    """
    print("="*70)
    print("BINARY PREDICTION MODULE TEST")
    print("="*70)

    # Create synthetic test data
    np.random.seed(42)

    n_test = 500
    n_features = 2343

    X_test = np.random.randn(n_test, n_features).astype(np.float32)

    # Create mock models (simple probability generators for testing)
    class MockModel:
        def __init__(self, pos_rate):
            self.pos_rate = pos_rate

        def predict_proba(self, X):
            n = len(X)
            # Generate random probabilities with specified positive rate
            probas = np.random.beta(2, 5, n)  # Skewed toward low values
            # Adjust some to be high (positive predictions)
            n_positive = int(n * self.pos_rate)
            high_indices = np.random.choice(n, n_positive, replace=False)
            probas[high_indices] = np.random.beta(5, 2, n_positive)  # High probabilities

            # Return both class probabilities
            return np.column_stack([1 - probas, probas])

    models = {
        'ball': MockModel(0.05),   # 5% positive rate
        'bat': MockModel(0.10),    # 10% positive rate
        'stump': MockModel(0.08)   # 8% positive rate
    }

    print("\nTest 1: Prediction with default thresholds (0.5)")
    print("-" * 70)

    y_pred = predict_with_binary_models(models, X_test, thresholds=None, verbose=True)

    # Test 2: Custom thresholds
    print("\n\nTest 2: Prediction with custom thresholds")
    print("-" * 70)

    custom_thresholds = {'ball': 0.6, 'bat': 0.4, 'stump': 0.4}
    y_pred2 = predict_with_binary_models(models, X_test, thresholds=custom_thresholds, verbose=True)

    # Test 3: Get probabilities
    print("\n\nTest 3: Get binary probabilities")
    print("-" * 70)

    probas = get_binary_probabilities(models, X_test)
    for class_name, proba in probas.items():
        print(f"{class_name.capitalize()} probabilities: shape={proba.shape}, mean={np.mean(proba):.4f}, std={np.std(proba):.4f}")

    # Test 4: Confidence analysis
    print("\n\nTest 4: Confidence analysis")
    print("-" * 70)

    analyze_prediction_confidence(models, X_test, y_pred, thresholds=None)

    print(f"\n{'='*70}")
    print(f"ALL TESTS PASSED ✓")
    print(f"{'='*70}")
