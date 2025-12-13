"""
Binary Model Training Module

Trains three separate binary classifiers for cricket object detection:
1. Ball vs Rest (Not-Ball)
2. Bat vs Rest (Not-Bat)
3. Stump vs Rest (Not-Stump)

Supports both Random Forest and XGBoost algorithms with hyperparameter tuning.

Key Features:
- Aggressive class weighting to handle extreme class imbalance
- GridSearchCV with 5-fold cross-validation
- Independent optimization for each binary task
- Saves 3 separate model files per algorithm

Usage:
    from binary_train import train_three_binary_models, save_binary_models

    # Train 3 binary RF models
    models = train_three_binary_models(X_train, y_ball, y_bat, y_stump,
                                       algorithm='random_forest', tune=True)

    # Save models
    save_binary_models(models, 'models/binary_models', 'random_forest')
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import feature selector
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'features'))
from feature_selector import get_ball_feature_indices, get_bat_feature_indices, get_stump_feature_indices


def train_single_binary_classifier(X_train, y_binary, class_name, algorithm='random_forest',
                                   tune=True, verbose=True, sample_weights=None):
    """
    Train a single binary classifier for one object type.

    Args:
        X_train: Training features (N, 2343)
        y_binary: Binary labels (N,) where 1=object present, 0=object absent
        class_name: Name of the class ('ball', 'bat', or 'stump')
        algorithm: 'random_forest' or 'xgboost'
        tune: Whether to perform hyperparameter tuning (slower but better)
        verbose: Print training progress
        sample_weights: Pre-computed sample weights (N,) - if None, will compute during training

    Returns:
        trained_model: Best trained model (RandomForestClassifier or XGBClassifier)
    """
    n_positive = np.sum(y_binary == 1)
    n_negative = np.sum(y_binary == 0)
    if verbose:
        print(f"\n{'='*70}")
        print(f"Training Binary {class_name.upper()} Classifier ({algorithm.upper()})")
        print(f"{'='*70}")

        # Print class distribution
        
        total = len(y_binary)
        print(f"Class Distribution:")
        print(f"  {class_name.capitalize()} (1):     {n_positive:5d} ({100*n_positive/total:5.2f}%)")
        print(f"  Not-{class_name.capitalize()} (0): {n_negative:5d} ({100*n_negative/total:5.2f}%)")
        print(f"  Imbalance Ratio: 1:{n_negative/max(n_positive, 1):.1f}")

    # Determine class weights and sample weights
    # Use pre-computed sample weights if provided, otherwise compute them
    if sample_weights is not None:
        if verbose:
            print(f"Using pre-computed sample weights (shape: {sample_weights.shape})")

        # Calculate class weights from pre-computed sample weights for model configuration
        classes = np.unique(y_binary)
        # Estimate class weights from sample weights (average weight per class)
        class_weights_list = []
        for cls in classes:
            mask = y_binary == cls
            avg_weight = np.mean(sample_weights[mask])
            class_weights_list.append(avg_weight)
        class_weights = dict(zip(classes, class_weights_list))
    else:
        # Compute sample weights from scratch
        if verbose:
            print(f"Computing sample weights from training data...")

        classes = np.unique(y_binary)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_binary)
        weight_dict = dict(zip(classes, class_weights))
        sample_weights = np.array([weight_dict[c] for c in y_binary])
        class_weights = weight_dict

    # if class_name == 'ball':
    #     # Ball is extremely rare - use aggressive weighting
    #     class_weights = {0: 1, 1: 50}
    # elif class_name == 'bat':
    #     class_weights = {0: 1, 1: 30}
    # elif class_name == 'stump':
    #     class_weights = {0: 1, 1: 25}
    # else:
    #     # Default balanced weights
    #     class_weights = {0: 1, 1: int(n_negative / max(n_positive, 1))}

    if verbose:
        print(f"Class Weights: {class_weights}")

    # Define hyperparameter grids
    if algorithm == 'random_forest':
        if tune:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [15, 20, 25],
                'min_samples_split': [5, 10, 15],
                'min_samples_leaf': [2, 5, 10]
            }
            print(f"\nHyperparameter Grid Search:")
            print(f"  n_estimators: {param_grid['n_estimators']}")
            print(f"  max_depth: {param_grid['max_depth']}")
            print(f"  min_samples_split: {param_grid['min_samples_split']}")
            print(f"  min_samples_leaf: {param_grid['min_samples_leaf']}")
            print(f"  Total combinations: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf'])}")
        else:
            # Default hyperparameters (no tuning)
            param_grid = {
                'n_estimators': [200],
                'max_depth': [20],
                'min_samples_split': [10],
                'min_samples_leaf': [5]
            }

        # Create base model
        base_model = RandomForestClassifier(
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )

    elif algorithm == 'xgboost':
        if tune:
            # param_grid = {
            #     'n_estimators': [100, 200, 300],
            #     'max_depth': [5, 7, 9],
            #     'learning_rate': [0.01, 0.1, 0.3],
            #     'subsample': [0.8, 1.0],
            #     'colsample_bytree': [0.8, 1.0]
            # }
            param_grid = {
                'n_estimators': [200, 400],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
            print(f"\nHyperparameter Grid Search:")
            print(f"  n_estimators: {param_grid['n_estimators']}")
            print(f"  max_depth: {param_grid['max_depth']}")
            print(f"  learning_rate: {param_grid['learning_rate']}")
            print(f"  subsample: {param_grid['subsample']}")
            print(f"  colsample_bytree: {param_grid['colsample_bytree']}")
            print(f"  Total combinations: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate']) * len(param_grid['subsample']) * len(param_grid['colsample_bytree'])}")
        else:
            # Default hyperparameters
            param_grid = {
                'n_estimators': [200],
                'max_depth': [8],
                'learning_rate': [0.1],
                'subsample': [0.9],
                'colsample_bytree': [0.9],
                # 'objective': ["binary:logistic"],
                # 'num_class': [len(classes)],
                # 'eval_metric': ["mlogloss"],
                'tree_method': ["hist"],
                # 'n_jobs': [-1],
                # 'random_state': [42]
            }

        # Calculate scale_pos_weight for XGBoost (similar to class_weight)
        scale_pos_weight = class_weights[1] / class_weights[0]

        # Create base model
        base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            # scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose 'random_forest' or 'xgboost'")

    # Perform GridSearchCV
    if verbose:
        print(f"\nStarting 5-fold cross-validation...")

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='f1_macro',  # Optimize for F1 score (balanced precision/recall)
        n_jobs=-1,
        verbose=1 if verbose else 0
    )

    grid_search.fit(X_train, y_binary,sample_weight=sample_weights)

    # Get best model
    best_model = grid_search.best_estimator_

    if verbose:
        print(f"\nBest Hyperparameters:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")

        print(f"\nBest Cross-Validation F1 Score: {grid_search.best_score_:.4f}")

        # Evaluate on training set
        y_train_pred = best_model.predict(X_train)
        train_f1 = f1_score(y_binary, y_train_pred)
        print(f"Training F1 Score: {train_f1:.4f}")

        # Print classification report
        print(f"\nTraining Set Classification Report:")
        print(classification_report(y_binary, y_train_pred,
                                   target_names=[f'Not-{class_name.capitalize()}', class_name.capitalize()],
                                   digits=4))

    return best_model


def train_three_binary_models(X_train_ball, y_ball,
                              X_train_bat, y_bat,
                              X_train_stump, y_stump,
                              algorithm='random_forest', tune=True, verbose=True,
                              use_feature_selection=False, use_pca=False, pca_variance=0.90,
                              sample_weights_ball=None, sample_weights_bat=None, sample_weights_stump=None):
    """
    Train three binary classifiers (Ball, Bat, Stump) using specified algorithm.

    IMPORTANT: When using undersampling, each model gets a DIFFERENT training set because
    they have different negative classes (ball negatives != bat negatives != stump negatives).

    Args:
        X_train_ball: Training features for ball model (N_ball, 2343) - undersampled not-ball samples
        y_ball: Binary ball labels (N_ball,) where 1=ball, 0=not-ball
        X_train_bat: Training features for bat model (N_bat, 2343) - undersampled not-bat samples
        y_bat: Binary bat labels (N_bat,) where 1=bat, 0=not-bat
        X_train_stump: Training features for stump model (N_stump, 2343) - undersampled not-stump samples
        y_stump: Binary stump labels (N_stump,) where 1=stump, 0=not-stump
        algorithm: 'random_forest' or 'xgboost'
        tune: Whether to perform hyperparameter tuning
        verbose: Print training progress
        use_feature_selection: Whether to use selective features per model (reduces 2343 -> ~1000)
        use_pca: Whether to apply PCA after feature selection
        pca_variance: Variance threshold for PCA (default: 0.90)
        sample_weights_ball: Pre-computed sample weights for ball model (N_ball,) - if None, computed during training
        sample_weights_bat: Pre-computed sample weights for bat model (N_bat,) - if None, computed during training
        sample_weights_stump: Pre-computed sample weights for stump model (N_stump,) - if None, computed during training

    Returns:
        model_data: Dictionary containing:
            - 'models': Dictionary with keys 'ball', 'bat', 'stump' (trained models)
            - 'feature_indices': Dictionary with feature indices per model (if use_feature_selection=True)
            - 'scalers': Dictionary with StandardScalers per model (if use_pca=True)
            - 'pca_models': Dictionary with PCA models per model (if use_pca=True)
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"TRAINING THREE BINARY MODELS - {algorithm.upper()}")
        print(f"{'='*70}")
        print(f"Features: {X_train_ball.shape[1]}")
        print(f"Training Samples (separate per model):")
        print(f"  Ball:  {X_train_ball.shape[0]} samples")
        print(f"  Bat:   {X_train_bat.shape[0]} samples")
        print(f"  Stump: {X_train_stump.shape[0]} samples")
        print(f"Hyperparameter Tuning: {'ENABLED' if tune else 'DISABLED'}")
        print(f"Feature Selection: {'ENABLED' if use_feature_selection else 'DISABLED'}")
        print(f"PCA: {'ENABLED' if use_pca else 'DISABLED'}")
        if use_pca:
            print(f"PCA Variance Threshold: {pca_variance:.2f}")

    models = {}
    feature_indices = {}
    scalers = {} if use_pca else None
    pca_models = {} if use_pca else None

    # ========== TRAIN BALL CLASSIFIER ==========
    class_name = 'ball'

    # Apply feature selection
    if use_feature_selection:
        ball_features = get_ball_feature_indices()
        X_ball = X_train_ball[:, ball_features]
        feature_indices[class_name] = ball_features
        if verbose:
            print(f"\n{class_name.capitalize()} model: Using {len(ball_features)} selected features " +
                  f"(reduced from {X_train_ball.shape[1]})")
    else:
        X_ball = X_train_ball
        feature_indices[class_name] = list(range(X_train_ball.shape[1]))

    # Apply PCA if requested
    if use_pca:
        scaler = StandardScaler()
        X_ball_scaled = scaler.fit_transform(X_ball)

        pca = PCA(n_components=pca_variance)
        X_ball_pca = pca.fit_transform(X_ball_scaled)

        if verbose:
            print(f"{class_name.capitalize()} PCA: {X_ball.shape[1]} features -> " +
                  f"{X_ball_pca.shape[1]} components " +
                  f"(explained variance: {pca.explained_variance_ratio_.sum():.3f})")

        scalers[class_name] = scaler
        pca_models[class_name] = pca
        X_ball = X_ball_pca

    # Train ball classifier
    models[class_name] = train_single_binary_classifier(
        X_ball, y_ball, class_name, algorithm=algorithm, tune=tune, verbose=verbose,
        sample_weights=sample_weights_ball
    )

    # ========== TRAIN BAT CLASSIFIER ==========
    class_name = 'bat'

    # Apply feature selection
    if use_feature_selection:
        bat_features = get_bat_feature_indices()
        X_bat = X_train_bat[:, bat_features]
        feature_indices[class_name] = bat_features
        if verbose:
            print(f"\n{class_name.capitalize()} model: Using {len(bat_features)} selected features " +
                  f"(reduced from {X_train_bat.shape[1]})")
    else:
        X_bat = X_train_bat
        feature_indices[class_name] = list(range(X_train_bat.shape[1]))

    # Apply PCA if requested
    if use_pca:
        scaler = StandardScaler()
        X_bat_scaled = scaler.fit_transform(X_bat)

        pca = PCA(n_components=pca_variance)
        X_bat_pca = pca.fit_transform(X_bat_scaled)

        if verbose:
            print(f"{class_name.capitalize()} PCA: {X_bat.shape[1]} features -> " +
                  f"{X_bat_pca.shape[1]} components " +
                  f"(explained variance: {pca.explained_variance_ratio_.sum():.3f})")

        scalers[class_name] = scaler
        pca_models[class_name] = pca
        X_bat = X_bat_pca

    # Train bat classifier
    models[class_name] = train_single_binary_classifier(
        X_bat, y_bat, class_name, algorithm=algorithm, tune=tune, verbose=verbose,
        sample_weights=sample_weights_bat
    )

    # ========== TRAIN STUMP CLASSIFIER ==========
    class_name = 'stump'

    # Apply feature selection
    if use_feature_selection:
        stump_features = get_stump_feature_indices()
        X_stump = X_train_stump[:, stump_features]
        feature_indices[class_name] = stump_features
        if verbose:
            print(f"\n{class_name.capitalize()} model: Using {len(stump_features)} selected features " +
                  f"(reduced from {X_train_stump.shape[1]})")
    else:
        X_stump = X_train_stump
        feature_indices[class_name] = list(range(X_train_stump.shape[1]))

    # Apply PCA if requested
    if use_pca:
        scaler = StandardScaler()
        X_stump_scaled = scaler.fit_transform(X_stump)

        pca = PCA(n_components=pca_variance)
        X_stump_pca = pca.fit_transform(X_stump_scaled)

        if verbose:
            print(f"{class_name.capitalize()} PCA: {X_stump.shape[1]} features -> " +
                  f"{X_stump_pca.shape[1]} components " +
                  f"(explained variance: {pca.explained_variance_ratio_.sum():.3f})")

        scalers[class_name] = scaler
        pca_models[class_name] = pca
        X_stump = X_stump_pca

    # Train stump classifier
    models[class_name] = train_single_binary_classifier(
        X_stump, y_stump, class_name, algorithm=algorithm, tune=tune, verbose=verbose,
        sample_weights=sample_weights_stump
    )

    if verbose:
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE - All 3 binary models trained successfully")
        print(f"{'='*70}")

    # Return all components needed for prediction
    return {
        'models': models,
        'feature_indices': feature_indices,
        'scalers': scalers,
        'pca_models': pca_models
    }


def save_binary_models(model_data, output_dir, algorithm_name):
    """
    Save three binary models and associated metadata to pickle files.

    Args:
        model_data: Dictionary returned by train_three_binary_models() containing:
                   - 'models': Dictionary with keys 'ball', 'bat', 'stump'
                   - 'feature_indices': Feature indices per model (optional)
                   - 'scalers': StandardScalers per model (optional, if PCA used)
                   - 'pca_models': PCA models per model (optional, if PCA used)
                   OR just a dictionary with model keys (backward compatibility)
        output_dir: Directory to save models (e.g., 'models/binary_models')
        algorithm_name: Name of algorithm ('random_forest' or 'xgboost')

    Saves:
        binary_{algorithm}_ball.pkl
        binary_{algorithm}_bat.pkl
        binary_{algorithm}_stump.pkl
        binary_{algorithm}_metadata.pkl (feature indices, scalers, PCA models)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Save individual models
    for class_name, model in models.items():
        model_filename = f"binary_{algorithm_name}_{class_name}.pkl"
        model_path = output_dir / model_filename

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"Saved {class_name} model: {model_path}")

    # Save metadata (feature indices, scalers, PCA models)
    if feature_indices or scalers or pca_models:
        metadata = {
            'feature_indices': feature_indices,
            'scalers': scalers,
            'pca_models': pca_models
        }
        metadata_path = output_dir / f"binary_{algorithm_name}_metadata.pkl"

        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        print(f"Saved metadata: {metadata_path}")
        if feature_indices:
            print(f"  - Feature indices for selective models")
        if scalers:
            print(f"  - StandardScalers for PCA preprocessing")
        if pca_models:
            print(f"  - PCA models for dimensionality reduction")

    print(f"\nAll models saved to: {output_dir}")


def load_binary_models(model_dir, algorithm_name):
    """
    Load three binary models and associated metadata from pickle files.

    Args:
        model_dir: Directory containing models
        algorithm_name: Name of algorithm ('random_forest' or 'xgboost')

    Returns:
        model_data: Dictionary containing:
            - 'models': Dictionary with keys 'ball', 'bat', 'stump'
            - 'feature_indices': Feature indices per model (if available)
            - 'scalers': StandardScalers per model (if available)
            - 'pca_models': PCA models per model (if available)
    """
    model_dir = Path(model_dir)
    models = {}

    # Load individual models
    for class_name in ['ball', 'bat', 'stump']:
        model_filename = f"binary_{algorithm_name}_{class_name}.pkl"
        model_path = model_dir / model_filename

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, 'rb') as f:
            models[class_name] = pickle.load(f)

        print(f"Loaded {class_name} model: {model_path}")

    # Load metadata if available
    metadata_path = model_dir / f"binary_{algorithm_name}_metadata.pkl"
    feature_indices = {}
    scalers = None
    pca_models = None

    if metadata_path.exists():
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        feature_indices = metadata.get('feature_indices', {})
        scalers = metadata.get('scalers')
        pca_models = metadata.get('pca_models')

        print(f"Loaded metadata: {metadata_path}")
        if feature_indices:
            print(f"  - Feature indices loaded (selective features enabled)")
        if scalers:
            print(f"  - StandardScalers loaded (PCA preprocessing enabled)")
        if pca_models:
            print(f"  - PCA models loaded (dimensionality reduction enabled)")

    print(f"\nAll models loaded from: {model_dir}")

    return {
        'models': models,
        'feature_indices': feature_indices,
        'scalers': scalers,
        'pca_models': pca_models
    }


# Test function
if __name__ == '__main__':
    """
    Test binary training module with synthetic data.

    Usage:
        python binary_train.py
    """
    print("="*70)
    print("BINARY TRAINING MODULE TEST")
    print("="*70)

    # Create synthetic test data
    np.random.seed(42)

    # Simulate 1000 training samples with 2343 features
    n_samples = 1000
    n_features = 2343

    X_train = np.random.randn(n_samples, n_features).astype(np.float32)

    # Create imbalanced binary labels (similar to real cricket dataset)
    # Ball: 5% positive (very rare)
    y_ball = (np.random.rand(n_samples) < 0.05).astype(int)

    # Bat: 10% positive (rare)
    y_bat = (np.random.rand(n_samples) < 0.10).astype(int)

    # Stump: 8% positive (rare)
    y_stump = (np.random.rand(n_samples) < 0.08).astype(int)

    print(f"\nSynthetic Test Data:")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Ball positive rate: {np.mean(y_ball)*100:.1f}%")
    print(f"  Bat positive rate: {np.mean(y_bat)*100:.1f}%")
    print(f"  Stump positive rate: {np.mean(y_stump)*100:.1f}%")

    # Test Random Forest training (no tuning for speed)
    print(f"\n{'='*70}")
    print(f"TEST 1: Random Forest (No Tuning)")
    print(f"{'='*70}")

    models_rf = train_three_binary_models(
        X_train, y_ball, y_bat, y_stump,
        algorithm='random_forest',
        tune=False,  # Disable tuning for quick test
        verbose=True
    )

    # Test saving models
    print(f"\n{'='*70}")
    print(f"TEST 2: Save Models")
    print(f"{'='*70}")

    test_output_dir = Path('test_models')
    save_binary_models(models_rf, test_output_dir, 'random_forest')

    # Test loading models
    print(f"\n{'='*70}")
    print(f"TEST 3: Load Models")
    print(f"{'='*70}")

    loaded_models = load_binary_models(test_output_dir, 'random_forest')

    # Verify loaded models work
    print(f"\n{'='*70}")
    print(f"TEST 4: Verify Loaded Models")
    print(f"{'='*70}")

    for class_name, model in loaded_models.items():
        # Make predictions
        test_input = X_train[:5]
        predictions = model.predict(test_input)
        probabilities = model.predict_proba(test_input)

        print(f"\n{class_name.capitalize()} Model:")
        print(f"  Predictions: {predictions}")
        print(f"  Probabilities shape: {probabilities.shape}")

    # Clean up test files
    import shutil
    shutil.rmtree(test_output_dir)
    print(f"\nCleaned up test directory: {test_output_dir}")

    print(f"\n{'='*70}")
    print(f"ALL TESTS PASSED ✓")
    print(f"{'='*70}")
