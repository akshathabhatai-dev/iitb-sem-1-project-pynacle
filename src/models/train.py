import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from typing import Tuple, Dict
import sys
import os

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'features'))

# ============================================================
#  XGBOOST TRAINING FUNCTION
# ============================================================
def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                  tune_hyperparameters: bool = True,
                  cv_folds: int = 5,
                  verbose: bool = True) -> XGBClassifier:

    if verbose:
        print("="*60)
        print("Training XGBoost Classifier")
        print("="*60)
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Classes: {np.unique(y_train)}")

    # Compute class weights → XGBoost uses scale_pos_weight for binary,
    # but for multi-class we apply weights manually.
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    weight_dict = dict(zip(classes, class_weights))

    sample_weights = np.array([weight_dict[c] for c in y_train])

    if tune_hyperparameters:
        if verbose:
            print("\nPerforming grid search for XGBoost...")

        param_grid = {
            'n_estimators': [200, 400],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        base_model = XGBClassifier(
            objective="multi:softmax",
            num_class=len(classes),
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42
        )

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv_folds,
            verbose=1 if verbose else 0,
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train, sample_weight=sample_weights)

        if verbose:
            print(f"\nBest XGBoost Params: {grid_search.best_params_}")
            print(f"Best F1-macro Score: {grid_search.best_score_:.4f}\n")

        model = grid_search.best_estimator_

    else:
        if verbose:
            print("\nTraining XGBoost with default parameters...")
        
        # Best XGBoost Params: {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 8, 'n_estimators': 200, 'subsample': 0.8}

        model = XGBClassifier(
            objective="multi:softmax",
            num_class=len(classes),
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42
        )

        # model = XGBClassifier(
        #     objective="multi:softmax",
        #     num_class=len(classes),
        #     n_estimators=400,
        #     max_depth=8,
        #     learning_rate=0.1,
        #     subsample=0.9,
        #     colsample_bytree=0.9,
        #     eval_metric="mlogloss",
        #     tree_method="hist",
        #     random_state=42
        # )

        model.fit(X_train, y_train, sample_weight=sample_weights)

        if verbose:
            scores = cross_val_score(
                model, X_train, y_train,
                scoring="f1_macro",
                cv=cv_folds
            )
            print(f"Cross-val F1-macro: {scores.mean():.4f} (+/- {scores.std():.4f})")

    print("="*60)
    print("XGBoost Training Complete!")
    print("="*60)

    return model


# ============================================================
# RANDOM FOREST TRAINING
# ============================================================
def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                       tune_hyperparameters: bool = True,
                       cv_folds: int = 5,
                       verbose: bool = True) -> RandomForestClassifier:
    """
    Train a Random Forest classifier for cricket object detection.

    Args:
        X_train: Training features of shape (n_samples, 88)
        y_train: Training labels of shape (n_samples,)
                 Labels: 0=no object, 1=ball, 2=bat, 3=stump
        tune_hyperparameters: Whether to perform grid search for hyperparameters
        cv_folds: Number of cross-validation folds
        verbose: Print training progress

    Returns:
        Trained Random Forest model
    """
    if verbose:
        print("="*60)
        print("Training Random Forest Classifier")
        print("="*60)
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Classes: {np.unique(y_train)}")

        # Class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        print("\nClass distribution:")
        class_names = {0: 'No Object', 1: 'Ball', 2: 'Bat', 3: 'Stump'}
        for cls, count in zip(unique, counts):
            print(f"  {class_names.get(cls, cls)}: {count} ({count/len(y_train)*100:.2f}%)")

    # Compute class weights to handle imbalance
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))

    if verbose:
        print("\nClass weights (for balancing):")
        for cls, weight in class_weight_dict.items():
            print(f"  {class_names.get(cls, cls)}: {weight:.3f}")

    if tune_hyperparameters:
        # Grid search for hyperparameter tuning
        if verbose:
            print("\nPerforming grid search for hyperparameter tuning...")

        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [15, 20, 25],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [2, 5, 10]
        }

        # Base model
        base_model = RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        # Grid search with cross-validation
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_folds,
            scoring='f1_macro',  # Balanced metric for imbalanced data
            n_jobs=-1,
            verbose=1 if verbose else 0
        )

        grid_search.fit(X_train, y_train)

        if verbose:
            print(f"\nBest parameters: {grid_search.best_params_}")
            print(f"Best CV F1-macro score: {grid_search.best_score_:.4f}")

        model = grid_search.best_estimator_

    else:
        # Train with default parameters
        if verbose:
            print("\nTraining with default parameters...")

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        # Cross-validation score
        if verbose:
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds,
                                       scoring='f1_macro', n_jobs=-1)
            print(f"Cross-validation F1-macro: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Feature importance analysis
    if verbose:
        print("\nTop 20 most important features:")
        from feature_extractor import get_all_feature_names

        feature_names = get_all_feature_names()
        feature_importances = model.feature_importances_

        # Get top 20 features
        top_indices = np.argsort(feature_importances)[-20:][::-1]

        for i, idx in enumerate(top_indices, 1):
            print(f"  {i:2d}. {feature_names[idx]:30s}: {feature_importances[idx]:.4f}")

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)

    return model


def save_model(model: RandomForestClassifier, output_path: str):
    """
    Save trained model to disk.

    Args:
        model: Trained model
        output_path: Path to save the model
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model saved to {output_path}")


def load_model(model_path: str) -> RandomForestClassifier:
    """
    Load trained model from disk.

    Args:
        model_path: Path to the saved model

    Returns:
        Loaded model
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print(f"Model loaded from {model_path}")
    return model


def load_training_data(features_path: str, labels_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training features and labels.

    Args:
        features_path: Path to features numpy file
        labels_path: Path to labels CSV file

    Returns:
        Tuple of (X_train, y_train)
    """
    # Load features
    X_train = np.load(features_path)

    # Load labels
    labels_df = pd.read_csv(labels_path)

    # Extract labels for all cells
    y_train = []
    for _, row in labels_df.iterrows():
        cell_labels = [row[f'c{i:02d}'] for i in range(1, 65)]
        y_train.extend(cell_labels)

    y_train = np.array(y_train, dtype=np.int32)

    print(f"Loaded training data: X={X_train.shape}, y={y_train.shape}")

    return X_train, y_train


def main():
    """Main training pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='Train Random Forest model for cricket object detection')
    parser.add_argument('--features', type=str, required=True,
                       help='Path to training features (.npy)')
    parser.add_argument('--labels', type=str, required=True,
                       help='Path to training labels (.csv)')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save trained model (.pkl)')
    parser.add_argument('--tune', action='store_true',
                       help='Perform hyperparameter tuning (slower)')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of cross-validation folds')

    args = parser.parse_args()

    # Load data
    print("Loading training data...")
    X_train, y_train = load_training_data(args.features, args.labels)

    # Train model
    # model = train_random_forest(
    #     X_train, y_train,
    #     tune_hyperparameters=args.tune,
    #     cv_folds=args.cv_folds,
    #     verbose=True
    # )
    if args.model == 'rf':
        model = train_random_forest(
            X_train, y_train,
            tune_hyperparameters=args.tune,
            cv_folds=args.cv_folds,
            verbose=True
        )
    else:
        model = train_xgboost(
            X_train, y_train,
            tune_hyperparameters=args.tune,
            cv_folds=args.cv_folds,
            verbose=True
        )

    # Save model
    save_model(model, args.output)


if __name__ == '__main__':
    main()
