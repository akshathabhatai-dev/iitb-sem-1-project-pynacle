import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional
import sys
import os

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'features'))

# Import configuration constants
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import PCA_VARIANCE_THRESHOLD, RANDOM_STATE, CV_FOLDS, CV_SCORING_METRIC


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                       tune_hyperparameters: bool = True,
                       cv_folds: int = CV_FOLDS,
                       verbose: bool = True,
                       use_pca: bool = True,
                       pca_variance: float = PCA_VARIANCE_THRESHOLD) -> Pipeline:
    """
    Train a Random Forest classifier with optional PCA preprocessing.

    Creates a sklearn Pipeline with:
        1. PCA dimensionality reduction (optional, preserves 95% variance by default)
        2. Random Forest classifier with balanced class weights

    Args:
        X_train: Training features of shape (n_samples, n_features)
        y_train: Training labels of shape (n_samples,)
                 Labels: 0=no object, 1=ball, 2=bat, 3=stump
        tune_hyperparameters: Whether to perform grid search for hyperparameters
        cv_folds: Number of cross-validation folds
        verbose: Print training progress
        use_pca: Whether to apply PCA dimensionality reduction (default: True)
        pca_variance: Amount of variance to preserve with PCA (default: 0.95 = 95%)

    Returns:
        Trained sklearn Pipeline containing PCA + Random Forest
    """
    if verbose:
        print("="*60)
        print("Training Random Forest Classifier with PCA Pipeline")
        print("="*60)
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Original features: {X_train.shape[1]}")
        print(f"PCA enabled: {use_pca}")
        if use_pca:
            print(f"PCA variance to preserve: {pca_variance*100:.1f}%")
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

    # Create pipeline steps (scale before PCA for proper feature scaling!)
    pipeline_steps = []

    if use_pca:
        pipeline_steps.append(('scaler', StandardScaler()))
        pipeline_steps.append(('pca', PCA(n_components=pca_variance, random_state=RANDOM_STATE)))

    pipeline_steps.append(('classifier', RandomForestClassifier(
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )))

    if tune_hyperparameters:
        # Grid search for hyperparameter tuning
        if verbose:
            print("\nPerforming grid search for hyperparameter tuning...")

        # Update param grid to use pipeline naming convention
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [15, 20, 25],
            'classifier__min_samples_split': [5, 10, 15],
            'classifier__min_samples_leaf': [2, 5, 10]
        }

        # Create pipeline
        pipeline = Pipeline(pipeline_steps)

        # Grid search with cross-validation
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv_folds,
            scoring=CV_SCORING_METRIC,
            n_jobs=-1,
            verbose=1 if verbose else 0
        )

        grid_search.fit(X_train, y_train)

        if verbose:
            print(f"\nBest parameters: {grid_search.best_params_}")
            print(f"Best CV F1-macro score: {grid_search.best_score_:.4f}")

            # Show PCA info if used
            if use_pca:
                pca_step = grid_search.best_estimator_.named_steps['pca']
                n_components = pca_step.n_components_
                explained_var = pca_step.explained_variance_ratio_.sum()
                print(f"\nPCA Info:")
                print(f"  Features reduced: {X_train.shape[1]} -> {n_components}")
                print(f"  Variance explained: {explained_var*100:.2f}%")

        pipeline = grid_search.best_estimator_

    else:
        # Train with default parameters
        if verbose:
            print("\nTraining with default parameters...")

        # Set default RF parameters
        pipeline_steps[-1] = ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))

        pipeline = Pipeline(pipeline_steps)
        pipeline.fit(X_train, y_train)

        # Cross-validation score
        if verbose:
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds,
                                       scoring=CV_SCORING_METRIC, n_jobs=-1)
            print(f"Cross-validation F1-macro: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

            # Show PCA info if used
            if use_pca:
                pca_step = pipeline.named_steps['pca']
                n_components = pca_step.n_components_
                explained_var = pca_step.explained_variance_ratio_.sum()
                print(f"\nPCA Info:")
                print(f"  Features reduced: {X_train.shape[1]} -> {n_components}")
                print(f"  Variance explained: {explained_var*100:.2f}%")

    # Feature importance analysis
    if verbose:
        print("\nTop 20 most important features:")
        from feature_extractor import get_all_feature_names

        feature_names = get_all_feature_names()

        # Get the classifier from the pipeline
        classifier = pipeline.named_steps['classifier']
        feature_importances = classifier.feature_importances_

        # If PCA was used, feature importances correspond to principal components
        if use_pca:
            print("  (Note: Importances are for PCA components, not original features)")
            for i in range(min(20, len(feature_importances))):
                print(f"  {i+1:2d}. PC{i+1:3d}: {feature_importances[i]:.4f}")
        else:
            # Get top 20 features
            top_indices = np.argsort(feature_importances)[-20:][::-1]
            for i, idx in enumerate(top_indices, 1):
                print(f"  {i:2d}. {feature_names[idx]:30s}: {feature_importances[idx]:.4f}")

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)

    return pipeline


def save_model(model, output_path: str):
    """
    Save trained model (Pipeline or RandomForestClassifier) to disk.

    Args:
        model: Trained model or pipeline
        output_path: Path to save the model
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model saved to {output_path}")


def load_model(model_path: str):
    """
    Load trained model (Pipeline or RandomForestClassifier) from disk.

    Args:
        model_path: Path to the saved model

    Returns:
        Loaded model or pipeline
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
    model = train_random_forest(
        X_train, y_train,
        tune_hyperparameters=args.tune,
        cv_folds=args.cv_folds,
        verbose=True
    )

    # Save model
    save_model(model, args.output)


if __name__ == '__main__':
    main()
