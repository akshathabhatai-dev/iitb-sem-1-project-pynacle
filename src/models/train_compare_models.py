"""
Model Comparison Module for Cricket Object Detection

This module implements comparison across 5 machine learning algorithms:
1. Logistic Regression
2. Random Forest
3. Support Vector Machine (SVM)
4. K-Nearest Neighbors (KNN)
5. XGBoost

Each model is trained with hyperparameter tuning via GridSearchCV and evaluated
using cross-validation with F1-macro as the primary metric.

The best model is selected based on F1-macro score and saved for deployment.
"""

import numpy as np
import pandas as pd
import pickle
import time
from pathlib import Path
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

import sys
import os

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'features'))


def train_logistic_regression(X: np.ndarray, y: np.ndarray,
                              tune_hyperparameters: bool = True,
                              cv_folds: int = 5,
                              verbose: bool = True) -> Tuple[Pipeline, Dict]:
    """
    Train Logistic Regression with feature scaling.

    Args:
        X: Training features (n_samples, n_features)
        y: Training labels (n_samples,)
        tune_hyperparameters: Whether to perform grid search
        cv_folds: Number of cross-validation folds
        verbose: Print training progress

    Returns:
        Tuple of (trained_pipeline, metrics_dict)
    """
    if verbose:
        print("\n" + "="*70)
        print("TRAINING: Logistic Regression")
        print("="*70)

    start_time = time.time()

    if tune_hyperparameters:
        # Hyperparameter grid
        param_grid = {
            'classifier__C': [0.01, 0.1, 1.0, 10.0],
            'classifier__penalty': ['l2'],
            'classifier__solver': ['lbfgs', 'saga'],
            'classifier__max_iter': [1000]
        }

        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                multi_class='multinomial',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ))
        ])

        # Grid search
        grid_search = GridSearchCV(
            pipeline, param_grid,
            cv=cv_folds,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X, y)
        best_pipeline = grid_search.best_estimator_

        if verbose:
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV F1-macro: {grid_search.best_score_:.4f}")

    else:
        # Default parameters
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='lbfgs',
                multi_class='multinomial',
                class_weight='balanced',
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            ))
        ])

        pipeline.fit(X, y)
        best_pipeline = pipeline

    # Cross-validation evaluation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_results = cross_validate(
        best_pipeline, X, y,
        cv=cv,
        scoring=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'],
        n_jobs=-1
    )

    train_time = time.time() - start_time

    metrics = {
        'model_name': 'Logistic Regression',
        'accuracy_mean': cv_results['test_accuracy'].mean(),
        'accuracy_std': cv_results['test_accuracy'].std(),
        'f1_macro_mean': cv_results['test_f1_macro'].mean(),
        'f1_macro_std': cv_results['test_f1_macro'].std(),
        'precision_macro_mean': cv_results['test_precision_macro'].mean(),
        'precision_macro_std': cv_results['test_precision_macro'].std(),
        'recall_macro_mean': cv_results['test_recall_macro'].mean(),
        'recall_macro_std': cv_results['test_recall_macro'].std(),
        'train_time': train_time
    }

    if verbose:
        print(f"\nCross-Validation Results:")
        print(f"  Accuracy:  {metrics['accuracy_mean']:.4f} (+/- {metrics['accuracy_std']:.4f})")
        print(f"  F1-macro:  {metrics['f1_macro_mean']:.4f} (+/- {metrics['f1_macro_std']:.4f})")
        print(f"  Precision: {metrics['precision_macro_mean']:.4f} (+/- {metrics['precision_macro_std']:.4f})")
        print(f"  Recall:    {metrics['recall_macro_mean']:.4f} (+/- {metrics['recall_macro_std']:.4f})")
        print(f"  Train Time: {train_time:.2f}s")

    return best_pipeline, metrics


def train_random_forest(X: np.ndarray, y: np.ndarray,
                       tune_hyperparameters: bool = True,
                       cv_folds: int = 5,
                       verbose: bool = True) -> Tuple[RandomForestClassifier, Dict]:
    """
    Train Random Forest classifier.

    Args:
        X: Training features (n_samples, n_features)
        y: Training labels (n_samples,)
        tune_hyperparameters: Whether to perform grid search
        cv_folds: Number of cross-validation folds
        verbose: Print training progress

    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    if verbose:
        print("\n" + "="*70)
        print("TRAINING: Random Forest")
        print("="*70)

    start_time = time.time()

    if tune_hyperparameters:
        # Hyperparameter grid (from existing train.py)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [15, 20, 25],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [2, 5, 10]
        }

        base_model = RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        grid_search = GridSearchCV(
            base_model, param_grid,
            cv=cv_folds,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_

        if verbose:
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV F1-macro: {grid_search.best_score_:.4f}")

    else:
        # Default parameters
        best_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        best_model.fit(X, y)

    # Cross-validation evaluation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_results = cross_validate(
        best_model, X, y,
        cv=cv,
        scoring=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'],
        n_jobs=-1
    )

    train_time = time.time() - start_time

    metrics = {
        'model_name': 'Random Forest',
        'accuracy_mean': cv_results['test_accuracy'].mean(),
        'accuracy_std': cv_results['test_accuracy'].std(),
        'f1_macro_mean': cv_results['test_f1_macro'].mean(),
        'f1_macro_std': cv_results['test_f1_macro'].std(),
        'precision_macro_mean': cv_results['test_precision_macro'].mean(),
        'precision_macro_std': cv_results['test_precision_macro'].std(),
        'recall_macro_mean': cv_results['test_recall_macro'].mean(),
        'recall_macro_std': cv_results['test_recall_macro'].std(),
        'train_time': train_time
    }

    if verbose:
        print(f"\nCross-Validation Results:")
        print(f"  Accuracy:  {metrics['accuracy_mean']:.4f} (+/- {metrics['accuracy_std']:.4f})")
        print(f"  F1-macro:  {metrics['f1_macro_mean']:.4f} (+/- {metrics['f1_macro_std']:.4f})")
        print(f"  Precision: {metrics['precision_macro_mean']:.4f} (+/- {metrics['precision_macro_std']:.4f})")
        print(f"  Recall:    {metrics['recall_macro_mean']:.4f} (+/- {metrics['recall_macro_std']:.4f})")
        print(f"  Train Time: {train_time:.2f}s")

    return best_model, metrics


def train_svm(X: np.ndarray, y: np.ndarray,
             tune_hyperparameters: bool = True,
             cv_folds: int = 5,
             verbose: bool = True) -> Tuple[Pipeline, Dict]:
    """
    Train Support Vector Machine with RBF kernel.

    Args:
        X: Training features (n_samples, n_features)
        y: Training labels (n_samples,)
        tune_hyperparameters: Whether to perform grid search
        cv_folds: Number of cross-validation folds
        verbose: Print training progress

    Returns:
        Tuple of (trained_pipeline, metrics_dict)
    """
    if verbose:
        print("\n" + "="*70)
        print("TRAINING: Support Vector Machine (SVM)")
        print("="*70)

    start_time = time.time()

    if tune_hyperparameters:
        # Hyperparameter grid (coarser for speed)
        param_grid = {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__gamma': ['scale', 'auto', 0.01],
            'classifier__kernel': ['rbf']
        }

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(
                class_weight='balanced',
                random_state=42,
                max_iter=5000
            ))
        ])

        grid_search = GridSearchCV(
            pipeline, param_grid,
            cv=cv_folds,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X, y)
        best_pipeline = grid_search.best_estimator_

        if verbose:
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV F1-macro: {grid_search.best_score_:.4f}")

    else:
        # Default parameters
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                class_weight='balanced',
                random_state=42,
                max_iter=5000
            ))
        ])

        pipeline.fit(X, y)
        best_pipeline = pipeline

    # Cross-validation evaluation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_results = cross_validate(
        best_pipeline, X, y,
        cv=cv,
        scoring=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'],
        n_jobs=-1
    )

    train_time = time.time() - start_time

    metrics = {
        'model_name': 'SVM',
        'accuracy_mean': cv_results['test_accuracy'].mean(),
        'accuracy_std': cv_results['test_accuracy'].std(),
        'f1_macro_mean': cv_results['test_f1_macro'].mean(),
        'f1_macro_std': cv_results['test_f1_macro'].std(),
        'precision_macro_mean': cv_results['test_precision_macro'].mean(),
        'precision_macro_std': cv_results['test_precision_macro'].std(),
        'recall_macro_mean': cv_results['test_recall_macro'].mean(),
        'recall_macro_std': cv_results['test_recall_macro'].std(),
        'train_time': train_time
    }

    if verbose:
        print(f"\nCross-Validation Results:")
        print(f"  Accuracy:  {metrics['accuracy_mean']:.4f} (+/- {metrics['accuracy_std']:.4f})")
        print(f"  F1-macro:  {metrics['f1_macro_mean']:.4f} (+/- {metrics['f1_macro_std']:.4f})")
        print(f"  Precision: {metrics['precision_macro_mean']:.4f} (+/- {metrics['precision_macro_std']:.4f})")
        print(f"  Recall:    {metrics['recall_macro_mean']:.4f} (+/- {metrics['recall_macro_std']:.4f})")
        print(f"  Train Time: {train_time:.2f}s")

    return best_pipeline, metrics


def train_knn(X: np.ndarray, y: np.ndarray,
             tune_hyperparameters: bool = True,
             cv_folds: int = 5,
             verbose: bool = True) -> Tuple[Pipeline, Dict]:
    """
    Train K-Nearest Neighbors classifier.

    Args:
        X: Training features (n_samples, n_features)
        y: Training labels (n_samples,)
        tune_hyperparameters: Whether to perform grid search
        cv_folds: Number of cross-validation folds
        verbose: Print training progress

    Returns:
        Tuple of (trained_pipeline, metrics_dict)
    """
    if verbose:
        print("\n" + "="*70)
        print("TRAINING: K-Nearest Neighbors (KNN)")
        print("="*70)

    start_time = time.time()

    if tune_hyperparameters:
        # Hyperparameter grid
        param_grid = {
            'classifier__n_neighbors': [3, 5, 7, 11],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan']
        }

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier(n_jobs=-1))
        ])

        grid_search = GridSearchCV(
            pipeline, param_grid,
            cv=cv_folds,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X, y)
        best_pipeline = grid_search.best_estimator_

        if verbose:
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV F1-macro: {grid_search.best_score_:.4f}")

    else:
        # Default parameters
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                metric='euclidean',
                n_jobs=-1
            ))
        ])

        pipeline.fit(X, y)
        best_pipeline = pipeline

    # Cross-validation evaluation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_results = cross_validate(
        best_pipeline, X, y,
        cv=cv,
        scoring=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'],
        n_jobs=-1
    )

    train_time = time.time() - start_time

    metrics = {
        'model_name': 'KNN',
        'accuracy_mean': cv_results['test_accuracy'].mean(),
        'accuracy_std': cv_results['test_accuracy'].std(),
        'f1_macro_mean': cv_results['test_f1_macro'].mean(),
        'f1_macro_std': cv_results['test_f1_macro'].std(),
        'precision_macro_mean': cv_results['test_precision_macro'].mean(),
        'precision_macro_std': cv_results['test_precision_macro'].std(),
        'recall_macro_mean': cv_results['test_recall_macro'].mean(),
        'recall_macro_std': cv_results['test_recall_macro'].std(),
        'train_time': train_time
    }

    if verbose:
        print(f"\nCross-Validation Results:")
        print(f"  Accuracy:  {metrics['accuracy_mean']:.4f} (+/- {metrics['accuracy_std']:.4f})")
        print(f"  F1-macro:  {metrics['f1_macro_mean']:.4f} (+/- {metrics['f1_macro_std']:.4f})")
        print(f"  Precision: {metrics['precision_macro_mean']:.4f} (+/- {metrics['precision_macro_std']:.4f})")
        print(f"  Recall:    {metrics['recall_macro_mean']:.4f} (+/- {metrics['recall_macro_std']:.4f})")
        print(f"  Train Time: {train_time:.2f}s")

    return best_pipeline, metrics


def train_xgboost(X: np.ndarray, y: np.ndarray,
                 tune_hyperparameters: bool = True,
                 cv_folds: int = 5,
                 verbose: bool = True) -> Tuple[xgb.XGBClassifier, Dict]:
    """
    Train XGBoost classifier.

    Args:
        X: Training features (n_samples, n_features)
        y: Training labels (n_samples,)
        tune_hyperparameters: Whether to perform grid search
        cv_folds: Number of cross-validation folds
        verbose: Print training progress

    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    if verbose:
        print("\n" + "="*70)
        print("TRAINING: XGBoost")
        print("="*70)

    start_time = time.time()

    # Compute class weights for imbalanced data
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    sample_weights = np.array([class_weights[int(label)] for label in y])

    if tune_hyperparameters:
        # Hyperparameter grid (coarser for speed)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        base_model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=len(classes),
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )

        grid_search = GridSearchCV(
            base_model, param_grid,
            cv=cv_folds,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X, y, sample_weight=sample_weights)
        best_model = grid_search.best_estimator_

        if verbose:
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV F1-macro: {grid_search.best_score_:.4f}")

    else:
        # Default parameters
        best_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softmax',
            num_class=len(classes),
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )

        best_model.fit(X, y, sample_weight=sample_weights)

    # Cross-validation evaluation (without sample_weight for CV - XGBoost already trained with it)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_results = cross_validate(
        best_model, X, y,
        cv=cv,
        scoring=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'],
        n_jobs=-1
    )

    train_time = time.time() - start_time

    metrics = {
        'model_name': 'XGBoost',
        'accuracy_mean': cv_results['test_accuracy'].mean(),
        'accuracy_std': cv_results['test_accuracy'].std(),
        'f1_macro_mean': cv_results['test_f1_macro'].mean(),
        'f1_macro_std': cv_results['test_f1_macro'].std(),
        'precision_macro_mean': cv_results['test_precision_macro'].mean(),
        'precision_macro_std': cv_results['test_precision_macro'].std(),
        'recall_macro_mean': cv_results['test_recall_macro'].mean(),
        'recall_macro_std': cv_results['test_recall_macro'].std(),
        'train_time': train_time
    }

    if verbose:
        print(f"\nCross-Validation Results:")
        print(f"  Accuracy:  {metrics['accuracy_mean']:.4f} (+/- {metrics['accuracy_std']:.4f})")
        print(f"  F1-macro:  {metrics['f1_macro_mean']:.4f} (+/- {metrics['f1_macro_std']:.4f})")
        print(f"  Precision: {metrics['precision_macro_mean']:.4f} (+/- {metrics['precision_macro_std']:.4f})")
        print(f"  Recall:    {metrics['recall_macro_mean']:.4f} (+/- {metrics['recall_macro_std']:.4f})")
        print(f"  Train Time: {train_time:.2f}s")

    return best_model, metrics


def compare_models(X_train: np.ndarray, y_train: np.ndarray,
                  tune_hyperparameters: bool = True,
                  cv_folds: int = 5,
                  verbose: bool = True) -> Dict[str, Dict]:
    """
    Train and compare all 5 models.

    Args:
        X_train: Training features
        y_train: Training labels
        tune_hyperparameters: Whether to tune each model
        cv_folds: Cross-validation folds
        verbose: Print progress

    Returns:
        Dictionary mapping model_name -> {model, metrics}
    """
    print("\n" + "="*70)
    print("MODEL COMPARISON - CRICKET OBJECT DETECTION")
    print("="*70)
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {np.unique(y_train)}")
    print(f"Hyperparameter tuning: {'ENABLED' if tune_hyperparameters else 'DISABLED'}")
    print(f"Cross-validation folds: {cv_folds}")
    print("="*70)

    results = {}

    # Train each model
    models_to_train = [
        ('logistic', train_logistic_regression),
        ('rf', train_random_forest),
        ('svm', train_svm),
        ('knn', train_knn),
        ('xgboost', train_xgboost)
    ]

    for model_key, train_func in models_to_train:
        model, metrics = train_func(X_train, y_train, tune_hyperparameters, cv_folds, verbose)
        results[model_key] = {
            'model': model,
            'metrics': metrics
        }

    return results


def select_best_model(comparison_results: Dict[str, Dict]) -> Tuple[str, Any, str]:
    """
    Select best model based on F1-macro score.

    Args:
        comparison_results: Results from compare_models()

    Returns:
        Tuple of (best_model_key, best_model, justification_text)
    """
    # Rank by F1-macro
    rankings = []
    for model_key, result in comparison_results.items():
        metrics = result['metrics']
        rankings.append({
            'model_key': model_key,
            'model_name': metrics['model_name'],
            'f1_macro': metrics['f1_macro_mean'],
            'accuracy': metrics['accuracy_mean'],
            'train_time': metrics['train_time']
        })

    # Sort by F1-macro (descending)
    rankings = sorted(rankings, key=lambda x: x['f1_macro'], reverse=True)

    best = rankings[0]
    best_model_key = best['model_key']
    best_model = comparison_results[best_model_key]['model']

    # Generate justification
    justification = f"""
MODEL SELECTION JUSTIFICATION
{'='*70}

Selected Model: {best['model_name']}
F1-Macro Score: {best['f1_macro']:.4f}
Accuracy: {best['accuracy']:.4f}
Training Time: {best['train_time']:.2f}s

RANKING (by F1-Macro):
"""

    for i, rank in enumerate(rankings, 1):
        justification += f"\n{i}. {rank['model_name']:20s} - F1: {rank['f1_macro']:.4f}, Acc: {rank['accuracy']:.4f}, Time: {rank['train_time']:.2f}s"

    justification += f"""

RATIONALE:
{best['model_name']} was selected as the final model based on the highest
F1-macro score across 5-fold cross-validation. F1-macro is the most appropriate
metric for this imbalanced multi-class classification task as it:

1. Treats all classes (no-object, ball, bat, stump) equally
2. Accounts for both precision and recall
3. Avoids accuracy bias from imbalanced class distribution

The selected model will be used for generating final predictions on test data.
{'='*70}
"""

    return best_model_key, best_model, justification


def save_comparison_results(comparison_results: Dict[str, Dict],
                           output_path: str,
                           justification: str = None):
    """
    Save model comparison results to CSV and TXT files.

    Args:
        comparison_results: Results from compare_models()
        output_path: Base path for output files
        justification: Optional justification text
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create comparison DataFrame
    rows = []
    for model_key, result in comparison_results.items():
        metrics = result['metrics']
        rows.append({
            'Model': metrics['model_name'],
            'Accuracy': f"{metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}",
            'F1-Macro': f"{metrics['f1_macro_mean']:.4f} ± {metrics['f1_macro_std']:.4f}",
            'Precision': f"{metrics['precision_macro_mean']:.4f} ± {metrics['precision_macro_std']:.4f}",
            'Recall': f"{metrics['recall_macro_mean']:.4f} ± {metrics['recall_macro_std']:.4f}",
            'Train Time (s)': f"{metrics['train_time']:.2f}",
            'F1_Score_Raw': metrics['f1_macro_mean']  # For sorting
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('F1_Score_Raw', ascending=False)
    df = df.drop('F1_Score_Raw', axis=1)

    # Save to CSV
    csv_path = str(output_path).replace('.txt', '.csv').replace('.pkl', '.csv')
    if not csv_path.endswith('.csv'):
        csv_path = str(output_path) + '_comparison.csv'

    df.to_csv(csv_path, index=False)
    print(f"\nComparison results saved to: {csv_path}")

    # Save justification if provided
    if justification:
        txt_path = str(output_path).replace('.csv', '.txt').replace('.pkl', '_justification.txt')
        if not txt_path.endswith('.txt'):
            txt_path = str(output_path) + '_justification.txt'

        with open(txt_path, 'w') as f:
            f.write(justification)

        print(f"Model justification saved to: {txt_path}")

    # Print comparison table
    print("\n" + "="*70)
    print("MODEL COMPARISON RESULTS")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)


def save_model(model: Any, output_path: str):
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

    print(f"\nBest model saved to: {output_path}")


def main():
    """Main function for model comparison."""
    import argparse

    parser = argparse.ArgumentParser(description='Compare ML models for cricket object detection')
    parser.add_argument('--features', type=str, required=True,
                       help='Path to training features (.npy)')
    parser.add_argument('--labels', type=str, required=True,
                       help='Path to training labels (.npy)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Directory for outputs')
    parser.add_argument('--model_path', type=str, default='models/cricket_model_best.pkl',
                       help='Path to save best model')
    parser.add_argument('--tune', action='store_true',
                       help='Perform hyperparameter tuning (slower)')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of cross-validation folds')

    args = parser.parse_args()

    # Load data
    print("Loading training data...")
    X_train = np.load(args.features)
    y_train = np.load(args.labels)

    print(f"Loaded X_train: {X_train.shape}, y_train: {y_train.shape}")

    # Compare models
    results = compare_models(
        X_train, y_train,
        tune_hyperparameters=args.tune,
        cv_folds=args.cv_folds,
        verbose=True
    )

    # Select best model
    best_key, best_model, justification = select_best_model(results)

    print(justification)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_comparison_results(results, output_dir / 'model_comparison.csv', justification)
    save_model(best_model, args.model_path)

    print("\n" + "="*70)
    print("MODEL COMPARISON COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
