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

from sklearn.calibration import CalibratedClassifierCV
warnings.filterwarnings('ignore')

from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

import sys
import os
from mutual_info_selector import MutualInfoSelector

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'features'))

# Import configuration constants
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import PCA_VARIANCE_THRESHOLD, RANDOM_STATE, CV_FOLDS, CV_SCORING_METRIC

from sklearn.metrics import make_scorer, classification_report
import numpy as np
import collections

# CLASS_NAMES = ["NoObject", "Ball", "Bat", "Stump", "OtherNoObjects"]
CLASS_NAMES = ["NoObject", "Ball", "Bat", "Stump"]
BALL_BAT_STUMP = ["Ball", "Bat", "Stump", "NoObject"]
# BALL_BAT_STUMP = ["NoObject", "Ball", "Bat", "Stump", "OtherNoObjects"]
BALL_BAT_STUMP_IDS = [0, 1, 2, 3] # Corresponding IDs for Ball, Bat, Stump

def f1_3class(y_true, y_pred):
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
    return np.mean([report[c]["f1-score"] for c in BALL_BAT_STUMP])

def recall_3class(y_true, y_pred):
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
    return np.mean([report[c]["recall"] for c in BALL_BAT_STUMP])

def precision_3class(y_true, y_pred):
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
    return np.mean([report[c]["precision"] for c in BALL_BAT_STUMP])

def accuracy_3class_from_report(y_true, y_pred):
    """
    Computes accuracy ONLY for Ball, Bat, Stump using classification_report data.
    This ignores all NoObject samples.
    """
    mask = np.isin(y_true, BALL_BAT_STUMP_IDS)  # keep only Ball, Bat, Stump
    if mask.sum() == 0:
        return 0.0
    return (y_true[mask] == y_pred[mask]).mean()

SCORING = {
    "f1_macro": make_scorer(f1_3class),
    "recall_macro": make_scorer(recall_3class),
    "precision_macro": make_scorer(precision_3class),
    "accuracy": make_scorer(accuracy_3class_from_report)
}


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

        # Create pipeline with scaling and PCA (scale before PCA!)
        pipeline = Pipeline([
            # ('scaler', StandardScaler()),
            # ('pca', PCA(n_components=PCA_VARIANCE_THRESHOLD, random_state=RANDOM_STATE)),
            ('classifier', LogisticRegression(
                multi_class='multinomial',
                class_weight='balanced',
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ])

        # Grid search
        grid_search = GridSearchCV(
            pipeline, param_grid,
            cv=cv_folds,
            scoring=CV_SCORING_METRIC,
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
            # ('scaler', StandardScaler()),
            # ('pca', PCA(n_components=PCA_VARIANCE_THRESHOLD, random_state=RANDOM_STATE)),
            ('classifier', LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='lbfgs',
                multi_class='multinomial',
                class_weight='balanced',
                max_iter=1000,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ])

        pipeline.fit(X, y)
        best_pipeline = pipeline

    # Cross-validation evaluation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
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
                       verbose: bool = True) -> Tuple[Pipeline, Dict]:
    """
    Train Random Forest classifier with PCA preprocessing.

    Note: With 431 features, PCA is essential for Random Forest to:
    - Reduce dimensionality and prevent overfitting
    - Speed up training time
    - Improve generalization performance
    - StandardScaler is required before PCA for proper variance scaling

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
        print("TRAINING: Random Forest")
        print("="*70)

    start_time = time.time()

    if tune_hyperparameters:
        # Hyperparameter grid (from existing train.py)
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [15, 20, 25],
            'classifier__min_samples_split': [5, 10, 15],
            'classifier__min_samples_leaf': [2, 5, 10]
        }

        # Create pipeline with scaling and PCA
        pipeline = Pipeline([
            # ('scaler', StandardScaler()),
            # ('pca', PCA(n_components=PCA_VARIANCE_THRESHOLD, random_state=RANDOM_STATE)),
            ('classifier', RandomForestClassifier(
                class_weight='balanced',
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ])

        grid_search = GridSearchCV(
            pipeline, param_grid,
            cv=cv_folds,
            scoring=CV_SCORING_METRIC,
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
        # class_weights = {
        #     0: 1.0,   # NoObject
        #     1: 5.0,   # Ball
        #     2: 3.0,   # Bat
        #     3: 3.0,   # Stump
        #     4: 1.0    # Other No Objects 
        # }
        pipeline = Pipeline([
            # ('scaler', StandardScaler()),
            # ('pca', PCA(n_components=PCA_VARIANCE_THRESHOLD, random_state=RANDOM_STATE)),
            ('classifier', RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ])

        pipeline.fit(X, y)
        best_pipeline = pipeline

    # Cross-validation evaluation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(
        best_pipeline, X, y,
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

    return best_pipeline, metrics


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
            # ('scaler', StandardScaler()),
            # ('pca', PCA(n_components=PCA_VARIANCE_THRESHOLD, random_state=RANDOM_STATE)),
            ('classifier', SVC(
                class_weight='balanced',
                random_state=RANDOM_STATE,
                max_iter=5000
            ))
        ])

        grid_search = GridSearchCV(
            pipeline, param_grid,
            cv=cv_folds,
            scoring=CV_SCORING_METRIC,
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
        # class_weights = {
        #     0: 1.0,   # NoObject (majority)
        #     1: 8.0,   # Ball (rarest)
        #     2: 5.0,   # Bat
        #     3: 5.0,   # Stump
        #     4: 1.0    # Other No Objects 
        # }
        best_pipeline = Pipeline([
            # ('scaler', StandardScaler()),
            # ('pca', PCA(n_components=PCA_VARIANCE_THRESHOLD, random_state=RANDOM_STATE)),
            ('classifier', SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                class_weight='balanced',
                random_state=RANDOM_STATE,
                max_iter=5000
            ))])
        best_pipeline.fit(X, y)

    # Cross-validation evaluation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
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
            # ('scaler', StandardScaler()),
            # ('pca', PCA(n_components=PCA_VARIANCE_THRESHOLD, random_state=RANDOM_STATE)),
            ('classifier', KNeighborsClassifier(n_jobs=-1))
        ])

        grid_search = GridSearchCV(
            pipeline, param_grid,
            cv=cv_folds,
            scoring=CV_SCORING_METRIC,
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
        best_pipeline = Pipeline([
            # ('scaler', StandardScaler()),
            # ('pca', PCA(n_components=PCA_VARIANCE_THRESHOLD, random_state=RANDOM_STATE)),
            ('classifier', KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                metric='euclidean',
                n_jobs=-1
            ))])
        best_pipeline.fit(X, y)

    # Cross-validation evaluation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
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

def compute_base_class_weights(y_train, target_classes=None, normalize_mean_one=True):
    """
    Inverse-frequency weights per class.
    Returns dict: {class_label: weight}
    """
    counts = collections.Counter(y_train)
    if target_classes is None:
        target_classes = sorted(counts.keys())
    total = sum(counts[c] for c in target_classes)
    # inverse freq
    weights = {c: (total / counts[c]) for c in target_classes}
    if normalize_mean_one:
        mean_w = np.mean(list(weights.values()))
        weights = {c: float(weights[c] / mean_w) for c in weights}
    return weights

def sample_weights_from_class_weights(y, class_weights):
    # y can be numpy array or list
    y = np.asarray(y)
    return np.array([class_weights[int(cl)] for cl in y], dtype=float)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score
# import xgboost as xgb
import numpy as np

def evaluate_weight_candidates(X, y, classifier_params, class_weights_base,
                               candidate_multipliers_list,
                               cv_splits=4, random_state=42,
                               score_metric='macro_f1_no0'):
    """
    candidate_multipliers_list: list of dicts {class_label: multiplier}
    score_metric: 'macro_f1_no0' or 'mean_recall_no0' or 'macro_f1'
    returns list of (multipliers_dict, mean_score, std_score)
    """
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    results = []
    classes_all = sorted(class_weights_base.keys())

    for multipliers in candidate_multipliers_list:
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # build class weights for this candidate
            class_weights = {c: class_weights_base[c] * multipliers.get(c, 1.0) for c in classes_all}
            sw_tr = sample_weights_from_class_weights(y_tr, class_weights)

            # instantiate classifier
            clf = xgb.XGBClassifier(**classifier_params)

            # fit with sample_weight
            clf.fit(X_tr, y_tr, sample_weight=sw_tr, verbose=False)

            # predict on val
            y_pred = clf.predict(X_val)

            # metric focusing only on classes 1,2,3 (exclude NoObject)
            focus_labels = [1,2,3]
            if score_metric == 'macro_f1_no0':
                s = f1_score(y_val, y_pred, labels=focus_labels, average='macro', zero_division=0)
            elif score_metric == 'mean_recall_no0':
                s = recall_score(y_val, y_pred, labels=focus_labels, average='macro', zero_division=0)
            else:
                s = f1_score(y_val, y_pred, average='macro', zero_division=0)

            scores.append(s)

        results.append((multipliers, np.mean(scores), np.std(scores)))

    # sort by mean desc
    results.sort(key=lambda x: x[1], reverse=True)
    return results

from sklearn.metrics import f1_score

def tune_thresholds(clf, X_val, y_val, labels_to_tune=[1,2,3]):
    # get probabilities
    probs = clf.predict_proba(X_val)  # shape (n, n_classes)
    n_classes = probs.shape[1]

    # default thresholds
    thr = {i: 0.5 for i in range(n_classes)}

    best_thr = thr.copy()
    best_f1 = f1_score(y_val, clf.predict(X_val), average='macro', zero_division=0)

    # grid search per class (simple)
    for c in labels_to_tune:
        best_local = (0.5, -1.0)
        for t in np.linspace(0.1, 0.9, 17):
            # apply thresholds: pick class with highest (prob - (1-threshold))?? Simpler: choose class c if prob_c >= t; else argmax of others
            preds = []
            for p in probs:
                # if class c prob >= t choose c else argmax of probs
                if p[c] >= t:
                    preds.append(c)
                else:
                    preds.append(int(np.argmax(p)))
            val_f1 = f1_score(y_val, preds, average='macro', zero_division=0)
            if val_f1 > best_local[1]:
                best_local = (t, val_f1)
        best_thr[c] = best_local[0]
    return best_thr

def get_final_class_weights_for_xgboost(X: np.ndarray, y: np.ndarray) -> Dict[int, float]:
    """
    Generate all combinations of class weight multipliers.

    base_classes: list of class labels
    multipliers_options: list of possible multipliers per class

    Returns list of dicts {class_label: multiplier}
    """
    class_weights = {0: 0.25, 1: 11.0, 2: 6.0, 3: 4.3}
    sample_weights = np.array([class_weights[y_wt] for y_wt in y])
    from sklearn.feature_selection import VarianceThreshold

    # X: pandas.DataFrame (preferred)
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # 1.1 remove near-constant features
    vt = VarianceThreshold(threshold=1e-4)   # tune threshold if needed
    X_v = pd.DataFrame(vt.fit_transform(X), columns=X.columns[vt.get_support()])

    # 1.2 remove highly correlated features (keep one of each correlated block)
    corr = X_v.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
    X_v = X_v.drop(columns=to_drop)
    print("After variance+corr filter:", X_v.shape)

    from sklearn.feature_selection import mutual_info_classif

    mi = mutual_info_classif(X_v.values, y, discrete_features=False, random_state=42)
    mi_series = pd.Series(mi, index=X_v.columns).sort_values(ascending=False)
    mi_series.head(20)

    # keep top k by MI or fraction
    keep_frac = 0.6   # keep top 60% by MI
    k = max(50, int(len(mi_series) * keep_frac))
    keep_cols = mi_series.index[:k].tolist()
    X_f = X_v[keep_cols]
    print("After MI filter:", X_f.shape)

    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=4,
        n_estimators=800,
        max_depth=3,
        learning_rate=0.02,
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=4,
        min_child_weight=10,
        max_delta_step=2,
        eval_metric='mlogloss',
        tree_method='hist',
        n_jobs=-1
    )
    base_wts = compute_base_class_weights(y)
    print(base_wts)
    clf_params = {
        'objective':'multi:softprob',
        'num_class':4,
        'n_estimators':800,
        'max_depth':3,
        'learning_rate':0.02,
        'subsample':0.7,
        'colsample_bytree':0.7,
        'gamma':4,
        'min_child_weight':10,
        'max_delta_step':2,
        'eval_metric':'mlogloss',
        'tree_method':'hist',
        'use_label_encoder':False,
        'n_jobs':-1,
        'random_state':42
    }

    # create small candidate list (coarse)
    candidates = [
        {0:1.0, 1:1.5, 2:1.3, 3:1.1},
        {0:1.0, 1:2.0, 2:1.5, 3:1.2},
        {0:1.0, 1:3.0, 2:1.8, 3:1.2},
        {0:1.0, 1:2.5, 2:1.6, 3:1.3},
    ]

    results = evaluate_weight_candidates(
        X, y,
        classifier_params=clf_params,
        class_weights_base=base_wts,
        candidate_multipliers_list=candidates,
        cv_splits=4,
        score_metric='macro_f1_no0'
    )

    print("Candidates sorted (best first):")
    for mult, mean_score, std in results:
        print(mult, mean_score, std)

    # choose top multipliers -> final_wts map then fit on full train + early stopping:
    chosen_mult = results[0][0]   # or whichever you like
    chosen_mult = {0:1.0, 1:1.5, 2:1.3, 3:1.1}
    final_class_weights = {c: base_wts[c] * chosen_mult.get(c,1.0) for c in base_wts}
    sample_weights = sample_weights_from_class_weights(y, final_class_weights)
    # You can then override/multiply certain classes manually if you want stronger boost:
    # e.g. increase Ball and Bat a bit:
    class_multipliers = {0:1.0, 1:2.0, 2:1.5, 3:1.2}
    final_wts = {c: base_wts[c] * class_multipliers.get(c, 1.0) for c in base_wts}
    sample_weights = sample_weights_from_class_weights(y, final_wts)
    print("Final class weights:", final_class_weights)
    return final_class_weights, X_f, X_v

def get_calibratred_xgboost_model(X_train: np.ndarray, y_train: np.ndarray,
                                  X_val: np.ndarray, y_val: np.ndarray, model: Pipeline, cv_folds: int) -> Pipeline:
    final_class_weights, X_f, X_v = get_final_class_weights_for_xgboost(X_train, y_train)
    # 3.2 Get importances
    importances = model.get_booster().get_score(importance_type="gain")
    # map to DataFrame
    imp_df = pd.Series(importances).sort_values(ascending=False)
    # Note: get_score returns names like 'f0','f1' if X is array; with DataFrame columns match.

    # 3.3 Select top features automatically with threshold or top_k
    from sklearn.feature_selection import SelectFromModel

    # Option A: select using threshold (median importance)
    threshold = np.median(list(importances.values()))
    sfm = SelectFromModel(model, threshold=threshold, prefit=True)
    X_s = sfm.transform(X_f)
    selected_cols = X_f.columns[sfm.get_support()]
    print("Selected by model:", X_s.shape)

    # Option B: select top N
    N = 300
    top_k = imp_df.index[:N].tolist()
    # if keys are 'f0' etc, map to columns - easier to use feature_importances_ attribute
    fi = pd.Series(model.feature_importances_, index=X_f.columns).sort_values(ascending=False)
    top_k_cols = fi.index[:N].tolist()
    X_s = X_f[top_k_cols]
    print("Selected top_k:", X_s.shape)


    calibrated_model = CalibratedClassifierCV(
        base_estimator=pipeline,
        cv="prefit",          # <-- important!
        method="sigmoid"
    )
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        min_child_weight=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("calib", CalibratedClassifierCV(
            base_estimator=xgb_model,
            cv=5,
            method="sigmoid"
        ))
    ])

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("calib", CalibratedClassifierCV(
            estimator=xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                min_child_weight=5,
                gamma=2.0,
                reg_lambda=1.2,
                reg_alpha=0.4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1
            ),
            cv=5,
            method="sigmoid"
        ))
    ])

    calibrated_model.fit(X_val, y_val)
    pipeline.fit(X_train, y_train)
    best_pipeline = pipeline

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    cv_results_full = cross_validate(
        best_pipeline, X_v, y_train,
        cv=cv,
        scoring=SCORING,
        # fit_params={"classifier__sample_weight": sample_weights},
        # return_estimator=True,
        n_jobs=-1
    )

    cv_results_sel = cross_validate(
        best_pipeline, X_s, y_train,
        cv=cv,
        scoring=SCORING,
        # fit_params={"classifier__sample_weight": sample_weights},
        # return_estimator=True,
        n_jobs=-1
    )
    print("F1 macro, full:", cv_results_full['test_f1_macro'].mean(), " selected:", cv_results_sel['test_f1_macro'].mean())

def train_xgboost(X: np.ndarray, y: np.ndarray,
                 tune_hyperparameters: bool = True,
                 cv_folds: int = 5,
                 verbose: bool = True) -> Tuple[Pipeline, Dict]:
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
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__subsample': [0.8, 1.0],
            'classifier__colsample_bytree': [0.8, 1.0]
        }

        pipeline = Pipeline([
            # ('scaler', StandardScaler()),
            # ('pca', PCA(n_components=PCA_VARIANCE_THRESHOLD, random_state=RANDOM_STATE)),
            ('classifier', xgb.XGBClassifier(
                objective='multi:softmax',
                num_class=len(classes),
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric='mlogloss'
            ))
        ])

        grid_search = GridSearchCV(
            pipeline, param_grid,
            cv=cv_folds,
            scoring=CV_SCORING_METRIC,
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X, y, classifier__sample_weight=sample_weights)
        best_pipeline = grid_search.best_estimator_

        if verbose:
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV F1-macro: {grid_search.best_score_:.4f}")

    else:
        # class_weights = {
        #     0: 0.25,
        #     1: 11.0,
        #     2: 6.0,
        #     3: 4.3
        # }
        # def compute_sample_weights(y, class_weights, normalize_mean=True):
        #     # y : numpy array of integer labels
        #     w = np.array([class_weights[int(c)] for c in y], dtype=float)
        #     if normalize_mean:
        #         w = w / w.mean()         # keeps average loss scale stable (recommended)
        #     return w
        # sample_weights = compute_sample_weights(y, class_weights, normalize_mean=True)
        # sample_weights = compute_sample_weights(y, class_weights)

        # final_weights, _, _ = get_final_class_weights_for_xgboost(X, y)
        # sample_weights = sample_weights_from_class_weights(y, final_weights)
        model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                num_class=len(classes),
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric='aucpr'
            )
        pipeline = Pipeline([
            # ("mi_select", MutualInfoSelector(keep_frac=0.6)),  # <- MI filter inside pipeline
            ("classifier", model)
        ])

        pipeline.fit(X, y)
        # pipeline= get_calibratred_xgboost_model(X, y, X, y, pipeline, cv_folds)

       
        best_pipeline = pipeline
        
    # Cross-validation evaluation (without sample_weight for CV - XGBoost already trained with it)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(
        best_pipeline, X, y,
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

    return best_pipeline, metrics


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
