"""
Configuration file for Cricket Object Detection Pipeline

This file contains constants used across multiple modules in the pipeline.
Module-specific constants (used in only one file) should remain in their respective modules.
"""

# ============================================================================
# PCA CONFIGURATION (used in train.py, train_compare_models.py, analyze_pca.py)
# ============================================================================

# Variance threshold for PCA dimensionality reduction
# Higher values preserve more information but result in more features
PCA_VARIANCE_THRESHOLD = 0.99  # 99% variance retention

# Random seed for reproducibility
RANDOM_STATE = 42


# ============================================================================
# CROSS-VALIDATION CONFIGURATION (used in train.py, train_compare_models.py)
# ============================================================================

# Number of folds for cross-validation
CV_FOLDS = 5

# Scoring metric for model selection
CV_SCORING_METRIC = 'f1_macro'  # Balanced metric for imbalanced data


# ============================================================================
# IMAGE AND GRID CONFIGURATION (used in grid_utils.py, feature_extractor.py)
# ============================================================================

# Image dimensions (resized if needed)
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600

# Grid dimensions for cell extraction
GRID_ROWS = 8
GRID_COLS = 8

# Derived cell dimensions
CELL_WIDTH = IMAGE_WIDTH // GRID_COLS  # 100 pixels
CELL_HEIGHT = IMAGE_HEIGHT // GRID_ROWS  # 75 pixels
TOTAL_CELLS = GRID_ROWS * GRID_COLS  # 64 cells


# ============================================================================
# CLASS LABELS (used across multiple modules)
# ============================================================================

# Number of classes
NUM_CLASSES = 4

# Class label mapping
CLASS_NO_OBJECT = 0
CLASS_BALL = 1
CLASS_BAT = 2
CLASS_STUMP = 3
