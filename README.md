# Cricket Object Detection

A machine learning project for detecting cricket objects (ball, bat, stumps) in images using hand-crafted feature engineering.

## Project Overview

This project implements a grid-based object detection system that:
- Divides cricket images (800×600) into 8×8 grids (64 cells of 100×75 pixels)
- Classifies each cell as: **0**=no object, **1**=ball, **2**=bat, **3**=stump
- Uses **ONLY taught hand-crafted features** (no CNNs or deep learning)
- Achieves robust detection through comprehensive feature engineering

## Features (88 Total)

### Color Features (28)
- **RGB Histograms (14)**: Channel statistics, ratios, color variance
- **HSV Features (14)**: Hue/saturation/value statistics, color bins, purity

### Texture Features (60)
- **LBP (10)**: Local binary patterns for texture discrimination
- **HOG (20)**: Histogram of oriented gradients for shape detection
- **Gabor Filters (20)**: Multi-scale texture and pattern analysis
- **GLCM (10)**: Gray-level co-occurrence matrix for texture statistics

## Project Structure

```
project/
├── dataset/
│   ├── train/
│   │   ├── images/          # Training images (800×600)
│   │   └── labels/          # Training labels CSV
│   └── test/
│       ├── images/          # Test images (800×600)
│       └── labels/          # Test labels CSV
│
├── src/
│   ├── features/
│   │   ├── rgb_features.py
│   │   ├── hsv_features.py
│   │   ├── lbp_features.py
│   │   ├── hog_features.py
│   │   ├── gabor_features.py
│   │   ├── glcm_features.py
│   │   └── feature_extractor.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   │
│   └── data_prep/
│       ├── grid_utils.py
│       ├── annotator.py
│       └── csv_label_converter.py  # Convert row-column-tag to cell format
│
├── models/                   # Saved models
├── outputs/                  # Predictions and metrics
├── main_pipeline.py         # Main execution script
├── requirements.txt
└── README.md
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Verify installation:
```bash
python -c "import cv2, numpy, sklearn, skimage; print('All dependencies installed!')"
```

## Usage

### 1. Prepare Labels

You have two options for creating label files:

#### Option A: Use the Interactive Annotator

Use the annotation tool to label grid cells in your images (0=no object, 1=ball, 2=bat, 3=stump):

```bash
python src/data_prep/annotator.py --image_dir dataset/train/images --output_csv dataset/train/labels/labels.csv
```

**Expected Output:** `dataset/train/labels/labels.csv` with format: `ImageFileName,c01,c02,...,c64`

**Controls:**
- Click cells to label them
- Press `0`, `1`, `2`, `3` to select class (no object, ball, bat, stump)
- Press `n` to save and move to next image
- Press `s` to save current image

#### Option B: Convert Existing Labels

If you already have labels in row-column-tag format, convert them using the CSV label converter:

```bash
python src/data_prep/csv_label_converter.py -i <input_csv> -o <output_csv>
```

**Example:**
```bash
python src/data_prep/csv_label_converter.py -i dataset/train/labels/atul_tagged_data.csv -o dataset/train/labels/labels-atul.csv
```

This converts from row-column-tag format:
```csv
imagename,cell_1_row,cell_1_column,cell_1_object_tag,cell_2_row,cell_2_column,cell_2_object_tag,...
ball_img_2439.jpg,1,4,Stump,1,5,Stump,1,7,Bat,1,8,Bat
```

To cell-based format:
```csv
ImageFileName,c01,c02,c03,c04,c05,c06,c07,c08,...,c64
ball_img_2439.jpg,0,0,0,3,3,0,2,2,...,0
```

**Object Mapping:** Ball→1, Bat→2, Stump→3, No object→0


### 2. Run Complete Pipeline

Extract features, train model, evaluate, and generate predictions:

```bash
python main_pipeline.py --mode full --tune
```

**With Visualization** (generates color-coded grid overlays on both train and test images):

```bash
python main_pipeline.py --mode full --tune --visualize
```

**Note**: Undersampling is **enabled by default** to handle class imbalance. To disable:

```bash
# Disable undersampling (use original imbalanced data)
python main_pipeline.py --mode full --no-undersample

# Change undersampling strategy (default is image_stratified)
python main_pipeline.py --mode full --undersample_strategy random

# Combined: undersampling + tuning + visualization
python main_pipeline.py --mode full --tune --visualize
```

### 3. Individual Pipeline Steps

**Extract features only:**
```bash
python main_pipeline.py --mode extract
```
**Expected Output:** `outputs/X_train.npy`, `outputs/y_train.npy`, `outputs/X_test.npy`, `outputs/y_test.npy`

**Train model only (with hyperparameter tuning):**
```bash
python main_pipeline.py --mode train --tune  --compare_models
```
**Expected Output:** `models/cricket_model.pkl` (trained Random Forest model with optimal hyperparameters)

**Evaluate model:**
```bash
python main_pipeline.py --mode evaluate
```
**Expected Output:** `outputs/metrics.txt`, `outputs/confusion_matrix.png`

**Generate predictions:**
```bash
python main_pipeline.py --mode predict

# With visualization
python main_pipeline.py --mode predict --visualize
```
**Expected Output:** `outputs/predictions.csv` (format: `ImageFileName,TrainOrTest,c01,...,c64`)

**Predict with custom paths and visualization:**
```bash
python main_pipeline.py --mode predict --model_path models/cricket_model.pkl --train_images dataset/train/images --test_images dataset/test/images --output_dir outputs --visualize
```
**Expected Output:** `outputs/test/predictions.csv` + `outputs/train/visualizations/*.png` + `outputs/test/visualizations/*.png` (color-coded grids)

### 4. Custom Paths

```bash
python main_pipeline.py \
    --mode full \
    --train_images dataset/train/images \
    --train_labels dataset/train/labels/labels.csv \
    --test_images dataset/test/images \
    --test_labels dataset/test/labels/labels.csv \
    --model_path models/my_model.pkl \
    --output_dir outputs \
    --tune
```

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Pipeline mode: `extract`, `train`, `evaluate`, `predict`, `eda`, `full` (Note: `full` mode automatically runs EDA) | Required |
| `--train_images` | Training images directory | `dataset/train/images` |
| `--test_images` | Test images directory | `dataset/test/images` |
| `--train_labels` | Training labels CSV | `dataset/train/labels/labels.csv` |
| `--test_labels` | Test labels CSV | `dataset/test/labels/labels.csv` |
| `--model_path` | Model save/load path | `models/cricket_model.pkl` |
| `--output_dir` | Output directory | `outputs` |
| `--tune` | Enable hyperparameter tuning | Flag (off by default) |
| `--visualize` | Generate visualization images for train and test | Flag (off by default) |
| `--no-undersample` | Disable undersampling (use original imbalanced data) | Flag (undersampling is **ON by default**) |
| `--undersample_strategy` | Undersampling strategy: `image_stratified` or `random` | `image_stratified` |

## Output Files

After running the pipeline, you'll find:

- `outputs/X_train.npy` - Training features
- `outputs/y_train.npy` - Training labels
- `outputs/X_test.npy` - Test features
- `outputs/y_test.npy` - Test labels
- `models/cricket_model.pkl` - Trained model
- `outputs/predictions.csv` - Predictions in required format
- `outputs/metrics.txt` - Evaluation metrics
- `outputs/confusion_matrix.png` - Confusion matrix visualization
- `outputs/train/visualizations/*.png` - Training images with colored prediction grids (if `--visualize` flag used)
- `outputs/test/visualizations/*.png` - Test images with colored prediction grids (if `--visualize` flag used)
- `outputs/undersampling_report.csv` - Detailed undersampling statistics (if `--undersample` flag used)

## CSV Format

### Labels CSV Format
```
ImageFileName,c01,c02,c03,...,c64
image001.png,0,0,1,0,...,2
image002.png,0,0,0,3,...,0
```

### Predictions CSV Format
```
ImageFileName,TrainOrTest,c01,c02,c03,...,c64
image001.png,train,0,0,1,0,...,2
image002.png,test,0,0,0,3,...,0
```

## Feature Extraction Details

### RGB Features (14)
Captures raw color information essential for detecting:
- Red cricket ball (high R channel)
- Brown bat (R≈G > B)
- Green grass background (high G channel)
- White stumps (R≈G≈B, all high)

### HSV Features (14)
Lighting-invariant color detection:
- Ball detection robust to shadows
- Hue-based color discrimination
- Saturation for vivid vs dull colors

### LBP Features (10)
Texture discrimination:
- Smooth ball/bat vs textured grass
- Rotation-invariant patterns

### HOG Features (20)
Shape and edge detection:
- Vertical stumps (90° gradients)
- Horizontal bat (0°, 180° gradients)
- Circular ball (uniform gradient distribution)

### Gabor Features (20)
Multi-scale texture analysis:
- 5 scales × 4 orientations
- Circular vs elongated pattern detection

### GLCM Features (10)
Texture statistics:
- Smooth objects: high homogeneity, low contrast
- Textured grass: low homogeneity, high contrast

## Visualization Feature

The pipeline includes an optional visualization feature that generates both train and test images with color-coded grid overlays showing model predictions.

### How to Use

Add the `--visualize` flag to any pipeline command:

```bash
python main_pipeline.py --mode full --visualize
python main_pipeline.py --mode predict --visualize
```

### Color Scheme

Each grid cell is colored based on its predicted class:

| Class | Object | Color | BGR Value |
|-------|--------|-------|-----------|
| 1 | Ball | Green | `(0, 255, 0)` |
| 2 | Bat | Red | `(0, 0, 255)` |
| 3 | Stump | Blue | `(255, 0, 0)` |

Cells with class 0 (no object) remain uncolored, showing the original image.

### Output Structure

Visualized images are automatically organized into separate directories:

```
outputs/
├── train/
│   └── visualizations/
│       ├── train_image001.png
│       ├── train_image002.png
│       └── ...
└── test/
    └── visualizations/
        ├── test_image001.png
        ├── test_image002.png
        └── ...
```

Each visualized image includes:
- Semi-transparent colored fills for predicted object cells (ball/bat/stump)
- Dark gray grid lines showing cell boundaries (8×8 grid)
- Cell labels (c01, c02, ..., c64) in the top-left corner of each cell
- Original image visible underneath

### Benefits

- **Visual Debugging**: Quickly see where model makes errors on both train and test sets
- **Overfitting Detection**: Compare train vs test predictions visually
- **Error Analysis**: Identify systematic misclassifications by class/position
- **Presentation**: Use visualized images in slides/reports/video
- **Quality Check**: Verify predictions without parsing CSV
- **Annotation Verification**: Check if training labels are correct by comparing with predictions

## Class Imbalance Handling (Undersampling)

The dataset suffers from severe class imbalance:
- **Class 0 (no object)**: 87% of samples
- **Classes 1, 2, 3 (ball, bat, stump)**: Only 13% combined

This imbalance can cause the model to be biased toward predicting "no object" and perform poorly on minority classes (the actual cricket objects we want to detect).

### Undersampling Solution

The pipeline includes undersampling that balances the class distribution by reducing Class 0 samples to match the largest minority class count, while preserving all minority class samples.

**Note**: Undersampling is **enabled by default** with image-stratified strategy.

### How to Use

```bash
# Default behavior: undersampling is automatically applied
python main_pipeline.py --mode full

# Disable undersampling (use original imbalanced data)
python main_pipeline.py --mode full --no-undersample

# Use random undersampling strategy instead
python main_pipeline.py --mode full --undersample_strategy random
```

### Undersampling Strategies

#### Image-Stratified (Recommended)
- Samples Class 0 cells **evenly from all images**
- Preserves spatial distribution and diversity
- Maintains representation of different backgrounds, lighting, angles
- Better generalization to test data

#### Random
- Randomly samples Class 0 cells from entire dataset
- Simpler and faster implementation
- May result in some images having no Class 0 representation
- Good for quick experiments

### Effect on Data

**Before Undersampling** (~13,248 samples):
- Class 0: 11,574 samples (87.37%)
- Class 1: 493 samples (3.72%)
- Class 2: 492 samples (3.71%)
- Class 3: 689 samples (5.20%)

**After Undersampling** (~2,363 samples):
- Class 0: 689 samples (29.16%) ← Reduced by 94%
- Class 1: 493 samples (20.86%) ← Unchanged
- Class 2: 492 samples (20.82%) ← Unchanged
- Class 3: 689 samples (29.16%) ← Unchanged

**Data Reduction**: 82% fewer samples (13,248 → 2,363)

### Expected Benefits

1. **Improved Minority Class Detection**:
   - Ball recall: +10% improvement
   - Bat recall: +10% improvement
   - Stump recall: +10% improvement

2. **Better F1-Macro Score**:
   - Expected improvement: +7-10%
   - From ~0.75-0.80 to ~0.82-0.88

3. **Faster Training**:
   - Training time: 80% reduction (5-10 min → 1-2 min)
   - Tuning time: 78% reduction (30-45 min → 6-10 min)

4. **Reduced Model Bias**:
   - Less tendency to predict "no object" for everything
   - Better decision boundaries for minority classes

### Trade-offs

- **Overall Accuracy**: May drop by ~5% (acceptable, as accuracy is misleading for imbalanced data)
- **Class 0 Precision**: May drop by 3-5% (from 95-97% to 90-93%)
- **Information Loss**: 10,885 Class 0 samples discarded (but these are redundant background samples)

### Output

When undersampling is enabled, the pipeline generates:
- `outputs/undersampling_report.csv` - Detailed statistics showing:
  - Original vs undersampled counts per class
  - Reduction amounts and percentages
  - Final class distribution percentages

### When to Disable Undersampling

Undersampling is **enabled by default** because it significantly improves minority class detection. However, you may want to disable it (`--no-undersample`) when:

- Class 0 (no object) detection accuracy is critical for your use case
- You need to compare performance with/without undersampling for analysis
- You want to establish a baseline with original imbalanced data
- Your model already achieves >85% recall on all minority classes without undersampling

## Model Training

The pipeline uses **Random Forest Classifier** with:
- Balanced class weights (handles imbalanced data)
- 5-fold cross-validation
- Optional grid search for hyperparameter tuning

**Hyperparameter Grid:**
- `n_estimators`: [100, 200, 300]
- `max_depth`: [15, 20, 25]
- `min_samples_split`: [5, 10, 15]
- `min_samples_leaf`: [2, 5, 10]

**Training Time:**
- Without tuning: ~2-5 minutes (70 images)
- With tuning: ~15-30 minutes (70 images)

## Troubleshooting

### ImportError: No module named 'cv2'
```bash
pip install opencv-python
```

### Image size mismatch
The pipeline automatically resizes images to 800×600. Ensure your images are at least 800×600 to avoid upscaling.

### Memory issues
If processing many images, extract features in batches:
```python
# Process 50 images at a time
for i in range(0, len(images), 50):
    batch = images[i:i+50]
    # Extract features for batch
```

### Low accuracy
1. Check annotation quality (are labels correct?)
2. Enable hyperparameter tuning: `--tune`
3. Increase training data
4. Check class balance in labels

## Development

### Testing Feature Extraction
```bash
cd src/features
python feature_extractor.py
```

### Testing Individual Features
```bash
cd src/features
python rgb_features.py
python hsv_features.py
# etc.
```

---

# Binary Model Pipeline

An advanced approach using **three separate binary classifiers** to improve minority class detection.

## Overview

The standard multi-class approach sometimes struggles with minority classes (Ball, Bat, Stump). The binary model pipeline addresses this by training three independent classifiers:

1. **Ball vs Rest** (Ball vs Not-Ball)
2. **Bat vs Rest** (Bat vs Not-Bat)
3. **Stump vs Rest** (Stump vs Not-Stump)

### Key Improvements

- **Independent threshold tuning** per object type
- **Aggressive class weighting** optimized for each binary task
- **Specialized feature importance** per object

## Quick Start

### 1. Label Format (Two Options)

The binary pipeline **auto-detects** label format and works with **BOTH**:

**Option A: Use existing single-label CSV directly (easiest)**
```bash
# No conversion needed! The pipeline auto-converts on-the-fly
python main_pipeline_binary.py --mode full --algorithm both --tune
```

**Option B: Convert to multi-label CSV first (optional)**
```bash
# Convert training labels (only if you want multi-label annotations later)
python src/data_prep/convert_singlelabel_to_multilabel.py \
    --input dataset/train/labels/labels.csv \
    --output dataset/train/labels/labels_multilabel.csv

# Convert test labels
python src/data_prep/convert_singlelabel_to_multilabel.py \
    --input dataset/test/labels/labels.csv \
    --output dataset/test/labels/labels_multilabel.csv

# Then use the multi-label files
python main_pipeline_binary.py --mode full --algorithm both --tune \
    --train_labels dataset/train/labels/labels_multilabel.csv \
    --test_labels dataset/test/labels/labels_multilabel.csv
```

**Format Auto-Detection:**
- **Single-label (64 columns):** `c01, c02, ..., c64` → Auto-converts to binary
- **Multi-label (196 columns):** `c01_ball, c01_bat, c01_stump, ...` → Uses directly

### 2. Train Binary Models

```bash
# Train both Random Forest and XGBoost binary models
python main_pipeline_binary.py --mode full --algorithm both --tune

# Train only Random Forest (faster)
python main_pipeline_binary.py --mode full --algorithm random_forest --tune

# Train only XGBoost
python main_pipeline_binary.py --mode full --algorithm xgboost --tune
```

### 3. Compare Models

Compare multi-class vs binary models:

```bash
python compare_all_models.py
```

## Binary Pipeline Modes

### Extract Features
```bash
python main_pipeline_binary.py --mode extract
```

**Output:** `outputs/binary_models/X_train.npy`, `outputs/binary_models/y_ball_train.npy`, etc.

### Train Models
```bash
# Train with hyperparameter tuning
python main_pipeline_binary.py --mode train --algorithm both --tune

# Train without tuning (faster)
python main_pipeline_binary.py --mode train --algorithm random_forest
```

**Output:** `models/binary_models/binary_{algorithm}_{class}.pkl` (6 models total for both algorithms)

### Evaluate Models
```bash
python main_pipeline_binary.py --mode evaluate --algorithm both
```

**Output:**
- `outputs/binary_models/{algorithm}/binary_metrics_independent.txt` - Per-binary-model metrics
- `outputs/binary_models/{algorithm}/binary_metrics_final.txt` - Final multi-class metrics
- `outputs/binary_models/{algorithm}/binary_confusion_matrices.png` - Visualization

### Predict
```bash
python main_pipeline_binary.py --mode predict --algorithm random_forest --visualize
```

**Output:** `outputs/binary_models/{algorithm}/predictions.csv` + visualizations

### Full Pipeline
```bash
python main_pipeline_binary.py --mode full --algorithm both --tune --visualize
```

Runs all steps: extract → train → evaluate → predict

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Pipeline mode: `extract`, `train`, `evaluate`, `predict`, `full` | Required |
| `--algorithm` | Algorithm: `random_forest`, `xgboost`, `both` | `random_forest` |
| `--train_images` | Training images directory | `dataset/train/images` |
| `--test_images` | Test images directory | `dataset/test/images` |
| `--train_labels` | Training labels CSV (**auto-detects** single-label or multi-label format) | `dataset/train/labels/labels.csv` |
| `--test_labels` | Test labels CSV (**auto-detects** format) | `dataset/test/labels/labels.csv` |
| `--model_dir` | Binary models directory | `models/binary_models` |
| `--output_dir` | Output directory | `outputs/binary_models` |
| `--tune` | Enable hyperparameter tuning | Flag (off by default) |
| `--visualize` | Generate color-coded grids | Flag (off by default) |

## Output Files

After running the binary pipeline:

```
models/binary_models/
├── binary_random_forest_ball.pkl
├── binary_random_forest_bat.pkl
├── binary_random_forest_stump.pkl
├── binary_xgboost_ball.pkl
├── binary_xgboost_bat.pkl
└── binary_xgboost_stump.pkl

outputs/binary_models/
├── X_train.npy, y_ball_train.npy, y_bat_train.npy, y_stump_train.npy
├── X_test.npy, y_ball_test.npy, y_bat_test.npy, y_stump_test.npy
├── y_train_multiclass.npy, y_test_multiclass.npy
├── random_forest/
│   ├── binary_metrics_independent.txt
│   ├── binary_metrics_final.txt
│   ├── binary_confusion_matrices.png
│   ├── binary_confusion_matrix_final.png
│   └── predictions.csv
└── xgboost/
    └── (same structure)

outputs/model_comparison_report.txt  # From compare_all_models.py
```

## How It Works

### 1. Independent Binary Classification

Each binary classifier focuses on one object type:

```python
# Ball model learns: "Is this a ball?" (yes/no)
ball_model.fit(X_train, y_ball_train)

# Bat model learns: "Is this a bat?" (yes/no)
bat_model.fit(X_train, y_bat_train)

# Stump model learns: "Is this a stump?" (yes/no)
stump_model.fit(X_train, y_stump_train)
```

### 2. Aggressive Class Weighting

Each model uses class weights optimized for its imbalance ratio:

- **Ball:** `{0: 1, 1: 50}` - Heaviest weight (rarest class ~5%)
- **Bat:** `{0: 1, 1: 30}` - Medium weight (~10%)
- **Stump:** `{0: 1, 1: 25}` - Medium weight (~8%)

### 3. Conflict Resolution

When predicting, each model outputs a probability:

```python
ball_prob = ball_model.predict_proba(X_test)[:, 1]
bat_prob = bat_model.predict_proba(X_test)[:, 1]
stump_prob = stump_model.predict_proba(X_test)[:, 1]

# Apply thresholds
ball_pred = (ball_prob >= threshold_ball)
bat_pred = (bat_prob >= threshold_bat)
stump_pred = (stump_prob >= threshold_stump)

# Resolve conflicts
if sum(predictions) == 0:
    final_class = 0  # No Object
elif sum(predictions) == 1:
    final_class = single_predicted_class  # 1, 2, or 3
else:  # Multiple predictions (conflict)
    final_class = class_with_highest_probability
```

### 4. Threshold Optimization (Optional)

Fine-tune thresholds on validation set:

```python
from threshold_tuning import optimize_thresholds_all_models

optimal_thresholds = optimize_thresholds_all_models(
    models, X_val, y_ball_val, y_bat_val, y_stump_val,
    metric='f1'
)
# Returns: {'ball': 0.6, 'bat': 0.4, 'stump': 0.4}
```

## Model Comparison

Use `compare_all_models.py` to generate comprehensive comparison:

```bash
python compare_all_models.py
```

**Output Report Includes:**
- Overall metrics (accuracy, F1-macro, F1-weighted)
- Per-class metrics (precision, recall, F1) for each class
- Best model recommendations by metric
- Improvement analysis (binary vs multi-class)

**Example Output:**
```
Model                Accuracy    F1-Macro    F1-Weighted
------------------------------------------------------------
Multi-Class RF         0.7917      0.4572      0.8418
Binary RF              0.8250      0.5750      0.8650
Binary XGBoost         0.8310      0.5920      0.8720

RECOMMENDED MODEL: Binary XGBoost
  F1-Macro Score: 0.5920
```

## Advanced: Multi-Label Annotation Tool

For future improvements, annotate cells with multiple objects:

```bash
python src/data_prep/annotator_multilabel.py \
    --image_dir dataset/train/images \
    --output_csv dataset/train/labels/labels_multilabel.csv
```

**Features:**
- 3 independent checkboxes per cell (Ball, Bat, Stump)
- Allows cells with multiple objects (e.g., ball + bat)
- Saves to 196-column format (64 cells × 3 labels)
- Color-coded visual feedback

**Controls:**
- Click cell to select
- `b` = Toggle Ball (Red highlight)
- `t` = Toggle Bat (Brown highlight)
- `s` = Toggle Stump (Yellow highlight)
- `c` = Clear all labels for cell
- `d` = Deselect cell
- `n` = Save and next image

## When to Use Binary Models

**Use Binary Pipeline When:**
- Multi-class baseline shows poor minority class recall (<60%)
- Precision on Bat/Stump is very low (<20%)
- Class imbalance is severe (>85% "No Object")
- Need fine-grained control over precision-recall tradeoff

**Stick with Multi-Class When:**
- Baseline already achieves >85% recall on all classes
- Training time is critical (binary pipeline takes ~2x longer)
- Simplicity is preferred over maximum accuracy

## Training Time Comparison

| Pipeline | Mode | Time (no tuning) | Time (with tuning) |
|----------|------|------------------|-------------------|
| Multi-Class | Full | ~5-10 min | ~30-45 min |
| Binary (RF only) | Full | ~8-15 min | ~45-70 min |
| Binary (both) | Full | ~15-25 min | ~90-135 min |

**Hardware:** Times assume modern CPU (8+ cores) and 16GB+ RAM

## Further Reading

- **Design Decision:** See [specs/memory.md](specs/memory.md) for detailed rationale
- **Label Conversion:** See [specs/CSV_LABEL_CONVERTER.md](specs/CSV_LABEL_CONVERTER.md)
- **Implementation Details:** See source files in `src/models/binary_*.py`

---

# Exploratory Data Analysis (EDA)

The project includes comprehensive EDA tools to analyze extracted features, visualize distributions, detect patterns, and evaluate multiple model configurations.

## Overview

The EDA pipeline provides:
- **Data quality checks** (missing values, duplicates)
- **Descriptive statistics** (Excel reports)
- **Distribution visualizations** (histograms, boxplots)
- **Correlation analysis** (R² heatmap)
- **Class balancing** (undersampling analysis)
- **Dimensionality reduction** (PCA with variance plots)
- **Model evaluation** (16 different model configurations with ROC curves and decision boundaries)

## Quick Start

### Standalone EDA Script

Run EDA on pre-extracted features:

```bash
# Default paths
python src/data_prep/project_eda.py

# Custom paths
python src/data_prep/project_eda.py \
    --train_features outputs/train_features.csv \
    --validate_features outputs/validate_features.csv \
    --output_dir eda_reports
```

### Integrated with Main Pipeline

Run EDA as part of the main pipeline:

```bash
# Run full pipeline (EDA runs automatically after feature extraction)
python main_pipeline.py --mode full

# Run EDA only (requires pre-extracted features)
python main_pipeline.py --mode eda

# Run extract + EDA (no training)
python main_pipeline.py --mode extract
python main_pipeline.py --mode eda
```

## EDA Workflow

### 1. Data Quality Checks

**Missing Data Detection:**
- Reports missing values per column and row
- Automatically imputes missing numeric values with column mean
- Skips non-numeric columns

**Duplicate Detection:**
- Identifies duplicate rows
- Detects duplicate columns (same data, different names)
- Reports only (does not remove)

### 2. Descriptive Analysis

Generates Excel spreadsheet (`descriptive_analysis.xlsx`) with:
- **Numeric features:** count, mean, std, min, 25%, 50%, 75%, max, missing count/percentage, unique values
- **Categorical features:** count, unique, top, freq, missing count/percentage
- **Excluded columns:** Lists metadata columns (label, image_name, cell_index)

### 3. Distribution Visualizations

**Feature Histograms** (`feature_histograms.png`):
- Grid of histograms (8 columns × N rows)
- 30 bins per histogram
- Auto-scales for readability

**Feature Boxplots** (`feature_boxplots.png`):
- Grid of boxplots (8 columns × N rows)
- Shows outliers and quartiles
- Helps identify skewed distributions

**Correlation Heatmap** (`feature_correlation_heatmap_r2.png`):
- R² correlation matrix (squared Pearson correlation)
- Color-coded heatmap with values
- Identifies highly correlated feature pairs

### 4. Class Balance Analysis

**Original Distribution:**
- Prints class counts and percentages
- Visualizes class imbalance

**Undersampling:**
- Reduces majority class (class 0) to 2000 samples
- Preserves all minority class samples
- Generates balanced visualizations in separate directory (`eda_reports_bal/`)

**Balanced Distribution:**
- Shows improved class balance
- Compares before/after statistics

### 5. Feature Standardization

**Z-Score Normalization:**
- Standardizes all numeric features (mean=0, std=1)
- Required for PCA and distance-based models
- Preserves feature relationships

### 6. Principal Component Analysis (PCA)

**PCA with 29 Components (default):**
- Reduces dimensionality while retaining variance
- Generates comprehensive outputs

**PCA Outputs:**
- `pca_explained_variance.png` - Scree plot showing variance per component
- `pca_cumulative_variance.png` - Cumulative variance plot
- `pca_loadings.csv` - Feature loadings for each principal component
- `pca_transformed.csv` - Transformed data (29 PCs + label)
- `pca_variance_summary.xlsx` - Excel with variance tables

**Test Set Transformation:**
- Applies same scaler + PCA from training data
- Ensures consistent feature space for evaluation

### 7. Model Evaluation (16 Models)

**Model Configurations:**
- **1 Logistic Regression** (max_iter=1000, multinomial)
- **4 K-Nearest Neighbors** (k=3, 5, 7, 11)
- **5 Random Forests** (varying depth and leaf size)
- **5 Neural Networks** (varying layer configurations)

**Evaluation on PCA Features:**
- Trains all 16 models on 29 PCA components
- Generates metrics Excel with per-class precision/recall/F1/AUC
- Creates visualization with decision boundaries + ROC curves

**Evaluation on Non-PCA Features:**
- Trains same 16 models on standardized original features
- Compares performance with/without PCA
- Helps determine if dimensionality reduction is beneficial

**Outputs:**
- `model_metrics_summary.xlsx` - PCA evaluation metrics
- `model_boundaries_and_roc.png` - PCA visualizations (decision boundaries + ROC)
- `nonpca_model_metrics_summary.xlsx` - Non-PCA evaluation metrics
- `nonpca_model_boundaries_and_roc.png` - Non-PCA visualizations

## Command-Line Arguments

### Standalone EDA Script

| Argument | Description | Default |
|----------|-------------|---------|
| `--train_features` | Path to training features CSV | `outputs/train_features.csv` |
| `--validate_features` | Path to validation features CSV | `outputs/validate_features.csv` |
| `--output_dir` | Directory for EDA outputs | `eda_reports` |

### Main Pipeline Integration

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode full` | Full pipeline (includes automatic EDA after feature extraction) | Mode option |
| `--mode eda` | Run EDA only (requires pre-extracted features) | Mode option |

## Output Structure

After running EDA, you'll find:

```
eda_reports/
├── descriptive_analysis.xlsx               # Numeric/categorical stats
├── feature_histograms.png                  # Distribution histograms
├── feature_boxplots.png                    # Boxplot grid
├── feature_correlation_heatmap_r2.png      # R² correlation matrix
├── pca_explained_variance.png              # PCA scree plot
├── pca_cumulative_variance.png             # Cumulative variance
├── pca_loadings.csv                        # Feature loadings
├── pca_transformed.csv                     # PCA-transformed data
├── pca_variance_summary.xlsx               # PCA variance tables
├── model_metrics_summary.xlsx              # PCA model evaluation
├── model_boundaries_and_roc.png            # PCA visualization
├── nonpca_model_metrics_summary.xlsx       # Non-PCA model evaluation
├── nonpca_model_boundaries_and_roc.png     # Non-PCA visualization
└── eda_reports_bal/                        # Balanced data visuals
    ├── feature_histograms.png
    ├── feature_boxplots.png
    └── feature_correlation_heatmap_r2.png
```

## Usage Examples

### Example 1: Quick EDA on Main Pipeline

```bash
# Extract features first
python main_pipeline.py --mode extract

# Run EDA
python main_pipeline.py --mode eda
```

### Example 2: Full Pipeline with EDA

```bash
# Extract → EDA → Train → Evaluate → Predict (EDA runs automatically)
python main_pipeline.py --mode full --tune
```

### Example 3: Standalone EDA with Custom Paths

```bash
python src/data_prep/project_eda.py \
    --train_features outputs/train_features.csv \
    --validate_features outputs/validate_features.csv \
    --output_dir eda_reports_custom
```

### Example 4: EDA on Binary Model Features

```bash
# Extract binary features first
python main_pipeline_binary.py --mode extract --algorithm ball

# Run standalone EDA on binary features
python src/data_prep/project_eda.py \
    --train_features outputs/binary_models/train_ball_features.csv \
    --validate_features outputs/binary_models/validate_ball_features.csv \
    --output_dir eda_reports_ball
```

## Troubleshooting

### Missing CSV Files

```
ERROR: Training features not found at outputs/binary_models/train_ball_features.csv
Please run: python main_pipeline_binary.py --mode extract
```

**Solution:** Run feature extraction first

### Memory Errors

If processing large datasets (>50,000 samples):
- Reduce PCA components
- Skip model evaluation section
- Process features in batches

### Slow Model Evaluation

The 16-model evaluation takes ~10-15 minutes. To speed up:
- Reduce number of models in `get_classification_models()`
- Skip non-PCA evaluation
- Use smaller PCA component count


---

# Image Crop & Scale Tool

A command-line Python tool that:

* Detects real image formats (JPEG/PNG/WEBP/GIF/BMP/AVIF/HEIC)
* Converts **AVIF/HEIC → PNG**
* Crops images to **4:3 aspect ratio** (center crop)
* Resizes them to **800 × 600 pixels**
* Saves results into a new **scaled/** directory
* Preserves original format for all other image types

---

## 🚀 Features

### ✔ Format Detection (Magic Number Based)

Works even when files have incorrect extensions.

### ✔ AVIF/HEIC Support

`pillow-avif-plugin` enables Pillow to decode AVIF & HEIC files.

### ✔ Intelligent Cropping

Images are center-cropped to **4:3** without distortion.

### ✔ Scaling

All cropped images are resized to **800×600** using high-quality LANCZOS filtering.

### ✔ Output Structure

Images are written to a new directory:

```
scaled/
```

Maintains original file extensions, except:

* **AVIF → PNG**

### ✔ Safe Processing

* Skips non-image files
* Handles corrupted/truncated images
* Converts modes safely (e.g., RGBA → RGB for JPG)

---

## 📦 Requirements

Create a `requirements.txt` with:

```
Pillow>=10.0.0
pillow-avif-plugin>=1.4.3
```

Install:

```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
image_scaler.py
requirements.txt
README.md
filtered/             ← Input folder (example)
scaled/               ← Output folder (auto-created)
```

---

## 🛠 Usage (Command Line)

### Basic usage:

```bash
python image_scaler.py filtered
```

### With a custom output folder:

```bash
python image_scaler.py filtered --output final_scaled
```

or shorter:

```bash
python image_scaler.py filtered -o final_scaled
```

---

## 📝 Script Behavior

### 🖼 Supported Formats (input)

| Format     | Supported | Notes                                    |
| ---------- | --------- | ---------------------------------------- |
| JPG / JPEG | ✔         | Preserved as JPG                         |
| PNG        | ✔         | Preserved as PNG (supports transparency) |
| WEBP       | ✔         | Preserved as WEBP                        |
| GIF        | ✔         | Static frame saved                       |
| BMP        | ✔         | Preserved as BMP                         |
| AVIF       | ✔         | Converted to PNG                         |
| HEIC/HEIF  | ✔         | Converted to PNG                         |

---

## 🔧 Processing Steps

For each image:

1. Detects real file format via signature
2. Opens image with Pillow
3. Converts mode (RGB/RGBA) if needed
4. Performs a **center crop** to 4:3
5. Resizes image to **800 × 600**
6. Saves into **scaled/** (or specified output directory)

---

## 🧪 Example

Input folder:

```
filtered/
    img1.jpg
    img2.png
    img3.avif
    doc.txt
```

After running:

```
scaled/
    img1.jpg         ← cropped & scaled
    img2.png         ← cropped & scaled
    img3.png         ← AVIF → PNG → cropped & scaled
```

Non-image file (`doc.txt`) is skipped.

---

## 🧩 Troubleshooting

### ❌ “Cannot write mode RGBA as JPEG”

Occurs when PNG/WEBP has transparency —
script auto-converts RGBA → RGB for JPG formats.

### ❌ Pillow cannot open some images

They may be corrupted or not real images.
These are automatically skipped.

---

## 📜 License

This project is for educational purposes as part of the Programming for ML course.

---

## Authors

Team: The Pynacle Group
Course: Programming for Machine Learning
Institution: IIT Bombay
